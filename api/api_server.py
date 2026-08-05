import time
import uuid
import asyncio
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


app = FastAPI()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

MAX_BATCH_SIZE = 8

request_queue = asyncio.Queue()
results = {}

# Note: distilgpt2 had no pad token. GPT-2 family tokenizer ship without one, so padding=True will error. Thus we need to set a pad token
tokenizer.pad_token = tokenizer.eos_token
# Note: Need to do left padding. Causal model generate by continuing from the right end of each sequence.
tokenizer.padding_side = "left"

# Stats
requests = 0
total_tokens = 0
batch_sizes = []
inference_times = []
latencies = []



class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker_loop())


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": device,
        "queue_size": request_queue.qsize(),
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    global requests # We're using the global variable
    requests += 1
    request_id = str(uuid.uuid4())
    start_time = time.time()

    results[request_id] = {
        "status": "pending",
        "result": None,
    }

    #Place a new request in the queue
    await request_queue.put({
        "request_id": request_id,
        "prompt": req.prompt,
        "max_new_tokens": req.max_new_tokens,
        "start_time": start_time,
    })

    #If it's still generating a request, look for something else to do
    while results[request_id]["status"] != "done":
        await asyncio.sleep(0.01)

    result = results.pop(request_id)

    return result["result"]

async def worker_loop():
    global total_tokens
    while True:
        job = await request_queue.get()
        await asyncio.sleep(0.005)  # collection window

        batch = [job]
        while len(batch) < MAX_BATCH_SIZE:
            try:
                batch.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        batch_sizes.append(len(batch))

        prompts = [j["prompt"] for j in batch]
        # tokenizer returns a dict-like with "input_ids" and "attention_mask",
        # each a 2-D tensor of shape [batch_size, seq_len]. padding=True makes
        # every row the same length by adding pad tokens (on the left, per line 28).
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        max_new_tokens = max(j["max_new_tokens"] for j in batch)

        try:
            # model.generate() is a slow, synchronous (blocking) call. Wrap it in a
            # function to hand the function itself to asyncio.to_thread below.
            def run():
                with torch.no_grad():  # skip gradient tracking; we're only doing inference
                    # **inputs unpacks the dict into keyword args:
                    # input_ids=..., attention_mask=...
                    return model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            # Run the blocking call in a background thread so the async event loop
            # stays free to accept other requests / serve the health check.
            t0 = time.time()
            output = await asyncio.to_thread(run) #Sends run to a separate thread
            inference_times.append(time.time() - t0)

            # input_ids has shape [batch_size, seq_len]; shape[1] is seq_len, i.e.
            # how many (padded) prompt tokens precede the generated text in each row.
            input_len = inputs["input_ids"].shape[1]
            for i, j in enumerate(batch):
                text = tokenizer.decode(output[i][input_len:], skip_special_tokens=True)
                latency = time.time() - j["start_time"]   # compute once, per request
                latencies.append(latency)                 # store for p95
                results[j["request_id"]] = {
                    "status": "done",
                    "result": {
                        "request_id": j["request_id"],
                        "prompt": j["prompt"],
                        "response": text,
                        "time_seconds": round(latency, 3),  # ← reuse the same value
                        "device": device,
                    },
                }

            new_tokens = (output.shape[1] - input_len) * len(batch)
            total_tokens += new_tokens

        except Exception as e:
            for j in batch:
                results[j["request_id"]] = {
                    "status": "done",
                    "result": {"request_id": j["request_id"], "error": str(e)},
                }

        finally:
            for _ in batch:
                request_queue.task_done()

@app.post("/reset")
def reset():
    # Zero out all stats so each benchmark run starts from a clean slate.
    # Scalars are reassigned (need `global`); lists are cleared in place (don't).
    global requests, total_tokens
    requests = 0
    total_tokens = 0
    batch_sizes.clear()
    inference_times.clear()
    latencies.clear()
    return {"status": "reset"}


@app.get("/stats")
def stats():
    return {
        "total_requests": requests,
        "total_batches": len(batch_sizes),
        "avg_inference_ms": round(1000 * sum(inference_times)/len(inference_times), 1) if inference_times else 0,
        "avg_batch_size": round((sum(batch_sizes)/len(batch_sizes) if batch_sizes else 0), 3), #safeguards against empty list
        "tokens_per_sec": round(total_tokens / sum(inference_times), 1) if inference_times else 0,
        "p95_latency_ms": round(1000 * percentile(latencies, 95), 1)
    }

def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = min(int(len(s) * p / 100), len(s) - 1)
    return s[k]