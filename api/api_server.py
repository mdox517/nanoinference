import time
import uuid
import asyncio
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


app = FastAPI()

MODEL_NAME = "distilgpt2"
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
    while True:
        job = await request_queue.get()
        await asyncio.sleep(0.005)  # collection window

        batch = [job]
        while len(batch) < MAX_BATCH_SIZE:
            try:
                batch.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        prompts = [j["prompt"] for j in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        max_new_tokens = max(j["max_new_tokens"] for j in batch)

        try:
            def run():
                with torch.no_grad():
                    return model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            output = await asyncio.to_thread(run)

            input_len = inputs["input_ids"].shape[1]
            for i, j in enumerate(batch):
                text = tokenizer.decode(output[i][input_len:], skip_special_tokens=True)
                results[j["request_id"]] = {
                    "status": "done",
                    "result": {
                        "request_id": j["request_id"],
                        "prompt": j["prompt"],
                        "response": text,
                        "time_seconds": round(time.time() - j["start_time"], 3),
                        "device": device,
                    },
                }

        except Exception as e:
            for j in batch:
                results[j["request_id"]] = {
                    "status": "done",
                    "result": {"request_id": j["request_id"], "error": str(e)},
                }

        finally:
            for _ in batch:
                request_queue.task_done()