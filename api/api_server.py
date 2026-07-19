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

request_queue = asyncio.Queue()
results = {}


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

        # Get the info the the job highest in the queue
        request_id = job["request_id"]
        prompt = job["prompt"]
        max_new_tokens = job["max_new_tokens"]
        start_time = job["start_time"]

        print(f"Worker processing request {request_id}")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = output[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)

            elapsed = time.time() - start_time

            results[request_id] = {
                "status": "done",
                "result": {
                    "request_id": request_id,
                    "prompt": prompt,
                    "response": text,
                    "time_seconds": round(elapsed, 3),
                    "device": device,
                },
            }

        except Exception as e: # Returns something so that generate can finish and process doesn't get frozen
            results[request_id] = {
                "status": "done",
                "result": {
                    "error": str(e),
                },
            }

        finally:
            request_queue.task_done()