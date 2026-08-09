import time
import uuid
import asyncio
import json
import redis.asyncio as redis

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
r = redis.Redis(host="localhost", port=6379, decode_responses=True)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


@app.get("/")
async def health_check():          # note: now async
    return {
        "status": "ok",
        "model": await r.get("model_name"),
        "device": await r.get("device"),
        "queue_size": await r.llen("queue"),
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    await r.incr("stat:requests")
    request_id = str(uuid.uuid4())
    start_time = time.time()


    job = {
        "request_id": request_id,
        "prompt": req.prompt,
        "max_new_tokens": req.max_new_tokens,
        "start_time": start_time,
    }

    # Push the job onto the Redis queue (a list). rpush adds to the right end;
    # the worker will pop from the left end -> FIFO
    await r.rpush("queue", json.dumps(job))

    # Poll Redis for this request's result. The worker writes a key named
    # result:<id> when it's done. Until then, get() returns None
    while True:
        result = await r.get(f"result:{request_id}")
        if result is not None:
            break
        await asyncio.sleep(0.01)

    await r.delete(f"result:{request_id}") # clean up the key
    return json.loads(result)
    


@app.post("/reset")
async def reset():
    await r.delete(
        "stat:requests", "stat:total_tokens", "stat:total_batches",
        "stat:inference_times", "stat:latencies", "stat:batch_sizes",
    )
    return {"status": "reset"}



@app.get("/stats")
async def stats():
    requests = int(await r.get("stat:requests") or 0)
    total_tokens = int(await r.get("stat:total_tokens") or 0)
    total_batches = int(await r.get("stat:total_batches") or 0)
    inference_times = [float(x) for x in await r.lrange("stat:inference_times", 0, -1)]
    latencies = [float(x) for x in await r.lrange("stat:latencies", 0, -1)]
    batch_sizes = [float(x) for x in await r.lrange("stat:batch_sizes", 0, -1)]
    return {
        "total_requests": requests,
        "total_batches": total_batches,
        "avg_inference_ms": round(1000 * sum(inference_times)/len(inference_times), 1) if inference_times else 0,
        "avg_batch_size": round(sum(batch_sizes)/len(batch_sizes), 3) if batch_sizes else 0,
        "tokens_per_sec": round(total_tokens / sum(inference_times), 1) if inference_times else 0,
        "p95_latency_ms": round(1000 * percentile(latencies, 95), 1),
    }


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = min(int(len(s) * p / 100), len(s) - 1)
    return s[k]