import time
import asyncio
import torch
import json
import redis.asyncio as redis

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load in 4-bit so a 7B model fits in ~5GB of VRAM instead of ~14GB (fp16).
# nf4 = the quant type tuned for LLM weights; double-quant squeezes a bit more.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# NOTE: no .to(device) for quantized models, device_map places the weights

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda",
)
model.eval()

MAX_BATCH_SIZE = 8


# Note: distilgpt2 had no pad token. GPT-2 family tokenizer ship without one, so padding=True will error. Thus we need to set a pad token
tokenizer.pad_token = tokenizer.eos_token
# Note: Need to do left padding. Causal model generate by continuing from the right end of each sequence.
tokenizer.padding_side = "left"

async def worker_loop():
    while True:

        # BLPOP blocks until a job appears, then pops it from the left of the list
        # It returns a (key_name, value) tuple, so unpack and throw out the key
        _, job_json = await r.blpop("queue")
        job = json.loads(job_json) # JSON string -> dict
        
        await asyncio.sleep(0.005)  # collection window

        batch = [job]
        while len(batch) < MAX_BATCH_SIZE:
            job_json = await r.lpop("queue")
            if job_json is None:
                break
            batch.append(json.loads(job_json))

        await r.incr("stat:total_batches")
        await r.rpush("stat:batch_sizes", len(batch))

        # Wrap each prompt in the model's own chat format (e.g. Qwen's
        # <|im_start|>user ... <|im_start|>assistant) so it answers as an
        # assistant instead of just autocompleting text. The tokenizer knows the
        # correct markers for whatever model is loaded, so this survives model
        # swaps. Base models (like distilgpt2) have no chat template, so we guard
        # and fall back to the raw prompt.
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": j["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,  # end with the assistant cue
                )
                for j in batch
            ]
        else:
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
            await r.rpush("stat:inference_times", time.time() - t0)

            # input_ids has shape [batch_size, seq_len]; shape[1] is seq_len, i.e.
            # how many (padded) prompt tokens precede the generated text in each row.
            input_len = inputs["input_ids"].shape[1]

            for i, j in enumerate(batch):

                text = tokenizer.decode(output[i][input_len:], skip_special_tokens=True)
                latency = time.time() - j["start_time"]   # compute once, per request
                await r.rpush("stat:latencies", latency)                 # store for p95

                result_payload = {
                    "request_id": j["request_id"],
                    "prompt": j["prompt"],
                    "response": text,
                    "time_seconds": round(latency, 3),
                    "device": device,
                }

                # Set the result key. EX=60 auto deletes it after 60 seconds if the client
                # never collects it
                await r.set(f"result:{j['request_id']}", json.dumps(result_payload), ex=60)
                

            new_tokens = (output.shape[1] - input_len) * len(batch)
            await r.incrby("stat:total_tokens", new_tokens)

        except Exception as e:
            for j in batch:
                await r.set(
                    f"result:{j['request_id']}",
                    json.dumps({"request_id": j["request_id"], "error": str(e)}),
                    ex=60
                )

async def main():
    await r.set("model_name", MODEL_NAME)
    await r.set("device", device)
    await worker_loop()

if __name__ == "__main__":
    asyncio.run(main())
