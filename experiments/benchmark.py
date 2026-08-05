"""
Load-test the nanoinference server.

Fires N requests at the /generate endpoint at a fixed concurrency, measures
client-side throughput and latency, then reads the server's own /stats.

Usage:
    venv/bin/python experiments/benchmark.py --requests 100 --concurrency 10

Compare batching effects by changing MAX_BATCH_SIZE in the server and re-running:
    # batch size 1 (batching effectively off) vs 8 (on)
"""

import time
import argparse
import asyncio
import statistics

import httpx


async def one_request(client, url, prompt, max_new_tokens, sem, latencies):
    """Send a single /generate request, recording its wall-clock latency."""
    async with sem:  # cap how many are in flight at once
        t0 = time.time()
        try:
            resp = await client.post(
                url,
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
                timeout=120.0,
            )
            resp.raise_for_status()
            latencies.append(time.time() - t0)
            return True
        except Exception as e:
            print(f"  request failed: {e}")
            return False


async def run_benchmark(args):
    gen_url = f"{args.base_url}/generate"
    stats_url = f"{args.base_url}/stats"

    latencies = []              # client-measured per-request latency (seconds)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        # fire all requests; the semaphore caps concurrency at args.concurrency
        wall_start = time.time()
        tasks = [
            one_request(client, gen_url, args.prompt, args.max_new_tokens, sem, latencies)
            for _ in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_total = time.time() - wall_start

        # pull the server's own view of what just happened
        try:
            server_stats = (await client.get(stats_url)).json()
        except Exception as e:
            server_stats = {"error": str(e)}

    ok = sum(results)
    print("\n=== CLIENT-SIDE ===")
    print(f"Requests:        {args.requests} (ok: {ok}, failed: {args.requests - ok})")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Total time:      {wall_total:.2f}s")
    print(f"Requests/sec:    {ok / wall_total:.1f}")
    if latencies:
        print(f"Avg latency:     {1000 * statistics.mean(latencies):.0f} ms")
        print(f"p95 latency:     {1000 * percentile(latencies, 95):.0f} ms")

    print("\n=== SERVER-SIDE (/stats, cumulative since server start) ===")
    for k, v in server_stats.items():
        print(f"{k:>18}: {v}")


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = min(int(len(s) * p / 100), len(s) - 1)
    return s[k]


def parse_args():
    ap = argparse.ArgumentParser(description="Load-test the nanoinference server.")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=100, help="total requests to send")
    ap.add_argument("--concurrency", type=int, default=10, help="max in-flight at once")
    ap.add_argument("--prompt", default="The future of AI is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
