"""
Load-test the nanoinference server.

Fires N requests at the /generate endpoint at a fixed concurrency, measures
client-side throughput and latency, then reads the server's own /stats.

In plain terms: this script pretends to be a crowd of users all hitting the
server at once. Batching only helps when requests arrive together, so a
single curl at a time would never show the effect; this does.

Usage:
    venv/bin/python experiments/benchmark.py --requests 100 --concurrency 10

Compare batching effects by changing MAX_BATCH_SIZE in the server and re-running:
    # batch size 1 (batching effectively off) vs 8 (on)
"""

import time
import argparse
import asyncio
import statistics

# httpx is like the `requests` library, but async: it can have many HTTP
# requests in flight at the same time instead of waiting for each one.
import httpx


async def one_request(client, url, prompt, max_new_tokens, sem, latencies):
    """Send a single /generate request, recording its wall-clock latency."""
    # One call of this function = one simulated user.

    # The semaphore is like a bouncer: only `concurrency` users can be inside
    # (talking to the server) at once. Everyone else waits here for a slot.
    async with sem:
        t0 = time.time()  # start this user's stopwatch
        try:
            # Send the prompt and wait for the generated answer.
            resp = await client.post(
                url,
                json={"prompt": prompt, "max_new_tokens": max_new_tokens},
                timeout=120.0,  # give up on this request after 2 minutes
            )
            resp.raise_for_status()  # treat HTTP errors (4xx/5xx) as failures
            latencies.append(time.time() - t0)  # how long this user waited
            return True   # success -> counted in the "ok" total
        except Exception as e:
            print(f"  request failed: {e}")
            return False  # failure -> counted in the "failed" total


async def run_benchmark(args):
    gen_url = f"{args.base_url}/generate"
    stats_url = f"{args.base_url}/stats"

    latencies = []              # client-measured per-request latency (seconds)
    sem = asyncio.Semaphore(args.concurrency)  # the "bouncer", see one_request

    # One shared HTTP client for all requests (reuses connections).
    async with httpx.AsyncClient() as client:
        wall_start = time.time()  # stopwatch for the whole run

        # Build one task per simulated user. Nothing is sent yet; these are
        # just lined up, ready to go.
        tasks = [
            one_request(client, gen_url, args.prompt, args.max_new_tokens, sem, latencies)
            for _ in range(args.requests)
        ]

        # Start ALL tasks at once and wait until every one has finished.
        # The semaphore keeps only `concurrency` of them actually in flight;
        # as one finishes, the next waiting one goes in. Because ~10 requests
        # are always waiting on the server, the worker can batch them.
        results = await asyncio.gather(*tasks)  # list of True/False per request
        wall_total = time.time() - wall_start

        # Ask the server for ITS view of what just happened (batch sizes,
        # tokens/sec, etc.) so we can compare it with what the client saw.
        try:
            server_stats = (await client.get(stats_url)).json()
        except Exception as e:
            server_stats = {"error": str(e)}

    # ---- Report ----
    ok = sum(results)  # True counts as 1, so this = number of successes
    print("\n=== CLIENT-SIDE ===")
    print(f"Requests:        {args.requests} (ok: {ok}, failed: {args.requests - ok})")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Total time:      {wall_total:.2f}s")
    # Throughput: how many requests the server completed per second overall.
    print(f"Requests/sec:    {ok / wall_total:.1f}")
    if latencies:
        # Latency: how long an individual user waited for their answer.
        print(f"Avg latency:     {1000 * statistics.mean(latencies):.0f} ms")
        print(f"p95 latency:     {1000 * percentile(latencies, 95):.0f} ms")

    # NOTE: server stats are cumulative since the server started (or since the
    # last POST /reset), so reset between runs or the numbers will mix together.
    print("\n=== SERVER-SIDE (/stats, cumulative since server start) ===")
    for k, v in server_stats.items():
        print(f"{k:>18}: {v}")


def percentile(data, p):
    """Return the p-th percentile of data.

    p95 = "95% of requests finished in this time or less". It shows the
    worst case most users actually see, which an average can hide.
    """
    if not data:
        return 0
    s = sorted(data)                           # fastest -> slowest
    k = min(int(len(s) * p / 100), len(s) - 1) # index 95% of the way up
    return s[k]


def parse_args():
    # Command-line options, e.g. --requests 40 --concurrency 10
    ap = argparse.ArgumentParser(description="Load-test the nanoinference server.")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=100, help="total requests to send")
    ap.add_argument("--concurrency", type=int, default=10, help="max in-flight at once")
    ap.add_argument("--prompt", default="The future of AI is")  # same prompt for every request
    ap.add_argument("--max-new-tokens", type=int, default=20)   # length of each answer
    return ap.parse_args()


# Runs only when the file is executed directly (not when imported):
# read the flags, then start the async benchmark.
if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
