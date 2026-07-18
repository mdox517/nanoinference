# Mini Distributed Inference Engine

A learning-focused distributed inference engine for serving language models across one or more GPU workers.

The goal of this project is to understand how modern ML inference systems work under the hood: request queues, batching, scheduling, GPU utilization, worker health checks, latency/throughput tradeoffs, and eventually distributed execution.


---

## Project Goals

* GPU inference performance
* Dynamic batching
* Request scheduling
* Distributed worker coordination
* Queue-based system design
* Backpressure and retries
* Latency and throughput measurement
* GPU memory and utilization monitoring
* Practical ML infrastructure design

---

## Initial Architecture

```text
Client
  |
  v
API Server
  |
  v
Request Queue
  |
  v
GPU Worker
  |
  v
Model Inference
  |
  v
Response
```

Later versions will expand toward:

```text
Client
  |
  v
API Server
  |
  v
Scheduler
  |
  +-------------------+-------------------+
  |                   |                   |
  v                   v                   v
GPU Worker 1     GPU Worker 2       GPU Worker 3
```

---

## Tech Stack

Initial stack:

* Python
* FastAPI
* PyTorch
* Hugging Face Transformers
* asyncio
* Redis, later
* Docker, later
* Prometheus/Grafana, later

Possible future stack:

* Go scheduler or API gateway
* gRPC
* CUDA kernels
* Kubernetes
* NVIDIA Nsight
* Triton-compatible model workers

---

## Milestones

### Phase 1: Basic Inference API

Build a simple FastAPI server with a `/generate` endpoint.

```text
Client -> FastAPI -> Model -> Response
```

Requirements:

* Load a small Hugging Face causal language model
* Accept a prompt
* Generate text
* Return generated output as JSON
* Run on GPU if CUDA is available

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_new_tokens": 50}'
```

---

### Phase 2: Worker Queue

Move model inference out of the API route.

```text
Client -> API Server -> Queue -> GPU Worker -> Response
```

Requirements:

* API receives request
* API assigns request ID
* Request enters queue
* GPU worker consumes queue
* Worker runs inference
* Result is returned to client


---

### Phase 3: Dynamic Batching

Allow the GPU worker to process multiple requests together.

```text
Request A
Request B
Request C
    |
    v
Batched GPU Inference
```

Requirements:

* Worker waits briefly for more requests
* Combine requests into a batch
* Tokenize together
* Run one batched model call
* Split outputs back to individual users

Metrics to track:

* Batch size
* Average latency
* p95 latency
* Requests per second
* Tokens per second
* GPU memory usage
* GPU utilization

---

### Phase 4: Separate API and Worker Processes

Run API server and worker as independent processes.

Possible communication methods:

* Redis queue
* HTTP
* gRPC

Initial recommendation: Redis.

```text
API Server -> Redis -> GPU Worker
```

Requirements:

* Start API server separately
* Start worker separately
* Submit jobs through Redis
* Worker writes results back
* API retrieves result for client

---

### Phase 5: Multiple Workers

Add support for multiple workers.

```text
API Server
  |
  v
Scheduler
  |
  +--> GPU Worker 1
  +--> GPU Worker 2
  +--> GPU Worker 3
```

Requirements:

* Worker registration
* Worker heartbeats
* Queue length tracking
* Basic load balancing
* Failed worker detection
* Retry failed requests

Scheduling strategies to test:

* Round robin
* Least queue depth
* Most available GPU memory
* Fastest recent latency

---

### Phase 6: Observability

Add metrics and logging.

Track:

* Request latency
* Queue wait time
* Inference time
* Tokens generated per second
* Batch size
* GPU memory usage
* GPU utilization
* Worker failures
* Error rate

Possible tools:

* Prometheus
* Grafana
* OpenTelemetry
* Structured JSON logs

---

### Phase 7: GPU Architecture Experiments

Use the system to study GPU behavior.

Experiments:

* Batch size 1 vs 2 vs 4 vs 8 vs 16
* FP32 vs FP16 inference
* CPU vs GPU inference
* Small model vs larger model
* Short prompt vs long prompt
* Short generation vs long generation
* Sequential requests vs batched requests

Questions to answer:

* When does batching improve throughput?
* When does batching hurt latency?
* How much VRAM does the model use?
* How does sequence length affect inference time?
* Is the GPU compute-bound or memory-bound?
* How much time is spent tokenizing on CPU?
* How much time is spent generating on GPU?

---

## Project Structure

```text
nanoinference/
  README.md
  requirements.txt

  api/
    server.py

  worker/
    gpu_worker.py

  scheduler/
    scheduler.py

  common/
    types.py
    config.py
    metrics.py
    logging.py

  experiments/
    benchmark.py
    batch_size_experiment.py
    latency_test.py

  scripts/
    run_api.sh
    run_worker.sh

  docs/
    architecture.md
    gpu_notes.md
    distributed_systems_notes.md
```

---

## Core Concepts to Learn

### GPU Concepts

* CUDA
* VRAM
* Tensor cores
* GPU memory bandwidth
* Kernel launch overhead
* CPU-GPU transfer overhead
* Batch size
* Mixed precision
* Quantization
* GPU utilization
* Model memory footprint

### Distributed Systems Concepts

* Queues
* Schedulers
* Load balancing
* Heartbeats
* Retries
* Timeouts
* Backpressure
* Worker failure
* Horizontal scaling
* Observability
* Fault tolerance

### ML Infrastructure Concepts

* Model serving
* Inference latency
* Throughput
* Dynamic batching
* Tokenization bottlenecks
* Model warmup
* Cold starts
* KV cache
* Request streaming
* Autoscaling

---

## Useful End Goal

A useful final version of this project would be:

> A local LLM inference gateway that accepts generation requests, batches them efficiently, routes them to available GPU workers, monitors worker health, exposes latency and GPU metrics, and provides a simple API for applications to use.

Possible real uses:

* Private local chatbot backend
* Research lab model-serving tool
* Internal document assistant backend
* GPU benchmarking framework
* ML systems portfolio project
* Educational replacement for black-box inference APIs

---

## Non-Goals

This project is not trying to:

* Beat vLLM
* Implement full CUDA kernels from scratch
* Train models
* Support massive production traffic
* Build a polished frontend
* Support every Hugging Face model
* Implement Kubernetes-level orchestration

First goal is to understand how these systems work, then worry about performance

---

## First Implementation Target

Build the simplest possible version first:

```text
POST /generate
```

Input:

```json
{
  "prompt": "The future of AI is",
  "max_new_tokens": 50
}
```

Output:

```json
{
  "text": "The future of AI is ..."
}
```

Once this works, move inference into a worker queue.

---

## Development Philosophy

Build in layers

1. Make inference work
2. Add queue
3. Add batching
4. Add metrics
5. Split API and worker
6. Add multiple workers
7. Add scheduling
8. Add failure handling
9. Add GPU profiling
10. Optimize

Every phase should produce a working system

---

## Benchmarking Ideas

Example benchmark command:

```bash
python experiments/benchmark.py \
  --url http://localhost:8000/generate \
  --requests 100 \
  --concurrency 10
```

Metrics to report:

```text
Total requests: 100
Successful requests: 100
Failed requests: 0
Average latency: 420 ms
p95 latency: 880 ms
Requests/sec: 18.4
Tokens/sec: 730
Average batch size: 5.2
```

---

## Long-Term Ideas

Advanced features:

* Streaming token responses
* KV cache reuse
* Continuous batching
* Model sharding
* Multi-GPU tensor parallelism
* Autoscaling workers
* Priority queues
* Request cancellation
* Rate limiting
* Authentication
* Docker Compose deployment
* Kubernetes deployment
* Go-based scheduler
* Custom CUDA matrix multiplication kernel
* Triton backend integration

---

## Recommended Small Models

Good starter models:

* `distilgpt2`
* `gpt2`
* `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* `Qwen/Qwen2.5-0.5B-Instruct`

Start small. Larger models can come later.

---

## Success Criteria

* A working inference API
* Separate GPU worker process
* Batched inference
* Basic scheduling
* Measured latency and throughput
* GPU utilization tracking
* Clear documentation of design tradeoffs
* Benchmarks showing how batching affects performance

The final result should be understandable, runnable, and useful as a learning artifact for GPU systems and distributed ML infrastructure.
