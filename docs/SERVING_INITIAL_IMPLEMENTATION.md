## Serving Initial Implementation (GemSpot)

This document captures the **serving-role deliverables** required for the
initial implementation milestone. It is designed so you can run the
measurements on Chameleon and fill in the results.

### 1) Shared Interface Samples (Joint Deliverable)

The shared JSON input/output samples live here:
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_input.json`
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_output.json`

These should match the agreed interface between data, training, and serving.

---

### 2) Serving Options Table (Measured Tradeoffs)

Fill in the table below using measured metrics **on Chameleon**.

| Option ID | Description | Endpoint | Model Version | Code Version | Hardware | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Error Rate | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| `baseline_http` | FastAPI + XGBoost (current) | `http://<ip>:8050/recommend` | `mock-v1.0` | `<git-sha>` | CPU |  |  |  |  |  | Baseline |
| `onnx_or_quantized` | XGBoost exported to ONNX or quantized model | `http://<ip>:8050/recommend` | `<model-id>` | `<git-sha>` | CPU |  |  |  |  |  | Model-level optimization |
| `batching` | Micro-batching requests | `http://<ip>:8050/recommend` | `<model-id>` | `<git-sha>` | CPU |  |  |  |  |  | System-level optimization |
| `infra_rightsized` | Same model, smaller/larger VM | `http://<ip>:8050/recommend` | `<model-id>` | `<git-sha>` | CPU/GPU |  |  |  |  |  | Infrastructure-level optimization |

**Minimum requirement**: Baseline + at least two optimized options
(model-level + system-level or infrastructure-level).

---

### 3) Right-Sizing Notes (Observed Usage)

Document resource usage for the most promising option:

- VM flavor: `<instance-type>`
- CPU usage under representative load: `<e.g., 35% avg, 65% peak>`
- RAM usage under representative load: `<e.g., 450MB avg, 900MB peak>`
- GPU usage (if applicable): `<utilization or N/A>`
- Conclusion: `<why this size is appropriate>`

---

### 4) Required Repository Artifacts (Already Present)

These satisfy the repo artifact requirement:
- Dockerfile: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/Dockerfile`
- Serving code: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/`
- Benchmark/load test scripts: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/`

---

### 5) Chameleon Run Checklist (For Measurement + Demo)

1. Start the ML serving container on Chameleon.
2. Run latency benchmark with `tests/benchmark_latency.py`.
3. Run load test with Locust to collect throughput/error rate.
4. Capture CPU/RAM usage during load.
5. Fill the table above and right-sizing notes.

---

### 6) Demo Video Checklist

The demo should show:
- Container startup on Chameleon
- `/health` response
- `/recommend` response for the shared input sample
- A short benchmark run (latency or load test)

