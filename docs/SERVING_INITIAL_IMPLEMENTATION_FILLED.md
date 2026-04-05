## Serving Initial Implementation (GemSpot)

This document captures the current serving-role submission status for the
GemSpot initial implementation using measurements collected on Chameleon.

### 1) Shared Interface Samples (Joint Deliverable)

The shared JSON input/output samples live here:
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_input.json`
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_output.json`

These represent the single recommendation model served by the GemSpot ML
service. The serving experiments below compare different deployment and
runtime configurations for this same model interface.

### 2) Serving Options Table (Measured Tradeoffs)

Measured on Chameleon Cloud using floating IP `129.114.27.244`.

| Option ID | Description | Endpoint | Model Version | Code Version | Hardware | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Error Rate | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| `baseline_http` | FastAPI + XGBoost, 1 worker | `http://129.114.27.244:8050/recommend` | `mock-v1.0` | `182836a9` | CPU | 4.91 | 9.92 | 11.06 | 38.64 | 0 | Best throughput, simplest baseline |
| `xgboost_workers2` | FastAPI + XGBoost, 2 workers | `http://129.114.27.244:8051/recommend` | `mock-v1.0` | `182836a9` | CPU | 5.10 | 9.70 | 9.86 | 15.42 | 0 | Slightly better tail latency, much lower throughput |
| `onnx_cpu` | ONNX Runtime + FastAPI, 1 worker | `http://129.114.27.244:8052/recommend` | `mock-v1.0` | `182836a9` | CPU | 4.02 | 4.19 | 4.44 | 15.12 | 0 | Best latency, strongest model-level optimization |
| `onnx_workers4` | ONNX Runtime + FastAPI, 4 workers | `http://129.114.27.244:8053/recommend` | `mock-v1.0` | `182836a9` | CPU | 4.07 | 4.35 | 5.87 | 15.77 | 0 | Combined optimization, slightly better throughput than ONNX 1-worker |
| `infra_rightsized` | `onnx_cpu` rerun on second Chameleon VM flavor | `http://129.114.27.109:8052/recommend` | `mock-v1.0` | `182836a9` | CPU | 4.29 | 4.45 | 5.07 | 15.67 | 0 | Infrastructure comparison row; assuming second VM flavor was `m1.large` |

### 3) Interpretation And Recommendation

- Best throughput: `baseline_http`
- Best latency: `onnx_cpu`
- Best overall recommended deployment: `onnx_cpu`

Reasoning:
- `baseline_http` provides the highest throughput and the simplest deployment.
- `onnx_cpu` cuts latency substantially versus both XGBoost variants.
- `onnx_workers4` does not materially improve latency over `onnx_cpu`, and only
  slightly improves throughput, so `onnx_cpu` is the clearest recommended
  serving choice on the current VM flavor.
- `infra_rightsized` shows that moving the ONNX configuration to the second VM
  flavor did not materially improve latency or throughput, which supports
  keeping the smaller baseline CPU flavor as the better right-sized choice.

### 4) Right-Sizing Notes (Observed Usage)

The right-sizing section is now substantially complete, based on the second VM
comparison.

- Primary VM flavor: Chameleon baseline CPU flavor used for the first set of measurements
- Comparison VM flavor: assumed `m1.large` based on the second lease context
- CPU usage under representative load: collected during `docker stats`, `top`,
  and `free -h` runs on Chameleon
- RAM usage under representative load: collected during the same observation
- GPU usage: `N/A`
- Final recommendation: the ONNX configuration on the smaller CPU flavor is the
  better right-sized deployment because the larger VM did not produce a
  meaningful latency or throughput gain.

### 5) Repository Artifacts Used For These Rows

- Dockerfile: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/Dockerfile`
- Experiment compose file: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/docker-compose.experiments.yml`
- Serving code: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/main.py`
- Backend-aware model loader: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/model.py`
- ONNX export script: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/scripts/export_onnx_model.py`
- Latency benchmark: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/benchmark_latency.py`
- Load test: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/locustfile.py`
- Result summarizer: `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/summarize_serving_runs.py`

Row mapping:
- `baseline_http` -> service `ml-serving-baseline`
- `xgboost_workers2` -> service `ml-serving-workers2`
- `onnx_cpu` -> service `ml-serving-onnx`
- `onnx_workers4` -> service `ml-serving-onnx-workers4`

### 6) Submission Status

Ready now:
- Q1 shared JSON samples
- Q2.2 repository artifacts
- Q2.1 serving options table draft with infrastructure comparison included

Still pending before a fully complete final submission:
- finalized CPU/RAM right-sizing note using the collected screenshots
- Q2.3 demo video recorded on Chameleon
