## GemSpot Serving Submission Playbook

This guide is written in the same spirit as the Chameleon lab handouts:
start from a fresh Chameleon instance, run the serving experiments there,
collect the exact artifacts Gradescope asks for, and then submit them.

### 1. What You Are Submitting

For the serving-role initial implementation, your submission should cover:

1. Joint JSON samples
2. A serving-options table in PDF form
3. Repository artifacts that produced those table rows
4. A short demo video showing the best option on Chameleon

The baseline service already exists in this repository. The stronger
submission path adds multiple measured serving variants:

- `baseline_http`: XGBoost + FastAPI, 1 worker
- `xgboost_workers2`: XGBoost + FastAPI, 2 workers
- `onnx_cpu`: ONNX Runtime + FastAPI, 1 worker
- `onnx_workers4`: ONNX Runtime + FastAPI, 4 workers
- `triton_batch`: Triton Inference Server + FastAPI proxy
- `infra_rightsized`: rerun the best configuration on a second VM size

### 2. Files That Matter

Core serving artifacts:

- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/Dockerfile`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/docker-compose.experiments.yml`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/main.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/model.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/scripts/export_onnx_model.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/triton/model_repository/gemspot_onnx/config.pbtxt`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/benchmark_latency.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/locustfile.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/summarize_serving_runs.py`

Joint sample files:

- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_input.json`
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_output.json`

Submission templates:

- `/Users/srikarsamudrala/Documents/AdventureLog/docs/SERVING_INITIAL_IMPLEMENTATION.md`
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/MODEL_SERVING.md`

### 3. Provision Chameleon Resources

Create a VM instance and assign a floating IP. After that, make sure the
instance allows inbound traffic on these ports:

- `22` for SSH
- `8050` for baseline serving
- `8051` for 2-worker XGBoost
- `8052` for ONNX serving
- `8053` for ONNX + 4 workers
- `8054` for Triton-backed FastAPI proxy
- `8060` for Triton HTTP
- `8089` for the Locust UI, if you want to show it in the demo

You can use one VM for the first five rows, then launch a second VM flavor
for the infrastructure comparison row.

### 4. SSH Into The VM

From your local terminal:

```bash
ssh -i ~/.ssh/<your-chameleon-key> cc@<floating-ip>
```

Everything below is run inside that SSH session unless explicitly stated.

### 5. Install Docker And Clone The Repo

```bash
sudo apt-get update
sudo apt-get install -y git docker.io docker-compose-plugin
sudo usermod -aG docker $USER
exit
```

Reconnect over SSH, then:

```bash
git clone <your-repo-url>
cd AdventureLog/ml-serving
```

### 6. Build And Start Experiment Variants

Run all serving variants:

```bash
docker compose -f docker-compose.experiments.yml up -d --build
```

Check they are healthy:

```bash
curl http://<floating-ip>:8050/health
curl http://<floating-ip>:8051/health
curl http://<floating-ip>:8052/health
curl http://<floating-ip>:8053/health
curl http://<floating-ip>:8054/health
```

Check the backend for each:

```bash
curl http://<floating-ip>:8050/model/info
curl http://<floating-ip>:8052/model/info
curl http://<floating-ip>:8054/model/info
curl http://<floating-ip>:8060/v2/models/gemspot_onnx
```

You should see `serving_backend` report `xgboost` for `8050` and `onnx`
for `8052`, and `triton` for `8054`.

### 7. Run Benchmarks For Each Row

Install test dependencies once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

Run latency and load tests for each configuration.

Baseline:

```bash
python tests/benchmark_latency.py --url http://<floating-ip>:8050 --rounds 50
mv benchmark_results.json baseline_benchmark.json

locust -f tests/locustfile.py --host http://<floating-ip>:8050 \
  --headless -u 50 -r 10 --run-time 60s --csv baseline

python tests/summarize_serving_runs.py \
  --label baseline_http \
  --benchmark baseline_benchmark.json \
  --locust-csv-prefix baseline \
  --output baseline_summary.json
```

2-worker XGBoost:

```bash
python tests/benchmark_latency.py --url http://<floating-ip>:8051 --rounds 50
mv benchmark_results.json workers2_benchmark.json

locust -f tests/locustfile.py --host http://<floating-ip>:8051 \
  --headless -u 50 -r 10 --run-time 60s --csv workers2

python tests/summarize_serving_runs.py \
  --label xgboost_workers2 \
  --benchmark workers2_benchmark.json \
  --locust-csv-prefix workers2 \
  --output workers2_summary.json
```

ONNX:

```bash
python tests/benchmark_latency.py --url http://<floating-ip>:8052 --rounds 50
mv benchmark_results.json onnx_benchmark.json

locust -f tests/locustfile.py --host http://<floating-ip>:8052 \
  --headless -u 50 -r 10 --run-time 60s --csv onnx

python tests/summarize_serving_runs.py \
  --label onnx_cpu \
  --benchmark onnx_benchmark.json \
  --locust-csv-prefix onnx \
  --output onnx_summary.json
```

ONNX + 4 workers:

```bash
python tests/benchmark_latency.py --url http://<floating-ip>:8053 --rounds 50
mv benchmark_results.json onnx_workers4_benchmark.json

locust -f tests/locustfile.py --host http://<floating-ip>:8053 \
  --headless -u 50 -r 10 --run-time 60s --csv onnx_workers4

python tests/summarize_serving_runs.py \
  --label onnx_workers4 \
  --benchmark onnx_workers4_benchmark.json \
  --locust-csv-prefix onnx_workers4 \
  --output onnx_workers4_summary.json
```

Triton-backed serving:

```bash
python tests/benchmark_latency.py --url http://<floating-ip>:8054 --rounds 50
mv benchmark_results.json triton_benchmark.json

locust -f tests/locustfile.py --host http://<floating-ip>:8054 \
  --headless -u 50 -r 10 --run-time 60s --csv triton

python tests/summarize_serving_runs.py \
  --label triton_batch \
  --benchmark triton_benchmark.json \
  --locust-csv-prefix triton \
  --output triton_summary.json
```

### 8. Capture Right-Sizing Evidence

While a load test is running, record observed resource usage:

```bash
docker stats --no-stream
top -o %CPU
free -h
```

Write down:

- VM flavor
- CPU usage under load
- memory usage under load
- whether latency/throughput improved enough to justify the cost

Repeat the best-performing configuration on a different Chameleon VM size
to produce the `infra_rightsized` row.

### 9. Fill The Gradescope Table

Open:

- `/Users/srikarsamudrala/Documents/AdventureLog/docs/SERVING_INITIAL_IMPLEMENTATION.md`

For each row, transfer values from the corresponding `*_summary.json` file,
plus the VM flavor and right-sizing notes from your Chameleon run.

Suggested row mapping:

- `baseline_http` -> port `8050`
- `xgboost_workers2` -> port `8051`
- `onnx_cpu` -> port `8052`
- `onnx_workers4` -> port `8053`
- `triton_batch` -> port `8054`
- `infra_rightsized` -> rerun best row on another instance type

Then export the filled markdown/table to PDF for `Q2.1`.

### 10. What To Upload For Each Gradescope Question

`Q1 Joint responsibilities`

Upload:

- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_input.json`
- `/Users/srikarsamudrala/Documents/AdventureLog/docs/samples/serving_output.json`

If you are on a 4-person team, also upload the container table PDF from your
team.

`Q2.1 Serving options table`

Upload the PDF exported from your filled
`SERVING_INITIAL_IMPLEMENTATION.md`.

`Q2.2 Repository artifacts`

Upload the files that produced your table rows:

- `ml-serving/Dockerfile`
- `ml-serving/docker-compose.experiments.yml`
- `ml-serving/app/main.py`
- `ml-serving/app/model.py`
- `ml-serving/scripts/export_onnx_model.py`
- `ml-serving/triton/model_repository/gemspot_onnx/config.pbtxt`
- `ml-serving/tests/benchmark_latency.py`
- `ml-serving/tests/locustfile.py`
- `ml-serving/tests/summarize_serving_runs.py`

Use the text box to explain:

- baseline row came from `docker-compose.experiments.yml` service
  `ml-serving-baseline`
- XGBoost worker row came from `ml-serving-workers2`
- ONNX row came from `ml-serving-onnx`
- ONNX worker row came from `ml-serving-onnx-workers4`
- Triton row came from `triton-server` plus `ml-serving-triton`
- latency came from `benchmark_latency.py`
- throughput/error-rate came from `locustfile.py`
- summary values came from `summarize_serving_runs.py`

`Q2.3 Demo video`

Record a short Chameleon demo showing:

1. SSH session on the VM
2. `docker compose -f docker-compose.experiments.yml up -d --build`
3. `curl http://<floating-ip>:8054/health`
4. `curl http://<floating-ip>:8054/model/info`
5. `curl -X POST http://<floating-ip>:8054/recommend -H "Content-Type: application/json" -d @/path/to/serving_input.json`
6. one short benchmark or Locust run

### 11. Recommended Story For The Report

Use language like this:

- `baseline_http` is the simplest deployment and the control configuration
- `xgboost_workers2` tests system-level concurrency scaling
- `onnx_cpu` tests model-level optimization by switching to an optimized
  artifact/runtime
- `onnx_workers4` tests combined model-level and system-level optimization
- `triton_batch` tests a dedicated serving framework with dynamic batching
- `infra_rightsized` tests infrastructure-level optimization by changing VM
  size while keeping the serving stack constant

### 12. Important Grading Risk

Gradescope explicitly says local-only work does not count. That means the code
added in this repo is only half of the job. To turn it into submission credit,
you still need to run the configurations on Chameleon, collect the numbers
there, and submit those results.
