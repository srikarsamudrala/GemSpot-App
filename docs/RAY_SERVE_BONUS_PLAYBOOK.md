## Ray Serve Bonus Playbook

Use this only after the main serving submission is stable.

### Goal

Run a serving framework not used in the lab by serving the GemSpot ONNX model
through Ray Serve on Chameleon, then benchmark it as an extra comparison row.

### Files

- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/ray_serve_app.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/docker-compose.experiments.yml`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/benchmark_latency.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/locustfile.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/tests/summarize_serving_runs.py`

### Ports

- `8055` for the Ray Serve HTTP endpoint

### Chameleon Commands

SSH to the VM:

```bash
ssh -i ~/.ssh/<your-key> cc@<floating-ip>
```

Open the project:

```bash
cd ~/GemSpot-App
git pull
cd ml-serving
```

Rebuild and start the Ray Serve row:

```bash
docker compose -f docker-compose.experiments.yml up -d --build ml-serving-rayserve
```

Check it:

```bash
docker ps
curl http://<floating-ip>:8055/health
curl http://<floating-ip>:8055/model/info
curl -X POST http://<floating-ip>:8055/recommend \
  -H "Content-Type: application/json" \
  -d @/home/cc/GemSpot-App/docs/samples/serving_input.json
```

Benchmark it:

```bash
source .venv/bin/activate

python tests/benchmark_latency.py --url http://<floating-ip>:8055 --rounds 50
mv benchmark_results.json rayserve_benchmark.json

locust -f tests/locustfile.py --host http://<floating-ip>:8055 \
  --headless -u 20 -r 5 --run-time 30s --csv rayserve

python tests/summarize_serving_runs.py \
  --label ray_serve \
  --benchmark rayserve_benchmark.json \
  --locust-csv-prefix rayserve \
  --output rayserve_summary.json

cat rayserve_summary.json
```

### What To Submit If It Works

If the run works and you want to claim the bonus, include:

- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/app/ray_serve_app.py`
- `/Users/srikarsamudrala/Documents/AdventureLog/ml-serving/docker-compose.experiments.yml`
- benchmark outputs generated from the Chameleon run

Suggested explanation:

Ray Serve was added as a non-lab serving framework to evaluate whether the same
GemSpot ONNX model could be served through a more deployment-oriented serving
stack. It preserves the same `/recommend` contract while enabling a framework
designed for scalable multi-replica and multi-model serving.
