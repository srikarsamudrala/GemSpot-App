## Chameleon Serving Runbook (GemSpot)

Use this checklist to run the serving subsystem on Chameleon and collect
measurements for the initial implementation deliverables.

### A) Provision Infrastructure

1. Create or use your Chameleon project.
2. Launch a compute instance (CPU is fine for baseline).
3. Add a floating IP and allow inbound traffic on ports:
   - `8050` (ML serving)
   - `8015` (frontend, optional demo)
   - `8016` (backend, optional demo)
   - `8089` (Locust UI, optional)
4. Name resources with your project ID suffix (course requirement).

### B) Install Dependencies on the VM

```bash
sudo apt-get update
sudo apt-get install -y git docker.io docker-compose-plugin
sudo usermod -aG docker $USER
```

Log out and back in so Docker group changes take effect.

### C) Run the Application

```bash
git clone <your-repo-url>
cd AdventureLog
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d --build
```

Verify serving health:
```bash
curl http://<floating-ip>:8050/health
```

### D) Run Benchmarks (for the options table)

Latency benchmark:
```bash
cd ml-serving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/benchmark_latency.py --rounds 50
```

Load test (headless):
```bash
locust -f tests/locustfile.py --host http://<floating-ip>:8050 \
  --headless -u 50 -r 10 --run-time 60s
```

Collect CPU/RAM usage while Locust runs:
```bash
top -o %CPU
```

### E) Populate Deliverables

1. Fill `/Users/srikarsamudrala/Documents/AdventureLog/docs/SERVING_INITIAL_IMPLEMENTATION.md`
2. Update the options table with measured values.
3. Update right-sizing notes with observed CPU/RAM.
