# GemSpot Model Serving — Technical Documentation

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Model Serving Pipeline](#2-model-serving-pipeline)
3. [API Reference](#3-api-reference)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Live Demo Guide](#5-live-demo-guide)
6. [Next Steps](#6-next-steps)

---

## 1. Architecture Overview

GemSpot uses a **microservice architecture** with a two-stage recommendation pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Request                              │
│              "Find hidden gems near Central Park"                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Candidate Generation (Django + PostGIS)               │
│  ──────────────────────────────────────────────────              │
│  • User provides lat/lon + radius                               │
│  • PostGIS executes spatial bounding-box query                  │
│  • Returns N candidate destinations within radius               │
│  • Extracts features: category, rating, reviews, price          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ POST /recommend
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: ML Scoring (FastAPI + XGBoost)                        │
│  ──────────────────────────────────────────────────              │
│  • Receives candidate list + user context                       │
│  • Assembles 25-dimensional feature vector per candidate        │
│  • Runs XGBoost binary classifier inference                     │
│  • Returns scored "Gem Cards" sorted by probability             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend Display (SvelteKit)                                    │
│  ──────────────────────────────────────────────────              │
│  • Renders ranked Gem Cards with confidence scores               │
│  • Shows vibe tags, distance, star ratings                       │
│  • Map view with recommendation markers                          │
│  • "Add to Collection" action per card                           │
└─────────────────────────────────────────────────────────────────┘
```

### Service Map

| Service | Technology | Port | Role |
|---------|-----------|------|------|
| `gemspot-frontend` | SvelteKit + DaisyUI | 8015 | Web UI |
| `gemspot-backend` | Django + DRF + PostGIS | 8016 | API + Candidate Generation |
| `gemspot-ml-serving` | FastAPI + XGBoost | 8050 | ML Inference |
| `gemspot-db` | PostgreSQL + PostGIS | 5432 | Spatial Database |
| `gemspot-redis` | Redis 7 | 6379 | Feature Cache (future) |

---

## 2. Model Serving Pipeline

### 2.1 Feature Engineering

Each candidate destination is encoded into a **25-dimensional feature vector**:

```
Index  0:       category_encoded    (int → float, 15 categories)
Index  1:       avg_rating          (1.0 – 5.0)
Index  2:       num_reviews         (0 – 100,000+)
Index  3:       price               (0 – 4 scale)
Index  4:       user_total_visits   (historical visit count)
Index  5-14:    vibe_tags           (10-dim binary vector, 3-hot encoding)
Index  15-24:   user_preferences    (10-dim preference weights)
```

**Vibe Tag Vocabulary** (10 tags):
```
scenic | relaxing | adventurous | cultural | romantic
family-friendly | nightlife | foodie | historic | trendy
```

Each destination gets 3 vibe tags encoded as a 3-hot binary vector.
User preferences are a 10-dimensional float vector of historical affinity per tag.

### 2.2 XGBoost Model

| Parameter | Value |
|-----------|-------|
| **Objective** | `binary:logistic` (probability of user visiting) |
| **Max Depth** | 4 |
| **Learning Rate (η)** | 0.1 |
| **Subsample** | 0.8 |
| **Col Sample by Tree** | 0.8 |
| **Boosting Rounds** | 100 |
| **Output** | Probability score (0.0 – 1.0) |

**Current Model**: A mock/synthetic model trained on 5,000 generated samples. It produces plausible scores based on patterns like:
- Higher ratings → higher scores
- More reviews → higher scores  
- Vibe tag overlap with user preferences → higher scores

### 2.3 Scoring & Output

The model outputs a probability per candidate. Results are:
1. Sorted descending by score
2. Formatted as **Gem Cards** with:
   - `score` (0.0 – 1.0)
   - `confidence_pct` (e.g., "91%")
   - `vibe_tags` (top 3 tags)
   - `distance_km` (haversine distance from user)

### 2.4 Fallback Behavior

If the ML model file is missing or fails to load:
- The service stays healthy (returns 200 on `/health`)
- Inference falls back to **random scoring** (uniform 0.3–0.95)
- `model_loaded` flag in health response will be `false`
- Allows the app to function during development without a trained model

---

## 3. API Reference

### `GET /health`
Health check for monitoring and readiness probes.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "mock-v1.0"
}
```

### `POST /recommend`
Core inference endpoint. Scores candidates and returns ranked Gem Cards.

**Request:**
```json
{
  "user_id": "user_abc123",
  "latitude": 40.7128,
  "longitude": -74.006,
  "candidates": [
    {
      "gmap_id": "ChIJ...",
      "name": "Central Park",
      "latitude": 40.7829,
      "longitude": -73.9654,
      "category": "park",
      "avg_rating": 4.8,
      "num_reviews": 50000,
      "price": 0,
      "vibe_tags": ["scenic", "relaxing", "family-friendly"]
    }
  ],
  "top_k": 10,
  "user_total_visits": 5,
  "user_personal_preferences": [10.0, 5.0, 0.0, 8.0, 0.0, 3.0, 0.0, 7.0, 2.0, 0.0]
}
```

**Response:**
```json
{
  "gem_cards": [
    {
      "gmap_id": "ChIJ...",
      "name": "Central Park",
      "latitude": 40.7829,
      "longitude": -73.9654,
      "score": 0.9107,
      "confidence_pct": "91%",
      "vibe_tags": ["scenic", "relaxing", "family-friendly"],
      "category": "park",
      "distance_km": 8.51,
      "avg_rating": 4.8,
      "num_reviews": 50000
    }
  ],
  "model_version": "mock-v1.0",
  "inference_time_ms": 2.8,
  "candidates_scored": 1
}
```

### `GET /model/info`
Returns model metadata.

**Response:**
```json
{
  "model_version": "mock-v1.0",
  "model_loaded": true,
  "feature_count": 25,
  "feature_names": ["category_encoded", "avg_rating", "..."]
}
```

---

## 4. Evaluation Metrics

### 4.1 Latency (Single Request)

Measured with `benchmark_latency.py` — 20 rounds per candidate count:

| Candidates | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|------------|----------|----------|----------|----------|
| 1 | **1.9** | 2.6 | 2.6 | 2.6 |
| 5 | **3.2** | 3.8 | 3.9 | 4.0 |
| 10 | **2.8** | 3.6 | 3.7 | 3.7 |
| 25 | **2.9** | 6.4 | 9.6 | 10.4 |
| 50 | **3.6** | 5.8 | 5.9 | 5.9 |
| 100 | **4.1** | 5.3 | 5.3 | 5.3 |
| 200 | **5.1** | 5.7 | 6.1 | 6.2 |

**Key Insight**: Inference is sub-linear — doubling candidates from 100→200 only adds ~1ms because XGBoost operates on batch matrices efficiently.

### 4.2 Throughput (Load Test)

Measured with Locust — 50 concurrent users for 30 seconds:

| Endpoint | Requests | Error Rate | Avg (ms) | P50 (ms) | P95 (ms) |
|----------|----------|-----------|----------|----------|----------|
| `/health` | 468 | 0% | 5 | 4 | 16 |
| `/recommend` [10] | 299 | 0% | 10 | 10 | 21 |
| `/recommend` [50] | 173 | 0% | 11 | 11 | 20 |
| `/recommend` [100] | 87 | 0% | 20 | 13 | 49 |
| **Total** | **1,102** | **0%** | **9** | **8** | **21** |

**Key Metrics:**
- **Throughput**: 38.3 requests/second (50 concurrent users)
- **Error Rate**: 0% — zero failures under load
- **P95 Aggregate**: 21ms — well under the 200ms SLA target

### 4.3 Metric Definitions

| Metric | Definition | Target |
|--------|-----------|--------|
| **P50 (Median)** | 50th percentile response time | < 20ms |
| **P95** | 95th percentile — captures tail latency | < 100ms |
| **P99** | 99th percentile — worst-case (excl. outliers) | < 200ms |
| **Throughput** | Requests handled per second | > 30 req/s |
| **Error Rate** | % of requests returning non-2xx | 0% |
| **Inference Time** | Time spent in XGBoost `.predict()` only | < 5ms |

### 4.4 How to Reproduce

```bash
# Start services
docker compose -f docker-compose.dev.yml up -d

# Latency benchmark
cd ml-serving && source .venv/bin/activate
python tests/benchmark_latency.py --rounds 50

# Load test (headless)
locust -f tests/locustfile.py --host http://localhost:8050 \
    --headless -u 50 -r 10 --run-time 60s

# Load test (with web dashboard at http://localhost:8089)
locust -f tests/locustfile.py --host http://localhost:8050
```

---

## 5. Live Demo Guide

### Prerequisites
- Docker Desktop installed and running
- Git clone of the GemSpot repo

### Step 1: Start the Application

```bash
cd GemSpot-App  # or AdventureLog
cp .env.example .env  # if not already done
docker compose -f docker-compose.dev.yml up -d --build
```

Wait ~30 seconds for all services to initialize.

### Step 2: Verify ML Service Health

```bash
curl http://localhost:8050/health
```
Expected: `{"status":"healthy","model_loaded":true,"model_version":"mock-v1.0"}`

### Step 3: Demo — Direct ML Inference (Terminal)

Show the raw ML service in action:

```bash
curl -s -X POST http://localhost:8050/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "latitude": 40.7128,
    "longitude": -74.006,
    "candidates": [
      {
        "gmap_id": "1", "name": "Central Park",
        "latitude": 40.7829, "longitude": -73.9654,
        "category": "park", "avg_rating": 4.8,
        "num_reviews": 50000, "price": 0,
        "vibe_tags": ["scenic", "relaxing", "family-friendly"]
      },
      {
        "gmap_id": "2", "name": "Joe'\''s Pizza",
        "latitude": 40.7306, "longitude": -74.0023,
        "category": "restaurant", "avg_rating": 4.5,
        "num_reviews": 8000, "price": 1,
        "vibe_tags": ["foodie", "trendy", "nightlife"]
      },
      {
        "gmap_id": "3", "name": "The Met Museum",
        "latitude": 40.7794, "longitude": -73.9632,
        "category": "museum", "avg_rating": 4.9,
        "num_reviews": 75000, "price": 2,
        "vibe_tags": ["cultural", "historic", "scenic"]
      }
    ]
  }' | python3 -m json.tool
```

**Talk about**: XGBoost scores each candidate, returns ranked Gem Cards with confidence percentages and vibe tags.

### Step 4: Demo — Frontend UI (Browser)

1. Open **http://localhost:8015**
2. Log in: **admin / admin**
3. Create a collection → "NYC Trip"
4. Add a location with coordinates (e.g., "Times Square" at lat `40.758`, lon `-73.9855`)
5. Click the **🤖 GemSpot AI** tab
6. Select the location, set radius, click **"Find Hidden Gems"**
7. Show the Gem Cards with scores, vibe tags, map markers

**Talk about**: The frontend calls the Django proxy, which runs PostGIS spatial filtering (Stage 1) and then forwards to the ML service for scoring (Stage 2).

### Step 5: Demo — Performance Metrics (Terminal)

```bash
cd ml-serving && source .venv/bin/activate

# Quick latency benchmark
python tests/benchmark_latency.py --rounds 10
```

**Talk about**: Sub-5ms P50 latency even with 200 candidates, all P95s under 10ms. XGBoost batch inference is highly efficient.

### Step 6: Demo — Load Test (Optional)

```bash
# Run with web UI for visual demo
locust -f tests/locustfile.py --host http://localhost:8050
```
Open **http://localhost:8089** → set 50 users, ramp rate 10 → Start.

**Talk about**: 0% error rate under 50 concurrent users, ~38 req/s throughput, real-time monitoring dashboard.

---

## 6. Next Steps

### Phase 2: Real Model Training
- [ ] Ingest Google Local Data (UCSD 2021, 10-cores subset)
- [ ] Build training pipeline with real user interaction data
- [ ] Train XGBoost on click/visit/rating signals
- [ ] Replace mock model at `/models/gemspot_xgb.json`

### Phase 3: Feature Store (Redis)
- [ ] Populate Redis with real-time user features (visit history, preferences)
- [ ] Update `ml-serving` to read features from Redis instead of using defaults
- [ ] Enable dynamic user preference updates

### Phase 4: Auto-Tagging Classifier
- [ ] Train NLP classifier to extract vibe tags from user comments
- [ ] Deploy as a separate microservice or add endpoint to `ml-serving`
- [ ] Replace hardcoded category→tag mapping

### Phase 5: MLOps Pipeline
- [ ] Add MLflow for experiment tracking and model registry
- [ ] Implement automated retraining on new data
- [ ] A/B testing framework for model versions
- [ ] Model monitoring (drift detection, performance degradation)

### Phase 6: Cloud Deployment
- [ ] Deploy to Chameleon Cloud (see Chameleon deployment docs)
- [ ] Run benchmarks on cloud infrastructure
- [ ] Compare local vs. cloud performance metrics
- [ ] Set up CI/CD for automated deployment
