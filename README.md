# 🗺️ GemSpot — AI-Powered Travel Recommendation Platform

> Discover hidden gem destinations using ML-powered recommendations scored by XGBoost.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   SvelteKit  │───▶│    Django     │───▶│   FastAPI     │
│   Frontend   │    │   Backend    │    │  ML Serving   │
│  :8015       │    │  :8016       │    │  :8050        │
└──────────────┘    └──────┬───────┘    └──────┬───────┘
                           │                    │
                    ┌──────▼───────┐    ┌──────▼───────┐
                    │  PostgreSQL  │    │    Redis      │
                    │  + PostGIS   │    │  (Features)   │
                    └──────────────┘    └──────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | SvelteKit, DaisyUI, MapLibre |
| Backend API | Django, Django REST Framework, PostGIS |
| ML Serving | FastAPI, XGBoost, Pydantic |
| Database | PostgreSQL + PostGIS |
| Cache/Features | Redis |
| Orchestration | Docker Compose |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/gemspot.git
cd gemspot

# 2. Create environment file
cp .env.example .env

# 3. Start all services
docker compose -f docker-compose.dev.yml up --build

# 4. Open the app
open http://localhost:8015
# Login: admin / admin
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 8015 | SvelteKit web app |
| Backend | 8016 | Django REST API |
| ML Serving | 8050 | FastAPI + XGBoost inference |
| PostgreSQL | 5432 | Database with PostGIS |
| Redis | 6379 | Feature store & cache |

## ML Pipeline

GemSpot uses a **two-stage recommendation pipeline**:

1. **Candidate Generation** — PostGIS spatial queries find destinations within a search radius
2. **ML Scoring** — XGBoost model scores candidates using a 25-feature vector:
   - 5 scalar features (category, rating, reviews, price, user visits)
   - 10 vibe tag features (scenic, relaxing, adventurous, cultural, etc.)
   - 10 user preference features

### Testing the ML Service

```bash
# Health check
curl http://localhost:8050/health

# Run inference
curl -X POST http://localhost:8050/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "latitude": 40.7128,
    "longitude": -74.006,
    "candidates": [{
      "gmap_id": "1",
      "name": "Central Park",
      "latitude": 40.7829,
      "longitude": -73.9654,
      "category": "park",
      "avg_rating": 4.8,
      "num_reviews": 50000,
      "price": 0,
      "vibe_tags": ["scenic", "relaxing", "family-friendly"]
    }]
  }'
```

## Project Structure

```
gemspot/
├── frontend/          # SvelteKit web application
├── backend/           # Django REST API
├── ml-serving/        # FastAPI ML inference service
│   ├── app/
│   │   ├── main.py    # FastAPI endpoints
│   │   ├── model.py   # XGBoost model & feature engineering
│   │   ├── schemas.py # Pydantic request/response models
│   │   └── vibe_tags.py # Vibe tag vocabulary & encoding
│   ├── scripts/
│   │   └── generate_mock_model.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.dev.yml
└── .env.example
```

## License

This project is built for educational purposes as part of an MLOps course project.
