# ── GemSpot ML Serving ──────────────────────────────────────────────────────
# Multi-stage build: install deps, generate mock model, run FastAPI
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim AS builder

WORKDIR /opt/gemspot

# Install system deps and Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Generate mock model (will be overridden by real model via volume mount)
RUN mkdir -p /models \
    && python scripts/generate_mock_model.py /models/gemspot_xgb.json \
    && python scripts/export_onnx_model.py --input /models/gemspot_xgb.json --output /models/gemspot_xgb.onnx

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /opt/gemspot

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages \
                    /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code and model
COPY --from=builder /opt/gemspot/app ./app
COPY --from=builder /opt/gemspot/scripts ./scripts
COPY --from=builder /models /models

ENV MODEL_PATH=/models/gemspot_xgb.json
ENV ONNX_MODEL_PATH=/models/gemspot_xgb.onnx
ENV SERVING_BACKEND=xgboost
ENV LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
