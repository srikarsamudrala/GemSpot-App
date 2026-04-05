"""
GemSpot ML Serving — Configuration
"""
import os


MODEL_PATH = os.getenv("MODEL_PATH", "/models/gemspot_xgb.json")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/models/gemspot_xgb.onnx")
SERVING_BACKEND = os.getenv("SERVING_BACKEND", "xgboost").strip().lower()
TRITON_URL = os.getenv("TRITON_URL", "http://triton-server:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "gemspot_onnx")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
DEFAULT_RADIUS_KM = float(os.getenv("DEFAULT_RADIUS_KM", "10.0"))
