"""
SENTINEL — Anomaly Detection Systems
FastAPI Application Entry Point
=====================================
Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Anomaly Detection Systems — SENTINEL",
    description="Market fraud & manipulation detection | Isolation Forest + GMM + LOF + Z-score ensemble",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "service" : "SENTINEL — Anomaly Detection Systems",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "api_base": "/api/v1",
        "status"  : "live",
    }
