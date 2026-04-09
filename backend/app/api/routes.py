"""All API endpoints for the SENTINEL platform."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.schemas import (
    TransactionRequest, ScoreResponse,
    FeedbackRequest, BulkScoreRequest, StatsResponse,
)
from app.services import transaction_service as svc

router = APIRouter()



@router.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "SENTINEL anomaly-detection-api"}


@router.post("/score", response_model=ScoreResponse, tags=["detection"])
def score_transaction(tx: TransactionRequest):
    """
    Run one transaction through the full ML ensemble pipeline.
    Returns risk score 0–100, severity level, Z-score, and per-model breakdown.
    """
    return svc.score_transaction(tx)


@router.post("/score/bulk", tags=["detection"])
def score_bulk(req: BulkScoreRequest):
    """Score multiple transactions in one call."""
    results = []
    for tx in req.transactions:
        try:
            results.append(svc.score_transaction(tx))
        except Exception as e:
            results.append({"error": str(e), "transaction_id": tx.id})
    return {"results": results, "count": len(results)}


@router.get("/transaction/generate", tags=["demo"])
def generate_transaction(
    force_anomaly: bool = Query(False, description="Force generation of an anomalous transaction"),
):
    """Generate a realistic synthetic transaction for demo / testing."""
    return svc.generate_transaction(force_anomaly=force_anomaly)

@router.get("/alerts", tags=["alerts"])
def get_alerts(status: Optional[str] = Query(None, description="open | resolved | false_positive")):
    """Return all alerts, optionally filtered by status."""
    alerts = svc.get_alerts(status)
    return {"alerts": alerts, "count": len(alerts)}


@router.get("/alerts/{alert_id}", tags=["alerts"])
def get_alert(alert_id: str):
    from app.core.detection_engine import get_engine
    alert = get_engine().get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@router.post("/feedback", tags=["feedback"])
def submit_feedback(req: FeedbackRequest):
    """
    Analyst confirms fraud OR marks as false positive.
    The label is queued for incremental XGBoost retraining.
    """
    ok = svc.submit_feedback(req)
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {
        "status"  : "accepted",
        "alert_id": req.alert_id,
        "label"   : "fraud" if req.is_fraud else "false_positive",
    }

@router.get("/stats", response_model=StatsResponse, tags=["ops"])
def get_stats():
    """Platform-wide detection KPIs."""
    return svc.get_stats()

@router.get("/history", tags=["ops"])
def get_history():
    """Rolling 20-point detection activity timeline for front-end charts."""
    return {"history": svc.get_history()}

@router.get("/model/info", tags=["ops"])
def model_info():
    return {
        "ensemble": [
            {"name": "Isolation Forest",    "weight": 0.35, "type": "unsupervised",
             "description": "Global anomaly detection via random partitioning. O(n log n). 200 trees."},
            {"name": "Gaussian Mixture Model", "weight": 0.30, "type": "unsupervised",
             "description": "8-component probabilistic density scoring. Score = -log P(x)."},
            {"name": "Local Outlier Factor", "weight": 0.20, "type": "unsupervised",
             "description": "Local density comparison — catches contextual anomalies. k=20."},
            {"name": "Statistical Z-score",  "weight": 0.15, "type": "statistical",
             "description": "Chi-squared Mahalanobis distance. Threshold at 97.5th percentile."},
        ],
        "thresholds": {
            "anomaly_gate": 72,
            "high"        : 76,
            "critical"    : 88,
        },
        "features": [
            "log(amount)", "log(volume)",
            "user_z_score_amount", "user_z_score_volume",
            "velocity_per_minute", "normalised_tx_count",
        ],
        "scaler"  : "MinMaxScaler [0, 1]",
        "latency_target_ms": 50,
    }
