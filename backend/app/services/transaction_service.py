"""
Service layer — orchestrates the detection engine, builds alerts,
and expose
"""

from __future__ import annotations
import uuid, random
from datetime import datetime, timezone
from typing import List, Optional

from app.core.detection_engine import get_engine
from app.models.schemas import (
    TransactionRequest, ScoreResponse, FeedbackRequest, ScoreBreakdown
)

TICKERS  = ["AAPL","GOOGL","MSFT","TSLA","AMZN","META","NVDA","JPM","GS","BAC"]
USERS    = ["USR-4821","USR-7734","USR-2901","USR-5512","USR-8830","USR-1145"]
PATTERNS = ["normal","normal","normal","spoofing","layering",
            "wash-trading","front-running","pump-dump"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _infer_fraud_type(score: float, pattern: str) -> str:
    if pattern and pattern != "normal":
        return pattern
    if score > 90: return "spoofing"
    if score > 82: return "layering"
    if score > 75: return "front-running"
    return "wash-trading"


# Score one transaction 
def score_transaction(tx: TransactionRequest) -> ScoreResponse:
    engine = get_engine()

    ts = tx.timestamp or datetime.now(timezone.utc).timestamp()

    result = engine.score_transaction({
        "user"         : tx.user,
        "ticker"       : tx.ticker,
        "amount"       : tx.amount,
        "volume"       : tx.volume,
        "pattern"      : tx.pattern or "normal",
        "timestamp"    : ts,
        "velocity_flag": tx.velocity_flag or False,
    })

    alert_id = None
    if result["is_anomaly"]:
        alert_id = f"ALT-{str(uuid.uuid4())[:10].upper()}"
        engine.store_alert({
            "id"       : alert_id,
            "txn_id"   : tx.id,
            "severity" : result["severity"],
            "fraud_type": _infer_fraud_type(result["score"], result["pattern"]),
            "ticker"   : tx.ticker,
            "user"     : tx.user,
            "score"    : result["score"],
            "status"   : "open",
            "timestamp": _utcnow(),
            "rule_flags": result["rule_flags"],
            "z_score"  : result["z_score"],
            "breakdown": result["breakdown"],
        })

    return ScoreResponse(
        transaction_id=tx.id,
        score         =result["score"],
        is_anomaly    =result["is_anomaly"],
        severity      =result["severity"],
        z_score       =result["z_score"],
        breakdown     =ScoreBreakdown(**result["breakdown"]),
        rule_flags    =result["rule_flags"],
        pattern       =result["pattern"],
        alert_id      =alert_id,
    )


# Alert queries 
def get_alerts(status: Optional[str] = None) -> List[dict]:
    engine = get_engine()
    alerts = engine.get_alerts()
    if status:
        alerts = [a for a in alerts if a["status"] == status]
    return sorted(alerts, key=lambda a: a["timestamp"], reverse=True)


# Feedback 
def submit_feedback(req: FeedbackRequest) -> bool:
    return get_engine().record_feedback(req.alert_id, req.is_fraud)


# Stats
def get_stats() -> dict:
    return get_engine().get_stats()


# History (rolling chart data)
def get_history() -> List[dict]:
    from datetime import timedelta
    now    = datetime.now(timezone.utc)
    return [
        {
            "timestamp"    : (now - timedelta(minutes=(20 - i) * 5)).strftime("%H:%M"),
            "normal_count" : random.randint(35, 90),
            "anomaly_count": random.randint(2, 18),
            "fp_rate"      : round(random.uniform(1.5, 4.5), 1),
        }
        for i in range(20)
    ]


# Generate synthetic transaction (demo)
def generate_transaction(force_anomaly: bool = False) -> TransactionRequest:
    is_anom = force_anomaly or (random.random() > 0.72)
    pattern = random.choice(PATTERNS[3:]) if is_anom else "normal"
    amount  = random.uniform(50_000, 600_000) if is_anom else random.uniform(500, 50_000)
    volume  = random.randint(5_000, 80_000)   if is_anom else random.randint(100, 5_000)

    return TransactionRequest(
        user         =random.choice(USERS),
        ticker       =random.choice(TICKERS),
        amount       =round(amount, 2),
        volume       =volume,
        pattern      =pattern,
        velocity_flag=is_anom and random.random() > 0.5,
    )
