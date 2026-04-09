"""Pydantic request / response schemas."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime, timezone
import uuid


# Helpers 
def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _txn_id() -> str:
    return f"TXN-{str(uuid.uuid4())[:8].upper()}"


# Inbound
class TransactionRequest(BaseModel):
    id          : str   = Field(default_factory=_txn_id)
    user        : str
    ticker      : str
    amount      : float = Field(gt=0, description="Transaction amount in USD")
    volume      : int   = Field(gt=0, description="Share/unit volume")
    pattern     : Optional[str]  = "normal"
    timestamp   : Optional[float] = None
    velocity_flag: Optional[bool] = False

    model_config = {"json_schema_extra": {"example": {
        "user": "USR-4821", "ticker": "AAPL",
        "amount": 125000, "volume": 5000, "pattern": "normal",
    }}}


class FeedbackRequest(BaseModel):
    alert_id     : str
    is_fraud     : bool
    analyst_notes: Optional[str] = None


class BulkScoreRequest(BaseModel):
    transactions: List[TransactionRequest]


# Model breakdown
class ScoreBreakdown(BaseModel):
    isolation_forest: float
    gmm             : float
    lof             : float
    statistical     : float


# Outbound
class ScoreResponse(BaseModel):
    transaction_id: str
    score         : float
    is_anomaly    : bool
    severity      : Literal["medium", "high", "critical"]
    z_score       : float
    breakdown     : ScoreBreakdown
    rule_flags    : List[str]
    pattern       : str
    alert_id      : Optional[str] = None
    timestamp     : str = Field(default_factory=_utcnow)


class AlertModel(BaseModel):
    id          : str
    txn_id      : str
    severity    : str
    fraud_type  : str
    ticker      : str
    user        : str
    score       : float
    status      : Literal["open", "resolved", "false_positive"]
    timestamp   : str
    rule_flags  : List[str] = []
    z_score     : float = 0.0
    breakdown   : Optional[Dict[str, float]] = None


class StatsResponse(BaseModel):
    total_scored   : int
    open_alerts    : int
    confirmed_fraud: int
    false_positives: int
    fp_rate        : float
    precision      : float
    recall         : float
    f1             : float
