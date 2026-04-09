#!/usr/bin/env python3
"""
SENTINEL — Backend startup script
===================================
Usage:
  python run.py            # starts uvicorn (if installed)
  python run.py --test     # runs the built-in engine smoke test
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def smoke_test():
    from app.core.detection_engine import AnomalyDetectionEngine
    engine = AnomalyDetectionEngine()
    print("\n✓ Detection engine initialised")
    print("  Models: IsolationForest (w=0.35) · GMM/8-component (w=0.30) · LOF (w=0.20) · Z-score (w=0.15)\n")

    cases = [
        ("Normal trade",    {"user":"USR-001","ticker":"AAPL","amount":5000, "volume":200, "pattern":"normal",       "timestamp":1700000000}),
        ("Normal large",    {"user":"USR-001","ticker":"MSFT","amount":45000,"volume":2000,"pattern":"normal",       "timestamp":1700000060}),
        ("Spoofing",        {"user":"USR-002","ticker":"TSLA","amount":580000,"volume":78000,"pattern":"spoofing",   "timestamp":1700000120,"velocity_flag":True}),
        ("Layering",        {"user":"USR-003","ticker":"META","amount":310000,"volume":44000,"pattern":"layering",   "timestamp":1700000180}),
        ("Wash trading",    {"user":"USR-004","ticker":"NVDA","amount":95000, "volume":12000,"pattern":"wash-trading","timestamp":1700000240}),
        ("Front running",   {"user":"USR-005","ticker":"JPM", "amount":420000,"volume":61000,"pattern":"front-running","timestamp":1700000300,"velocity_flag":True}),
        ("Pump & dump",     {"user":"USR-006","ticker":"GME", "amount":600000,"volume":90000,"pattern":"pump-dump",  "timestamp":1700000360,"velocity_flag":True}),
    ]

    detected = 0
    for label, tx in cases:
        r     = engine.score_transaction(tx)
        icon  = "🔴" if r["severity"] == "critical" else "🟡" if r["severity"] == "high" else "🟢"
        flags = ", ".join(r["rule_flags"]) or "—"
        print(f"  {icon} {label:<20} score={r['score']:5.1f}  sev={r['severity']:<9}  anom={str(r['is_anomaly']):<5}  flags={flags}")
        if r["is_anomaly"]:
            detected += 1

    fraud_cases = len([c for c in cases if c[0] != "Normal trade" and c[0] != "Normal large"])
    print(f"\n  Detection rate: {detected}/{fraud_cases} known fraud cases → {detected/fraud_cases*100:.0f}%")
    print("\n✓ Open frontend/index.html to use the dashboard.\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        smoke_test()
    else:
        try:
            import uvicorn
            print("Starting SENTINEL API on http://0.0.0.0:8000")
            print("Swagger docs at http://localhost:8000/docs\n")
            uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
        except ImportError:
            print("uvicorn not installed — running smoke test instead.\n")
            print("To run the API:  pip install -r requirements.txt && uvicorn app.main:app --reload\n")
            smoke_test()
