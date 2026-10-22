import json
import time

import joblib
import pandas as pd

from src.config import CATEGORICALS, DEFAULT_MODEL, METRICS, NUMERICS, PIPELINE, SUMMARY

ORDERED = ["CollegeTier", "CGPA_Tier"]

_bucket = {}
_ready = False


def init_artifacts():
    global _ready
    try:
        _bucket["pipeline"] = joblib.load(PIPELINE)
        _bucket["model"] = joblib.load(DEFAULT_MODEL)
        _bucket["metrics"] = json.loads(METRICS.read_text())
        _bucket["summary"] = json.loads(SUMMARY.read_text())
        tc = _bucket["summary"].get("target_counts", {})
        _bucket["summary"]["target_counts"] = {int(k): v for k, v in tc.items()}
        _ready = True
    except Exception as exc:  # artifacts not built yet
        _ready = False
        _bucket["error"] = str(exc)
    return _ready


def ready():
    return _ready


def state():
    return _bucket


def derive_cgpa_tier(cgpa):
    if cgpa >= 8.5:
        return "High"
    if cgpa >= 7.0:
        return "Mid"
    return "Low"


def build_frame(data):
    frame = {}
    cgpa = float(data.get("CGPA", data.get("cgpa", 0.0)) or 0.0)
    for col in NUMERICS:
        if col in ORDERED:
            continue  # handled by the ordinal branch below
        key = col if col in data else col.replace("_", "")
        val = data.get(col, data.get(key, 0))
        val = float(val) if val not in (None, "") else 0.0
        if col.startswith("SGPA_Sem") and val == 0:
            val = cgpa  # blank semesters default to the student's CGPA
        frame[col] = val
    for col in CATEGORICALS + ORDERED:
        frame[col] = data.get(col, "Missing" if col in CATEGORICALS else "Low")
    for col in CATEGORICALS:
        frame[col] = frame[col] if frame[col] in {
            "Male", "Female", "Ahmedabad", "Bangalore", "Chennai", "Delhi",
            "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune",
            "CS", "Civil", "ECE", "EE", "IT", "Mechanical", "AI",
            "DataScience", "Embedded", "Networking", "Yes", "No",
        } else "Missing"
    frame["CGPA_Tier"] = derive_cgpa_tier(frame.get("CGPA", 0.0))
    return pd.DataFrame([frame])


def predict(data):
    if not _ready:
        init_artifacts()
    t0 = time.perf_counter()
    df = build_frame(data)
    feat = _bucket["pipeline"].get_feature_names_out()
    X = pd.DataFrame(_bucket["pipeline"].transform(df), columns=feat)
    proba = float(_bucket["model"].predict_proba(X)[0, 1])
    label = "Placed" if proba >= 0.5 else "Not Placed"
    latency = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "prediction": label,
        "probability": round(proba, 4),
        "probability_placed": round(proba, 4),
        "latency_ms": latency,
        "model": _bucket["model"].__class__.__name__,
        "features_used": int(X.shape[1]),
    }