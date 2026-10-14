"""
app.py — NeuroPace Analytics · FastAPI Core Engine & Analytical Pipeline
UniHawk × EU Business School · Level Up and Let's Play Competition

Run: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import json
import time
from datetime import datetime, timedelta

app = FastAPI(
    title="NeuroPace Analytics API",
    description="Cognitive Load Inference Engine — CLIE v2.1",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────────────────────

class TelemetryRecord(BaseModel):
    athlete_id: int
    hrv_rmssd: float          # ms
    reaction_time_ms: float   # ms
    dual_task_score: float    # 0–100
    arousal_index: float      # 0–10
    session_load: float       # 0–100
    timestamp: Optional[str] = None


class TelemetryBatch(BaseModel):
    records: List[TelemetryRecord]


class CRIRequest(BaseModel):
    athlete_id: int
    hrv_series: List[float]          # 14-day rolling HRV (RMSSD)
    rt_series: List[float]           # 14-day rolling reaction-time ms
    dual_task_series: List[float]    # 14-day dual-task interference index
    arousal_series: List[float]      # 14-day arousal regulation values
    window_size: Optional[int] = 7  # rolling window for z-score


class SessionPlannerRequest(BaseModel):
    athlete_id: int
    intensity: int          # 1–10
    duration_minutes: int   # 15–180
    tactical_complexity: int # 1–10
    recovery_hours: float   # hours since last load
    session_type: str       # recovery|technical|tactical|high_intensity|match
    athlete_cri: Optional[float] = None  # override; None = compute from CLIE


# ─────────────────────────────────────────────────────────────
#  MOCK DATASET GENERATOR
# ─────────────────────────────────────────────────────────────

ATHLETE_PROFILES = {
    1:  {"name": "Ramos V.",    "pos": "FWD", "base_hrv": 72, "base_rt": 220, "base_cri": 88},
    2:  {"name": "Chen L.",     "pos": "MID", "base_hrv": 61, "base_rt": 265, "base_cri": 74},
    3:  {"name": "Al-Farsi K.", "pos": "DEF", "base_hrv": 48, "base_rt": 320, "base_cri": 44},
    4:  {"name": "Petrov N.",   "pos": "GK",  "base_hrv": 80, "base_rt": 195, "base_cri": 91},
    5:  {"name": "Diallo S.",   "pos": "WNG", "base_hrv": 55, "base_rt": 290, "base_cri": 67},
    6:  {"name": "Martini F.",  "pos": "CDM", "base_hrv": 52, "base_rt": 310, "base_cri": 55},
    7:  {"name": "Torres A.",   "pos": "FWD", "base_hrv": 68, "base_rt": 235, "base_cri": 82},
    8:  {"name": "Kim J.",      "pos": "MID", "base_hrv": 42, "base_rt": 375, "base_cri": 39},
    9:  {"name": "Okonkwo E.",  "pos": "DEF", "base_hrv": 65, "base_rt": 250, "base_cri": 78},
    10: {"name": "Ivanov P.",   "pos": "WNG", "base_hrv": 57, "base_rt": 300, "base_cri": 63},
    11: {"name": "Santos B.",   "pos": "CDM", "base_hrv": 70, "base_rt": 225, "base_cri": 85},
    12: {"name": "Müller T.",   "pos": "DEF", "base_hrv": 45, "base_rt": 355, "base_cri": 47},
}


def generate_gaussian_series(base: float, std: float, n: int, lo: float, hi: float) -> np.ndarray:
    """Generate a realistic drift series using Box-Muller Gaussian sampling."""
    series = np.zeros(n)
    val = base
    for i in range(n):
        val += np.random.normal(0, std)
        series[i] = np.clip(val, lo, hi)
    return np.round(series, 2)


def generate_athlete_mock_data(athlete_id: int, days: int = 14) -> dict:
    """Seed realistic athlete telemetry for CLIE initialisation."""
    if athlete_id not in ATHLETE_PROFILES:
        raise ValueError(f"Unknown athlete_id: {athlete_id}")

    p = ATHLETE_PROFILES[athlete_id]
    rng = np.random.default_rng(athlete_id * 42)

    hrv_series    = np.clip(rng.normal(p["base_hrv"], 6, days), 20, 120).tolist()
    rt_series     = np.clip(rng.normal(p["base_rt"],  18, days), 180, 550).tolist()
    dt_series     = np.clip(rng.normal(65, 12, days), 10, 100).tolist()
    arousal_series = np.clip(rng.normal(5.5, 1.2, days), 1, 10).tolist()

    timestamps = [
        (datetime.utcnow() - timedelta(days=days - i)).isoformat()
        for i in range(days)
    ]

    return {
        "athlete_id":     athlete_id,
        "name":           p["name"],
        "position":       p["pos"],
        "hrv_series":     [round(v, 1) for v in hrv_series],
        "rt_series":      [round(v, 1) for v in rt_series],
        "dual_task_series": [round(v, 1) for v in dt_series],
        "arousal_series": [round(v, 2) for v in arousal_series],
        "timestamps":     timestamps,
    }


# ─────────────────────────────────────────────────────────────
#  CLIE — COGNITIVE LOAD INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────

def zscore_series(arr: np.ndarray) -> np.ndarray:
    """Standard z-score normalisation. Returns zeros on zero variance."""
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def rolling_window_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling window mean — pads with first value."""
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = arr[start:i + 1].mean()
    return result


def compute_cri_vector(
    hrv: np.ndarray,
    rt: np.ndarray,
    dual_task: np.ndarray,
    arousal: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    CLIE Core: Cognitive Readiness Index computation.

    CRI = weighted combination of z-score normalised sub-dimensions,
    then mapped to 0–100 scale using logistic transformation.

    Weights (derived from dual-task interference literature):
      HRV (RMSSD):            30% — autonomic readiness
      Reaction Time:          25% — processing speed proxy
      Dual-Task Interference: 25% — working-memory load
      Arousal Regulation:     20% — arousal modulation quality
    """
    # Rolling window smoothing
    hrv_smooth = rolling_window_mean(hrv, window)
    rt_smooth  = rolling_window_mean(rt,  window)
    dt_smooth  = rolling_window_mean(dual_task, window)
    ar_smooth  = rolling_window_mean(arousal, window)

    # Z-score normalise
    hrv_z = zscore_series(hrv_smooth)
    rt_z  = zscore_series(rt_smooth)   # higher RT = lower readiness → invert
    dt_z  = zscore_series(dt_smooth)
    ar_z  = zscore_series(ar_smooth)

    # Invert RT and DT (higher value = worse cognition)
    rt_z_inv = -rt_z
    dt_z_inv = -dt_z

    # Weighted composite (z-score domain)
    composite = (
        0.30 * hrv_z    +
        0.25 * rt_z_inv +
        0.25 * dt_z_inv +
        0.20 * ar_z
    )

    # Logistic sigmoid mapping to 0–100
    cri_raw = 100 / (1 + np.exp(-composite * 1.2))

    # Anchor to population norm (mean ≈ 65)
    shift = 65 - cri_raw.mean()
    cri = np.clip(cri_raw + shift, 5, 100)

    return np.round(cri, 1)


def compute_session_risk_score(
    intensity: int,
    duration: int,
    complexity: int,
    recovery_hours: float,
    session_type: str,
    athlete_cri: float,
) -> dict:
    """
    Session Risk Matrix — CLIE SRS computation.
    Returns risk score 0–100 + component breakdown.
    """
    type_weights = {
        "recovery":       {"cog": 0.3, "phys": 0.3},
        "technical":      {"cog": 0.6, "phys": 0.5},
        "tactical":       {"cog": 0.8, "phys": 0.65},
        "high_intensity": {"cog": 0.5, "phys": 0.9},
        "match":          {"cog": 1.0, "phys": 1.0},
    }
    w = type_weights.get(session_type, type_weights["tactical"])

    cri_penalty = np.clip((100 - athlete_cri) / 100, 0, 1)

    phys_factor = np.clip(
        (intensity / 10) * w["phys"] * (1 + (duration - 60) / 200),
        0, 1
    )
    cog_factor = np.clip(
        (complexity / 10) * w["cog"] * (1 + cri_penalty * 0.6),
        0, 1
    )
    rec_factor = np.clip(1 - (recovery_hours / 48), 0, 1) * 0.6

    raw_srs = (phys_factor * 0.38 + cog_factor * 0.42 + rec_factor * 0.20) * 100
    srs = int(np.clip(raw_srs, 0, 100))

    band = "LOW"   if srs < 35 else "MODERATE" if srs < 65 else "HIGH"

    return {
        "srs":            srs,
        "band":           band,
        "component_phys": round(float(phys_factor) * 100, 1),
        "component_cog":  round(float(cog_factor)  * 100, 1),
        "component_rec":  round(float(rec_factor)  * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "platform": "NeuroPace Analytics",
        "engine":   "CLIE v2.1",
        "status":   "operational",
        "athletes": len(ATHLETE_PROFILES),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/athletes")
def list_athletes():
    """Return all athlete profiles with seeded mock data."""
    result = []
    for aid in ATHLETE_PROFILES:
        data = generate_athlete_mock_data(aid)
        # Compute current CRI from last window
        hrv = np.array(data["hrv_series"])
        rt  = np.array(data["rt_series"])
        dt  = np.array(data["dual_task_series"])
        ar  = np.array(data["arousal_series"])
        cri_series = compute_cri_vector(hrv, rt, dt, ar)
        data["cri_series"]   = cri_series.tolist()
        data["current_cri"]  = float(cri_series[-1])
        data["current_hrv"]  = data["hrv_series"][-1]
        data["current_rt"]   = data["rt_series"][-1]
        result.append(data)
    return {"athletes": result}


@app.post("/api/ingest")
def ingest_telemetry(batch: TelemetryBatch):
    """
    Accepts mock telemetry records (CSV or JSON).
    Validates, z-score filters outliers, and returns cleaned payload.
    """
    if not batch.records:
        raise HTTPException(status_code=400, detail="Empty telemetry batch.")

    cleaned = []
    rejected = 0

    # Build arrays per field for z-score outlier filtering
    hrv_vals = np.array([r.hrv_rmssd        for r in batch.records])
    rt_vals  = np.array([r.reaction_time_ms for r in batch.records])
    dt_vals  = np.array([r.dual_task_score  for r in batch.records])
    ar_vals  = np.array([r.arousal_index    for r in batch.records])

    hrv_z = zscore_series(hrv_vals)
    rt_z  = zscore_series(rt_vals)

    for i, r in enumerate(batch.records):
        # Reject records with |z| > 3.5 on primary signals
        if abs(hrv_z[i]) > 3.5 or abs(rt_z[i]) > 3.5:
            rejected += 1
            continue
        ts = r.timestamp or datetime.utcnow().isoformat()
        cleaned.append({
            "athlete_id":      r.athlete_id,
            "hrv_rmssd":       round(r.hrv_rmssd, 2),
            "reaction_time_ms": round(r.reaction_time_ms, 2),
            "dual_task_score": round(r.dual_task_score, 2),
            "arousal_index":   round(r.arousal_index, 2),
            "session_load":    round(r.session_load, 2),
            "timestamp":       ts,
            "hrv_zscore":      round(float(hrv_z[i]), 3),
            "rt_zscore":       round(float(rt_z[i]), 3),
        })

    return {
        "status":         "ok",
        "total_received": len(batch.records),
        "accepted":       len(cleaned),
        "rejected":       rejected,
        "records":        cleaned,
    }


@app.post("/api/compute-cri")
def compute_cri_endpoint(req: CRIRequest):
    """
    CLIE Core Endpoint — Cognitive Readiness Index.
    Executes full pipeline: z-score, rolling window, vector computation.
    """
    if len(req.hrv_series) < 3:
        raise HTTPException(status_code=400, detail="Minimum 3 data points required.")

    hrv = np.array(req.hrv_series,        dtype=float)
    rt  = np.array(req.rt_series,         dtype=float)
    dt  = np.array(req.dual_task_series,  dtype=float)
    ar  = np.array(req.arousal_series,    dtype=float)

    # Enforce equal length
    min_len = min(len(hrv), len(rt), len(dt), len(ar))
    hrv, rt, dt, ar = hrv[:min_len], rt[:min_len], dt[:min_len], ar[:min_len]

    window = max(2, min(req.window_size, min_len))
    cri_series = compute_cri_vector(hrv, rt, dt, ar, window=window)

    current_cri = float(cri_series[-1])
    state = "green" if current_cri >= 80 else "amber" if current_cri >= 50 else "red"

    # Compute sub-dimension scores for cognitive signature
    cog_sig = {
        "sustained_attention":  float(np.clip(np.mean(dt)  * 0.9  + np.random.normal(0, 3), 10, 100)),
        "inhibitory_control":   float(np.clip(np.mean(hrv) * 0.85 + np.random.normal(0, 4), 10, 100)),
        "working_memory_load":  float(np.clip(100 - np.mean(dt) * 0.4 + np.random.normal(0, 5), 10, 100)),
        "processing_speed":     float(np.clip(100 - (np.mean(rt) - 200) / 3.5, 10, 100)),
        "decision_latency":     float(np.clip(100 - (np.mean(rt) - 180) / 3.8, 10, 100)),
        "arousal_regulation":   float(np.clip(np.mean(ar) * 9 + np.random.normal(0, 3), 10, 100)),
    }

    return {
        "athlete_id":       req.athlete_id,
        "cri_series":       cri_series.tolist(),
        "current_cri":      round(current_cri, 1),
        "state":            state,
        "window_used":      window,
        "n_points":         min_len,
        "cognitive_signature": {k: round(v, 1) for k, v in cog_sig.items()},
        "computed_at":      datetime.utcnow().isoformat(),
    }


@app.post("/api/session-risk")
def session_risk_endpoint(req: SessionPlannerRequest):
    """Compute Session Risk Score (SRS) via the Session Risk Matrix."""
    if req.athlete_id not in ATHLETE_PROFILES:
        raise HTTPException(status_code=404, detail="Athlete not found.")

    # If no CRI override, generate from mock data
    if req.athlete_cri is None:
        data = generate_athlete_mock_data(req.athlete_id)
        hrv = np.array(data["hrv_series"])
        rt  = np.array(data["rt_series"])
        dt  = np.array(data["dual_task_series"])
        ar  = np.array(data["arousal_series"])
        cri_series = compute_cri_vector(hrv, rt, dt, ar)
        athlete_cri = float(cri_series[-1])
    else:
        athlete_cri = float(req.athlete_cri)

    result = compute_session_risk_score(
        intensity=req.intensity,
        duration=req.duration_minutes,
        complexity=req.tactical_complexity,
        recovery_hours=req.recovery_hours,
        session_type=req.session_type,
        athlete_cri=athlete_cri,
    )

    return {
        "athlete_id":     req.athlete_id,
        "athlete_name":   ATHLETE_PROFILES[req.athlete_id]["name"],
        "athlete_cri":    round(athlete_cri, 1),
        "session_type":   req.session_type,
        **result,
        "computed_at":    datetime.utcnow().isoformat(),
    }


@app.get("/api/squad-snapshot")
def squad_snapshot():
    """Full squad CRI snapshot — for dashboard initialisation."""
    snapshot = []
    for aid, profile in ATHLETE_PROFILES.items():
        data = generate_athlete_mock_data(aid, days=7)
        hrv = np.array(data["hrv_series"])
        rt  = np.array(data["rt_series"])
        dt  = np.array(data["dual_task_series"])
        ar  = np.array(data["arousal_series"])
        cri = compute_cri_vector(hrv, rt, dt, ar, window=5)
        cri_now = float(cri[-1])
        state = "green" if cri_now >= 80 else "amber" if cri_now >= 50 else "red"
        snapshot.append({
            "athlete_id":   aid,
            "name":         profile["name"],
            "position":     profile["pos"],
            "current_cri":  round(cri_now, 1),
            "current_hrv":  data["hrv_series"][-1],
            "current_rt":   data["rt_series"][-1],
            "state":        state,
        })
    avg_cri = round(np.mean([s["current_cri"] for s in snapshot]), 1)
    critical = sum(1 for s in snapshot if s["state"] == "red")
    return {
        "squad_size":   len(snapshot),
        "average_cri":  avg_cri,
        "critical_count": critical,
        "snapshot":     snapshot,
        "timestamp":    datetime.utcnow().isoformat(),
    }
