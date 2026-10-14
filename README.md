# NeuroPace Analytics

**Cognitive load and athlete-readiness analytics: a Python back end plus an interactive dashboard prototype.**

Built by May Ahmed Mourad (sole developer) for the UniHawk × EU Business School *Level Up and Let's Play* competition.

**Dashboard:** https://maymourad.github.io/NeuroPace-Analytics/

> **Status:** prototype. All athlete data is **synthetic** (generated with NumPy). No machine-learning model is trained. The FastAPI back end is a working, runnable implementation, but the dashboard doesn't call it yet; the two currently run independently.

---

## What's in this repository

- [`index.html`](index.html): the **dashboard** (HTML, CSS and JavaScript). It runs standalone in any browser.
- [`backend/app.py`](backend/app.py): the **FastAPI back end**, with [`backend/requirements.txt`](backend/requirements.txt).

The back end was originally submitted embedded at the bottom of `index.html` (inside `<script type="text/python-backend" id="appPyReference">`). That copy is unchanged; `backend/app.py` is an exact copy of it, so it can be run and read directly.

## The analytics engine

**Cognitive Readiness Index (CRI), 0–100.** For each athlete's daily series:

1. Smooth four signals with a 7-day rolling mean: heart-rate variability (RMSSD), reaction time, dual-task interference and arousal regulation.
2. Z-score normalise each signal, inverting reaction time and dual-task interference (higher means worse).
3. Combine them with fixed weights: **HRV 30% · reaction time 25% · dual-task interference 25% · arousal 20%**.
4. Map the composite to 0–100 with a logistic function, `100 / (1 + e^(−1.2·x))`, re-centre the series on a mean of 65, and clip to 5–100.

**Session Risk Score (SRS), 0–100.** For a planned session, a physical factor (intensity × session-type weight, adjusted for duration), a cognitive factor (complexity × session-type weight, raised when the athlete's CRI is low) and a recovery factor (hours since the last session) are combined as **38% physical · 42% cognitive · 20% recovery**. The result is banded LOW (< 35), MODERATE (< 65) or HIGH, and returned with its component breakdown.

The weights are fixed, documented assumptions in the code. They haven't been validated against real athlete data.

## API endpoints

| Method | Path | Purpose |
| :-- | :-- | :-- |
| GET | `/` | Service status |
| GET | `/api/athletes` | Synthetic athlete profiles, each with its computed CRI series |
| POST | `/api/ingest` | Validate a telemetry batch and filter outliers by z-score |
| POST | `/api/compute-cri` | Compute a CRI series from submitted HRV, reaction-time, dual-task and arousal series |
| POST | `/api/session-risk` | Score a planned training session |
| GET | `/api/squad-snapshot` | CRI snapshot for the whole squad |

## Run the back end

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open http://localhost:8000/docs for FastAPI's interactive API docs.

All six endpoints were run successfully against the current FastAPI, Uvicorn and NumPy releases on 14 October 2026.

**Next step:** connect the dashboard to the API.

## Stack

Python · FastAPI · Pydantic · NumPy · HTML/CSS/JavaScript
