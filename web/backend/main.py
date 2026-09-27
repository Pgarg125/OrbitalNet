"""
web/backend/main.py
FastAPI backend for the OrbitalNet interactive website.

Loads the trained PINN model (models/pinn_model.pt) from the project root and
exposes a small REST API that the React frontend (web/frontend) calls:

    GET  /api/health   -> { status, model_loaded }
    POST /api/predict  -> given initial conditions, returns the TRUE
                           (RK45-integrated) trajectory, the PINN's
                           PREDICTED trajectory, and MSE/MAE between them.

This reuses the exact same src/ code (physics.py, model.py) as the rest of
the project, so the website is guaranteed to be consistent with the
Streamlit demo (app.py) and the evaluation scripts -- there is only ONE
implementation of the model and the physics, never a second copy.

Run with (from the project root, with the venv active):
    uvicorn web.backend.main:app --reload --port 8000
or simply:
    python web/backend/main.py
"""

from __future__ import annotations

import os
import sys

# Make the project root's src/ package importable, regardless of the
# directory this script is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.model import OrbitalMLP  # noqa: E402
from src.physics import GM, simulate_trajectory  # noqa: E402

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "pinn_model.pt")
STATS_PATH = os.path.join(_PROJECT_ROOT, "models", "pinn_stats.npz")

app = FastAPI(title="OrbitalNet API", version="1.0.0")

# Allow the local Vite dev server (default http://localhost:5173) and the
# built static preview server to call this API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: OrbitalMLP | None = None
_stats: dict | None = None


def _load_model_if_needed():
    """Lazily loads the trained model on first request (and caches it)."""
    global _model, _stats
    if _model is not None:
        return
    if not (os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)):
        return  # leave _model as None; endpoints report this clearly
    model = OrbitalMLP()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    stats_npz = np.load(STATS_PATH)
    _stats = {k: stats_npz[k] for k in stats_npz.files}
    _model = model


class PredictRequest(BaseModel):
    x0: float = Field(..., ge=-10, le=10, description="Initial x position")
    y0: float = Field(..., ge=-10, le=10, description="Initial y position")
    vx0: float = Field(..., ge=-10, le=10, description="Initial x velocity")
    vy0: float = Field(..., ge=-10, le=10, description="Initial y velocity")
    t_max: float = Field(..., gt=0, le=200, description="Time horizon to predict over")
    n_points: int = Field(200, ge=10, le=1000, description="Number of time samples")


class PredictResponse(BaseModel):
    t: list[float]
    true_x: list[float]
    true_y: list[float]
    true_vx: list[float]
    true_vy: list[float]
    pred_x: list[float]
    pred_y: list[float]
    pred_vx: list[float]
    pred_vy: list[float]
    mse: float
    mae: float
    gm: float


@app.get("/api/health")
def health():
    _load_model_if_needed()
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _load_model_if_needed()
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained PINN model found. From the project root, run "
                "'python scripts/generate_data.py' then 'python scripts/train_pinn.py' "
                "first, then restart this server."
            ),
        )

    state0 = np.array([req.x0, req.y0, req.vx0, req.vy0], dtype=np.float64)
    t_eval = np.linspace(0.0, req.t_max, req.n_points)

    try:
        true_traj = simulate_trajectory(state0, t_eval)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not integrate the true trajectory (likely a collision orbit through the origin): {e}",
        )

    X_query = np.zeros((req.n_points, 5), dtype=np.float64)
    X_query[:, 0:4] = state0
    X_query[:, 4] = t_eval
    Xn = (X_query - _stats["X_mean"]) / _stats["X_std"]

    with torch.no_grad():
        pred_traj = _model(torch.tensor(Xn, dtype=torch.float32)).numpy()

    mse = float(np.mean((pred_traj - true_traj) ** 2))
    mae = float(np.mean(np.abs(pred_traj - true_traj)))

    return PredictResponse(
        t=t_eval.tolist(),
        true_x=true_traj[:, 0].tolist(),
        true_y=true_traj[:, 1].tolist(),
        true_vx=true_traj[:, 2].tolist(),
        true_vy=true_traj[:, 3].tolist(),
        pred_x=pred_traj[:, 0].tolist(),
        pred_y=pred_traj[:, 1].tolist(),
        pred_vx=pred_traj[:, 2].tolist(),
        pred_vy=pred_traj[:, 3].tolist(),
        mse=mse,
        mae=mae,
        gm=float(GM),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
