"""
app.py
Streamlit demo for OrbitalNet: enter initial orbital conditions and see the
trained PINN's predicted trajectory plotted live against the true
(RK45-integrated) orbit, per the synopsis's "Live Streamlit Demo" objective.

Run with:
    streamlit run app.py

Requires a trained PINN model at models/pinn_model.pt (run
scripts/generate_data.py then scripts/train_pinn.py first).
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
import torch

from src.model import OrbitalMLP
from src.physics import GM, simulate_trajectory
from src.visualize import plot_trajectory_plotly

st.set_page_config(page_title="OrbitalNet Demo", page_icon="🛰️", layout="wide")

st.title("🛰️ OrbitalNet — Physics-Informed Orbit Prediction")
st.caption(
    "Enter initial conditions for a two-body orbit and compare the PINN's "
    "prediction against the true numerically-integrated (RK45) trajectory."
)


@st.cache_resource
def load_model():
    model_path = "models/pinn_model.pt"
    stats_path = "models/pinn_stats.npz"
    if not (os.path.exists(model_path) and os.path.exists(stats_path)):
        return None, None
    model = OrbitalMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    stats_npz = np.load(stats_path)
    stats = {k: stats_npz[k] for k in stats_npz.files}
    return model, stats


model, stats = load_model()

if model is None:
    st.warning(
        "No trained PINN model found at `models/pinn_model.pt`.\n\n"
        "Run these two commands first, then restart this app:\n\n"
        "```\npython scripts/generate_data.py\npython scripts/train_pinn.py\n```"
    )
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Initial conditions")
    st.caption("Canonical units, central mass fixed at GM = 1.0")
    x0 = st.slider("x₀ (initial x position)", -2.0, 2.0, 1.0, 0.05)
    y0 = st.slider("y₀ (initial y position)", -2.0, 2.0, 0.0, 0.05)
    vx0 = st.slider("vx₀ (initial x velocity)", -2.0, 2.0, 0.0, 0.05)
    vy0 = st.slider("vy₀ (initial y velocity)", -2.0, 2.0, 1.0, 0.05)
    t_max = st.slider("Time horizon", 1.0, 30.0, 10.0, 0.5)
    n_points = st.slider("Number of points", 50, 400, 200, 10)

state0 = np.array([x0, y0, vx0, vy0])
t_eval = np.linspace(0, t_max, n_points)

with col2:
    with st.spinner("Integrating true trajectory and running PINN inference..."):
        try:
            true_traj = simulate_trajectory(state0, t_eval)
        except Exception as e:  # e.g. a collision / singular orbit
            st.error(f"True trajectory integration failed (likely a collision orbit through the origin): {e}")
            st.stop()

        X_query = np.zeros((n_points, 5))
        X_query[:, 0:4] = state0
        X_query[:, 4] = t_eval
        Xn = (X_query - stats["X_mean"]) / stats["X_std"]
        with torch.no_grad():
            pred_traj = model(torch.tensor(Xn, dtype=torch.float32)).numpy()

    mse = float(np.mean((pred_traj - true_traj) ** 2))
    mae = float(np.mean(np.abs(pred_traj - true_traj)))

    fig = plot_trajectory_plotly(true_traj, pred_traj, title="Predicted vs true orbit")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("MSE", f"{mse:.5f}")
    m2.metric("MAE", f"{mae:.5f}")

st.caption(f"Canonical units, GM = {GM} (fixed across the whole dataset). Two-body problem only.")
