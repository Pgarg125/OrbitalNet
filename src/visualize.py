"""
visualize.py
Plotting utilities: orbit trajectories (predicted vs. true) and training
loss-convergence curves. Matplotlib is used for static plots saved to
results/ (for the report/paper), Plotly for the interactive Streamlit demo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_trajectory_matplotlib(true_traj, pred_traj, title: str = "Orbit: predicted vs true", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(true_traj[:, 0], true_traj[:, 1], "b-", label="True (RK45)", linewidth=2)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "r--", label="Predicted", linewidth=2)
    ax.scatter([0], [0], c="orange", marker="*", s=200, label="Central body", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_curve(history: dict, title: str = "Training loss convergence", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectory_plotly(true_traj, pred_traj, title: str = "Orbit: predicted vs true"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_traj[:, 0], y=true_traj[:, 1], mode="lines", name="True (RK45)"))
    fig.add_trace(
        go.Scatter(
            x=pred_traj[:, 0],
            y=pred_traj[:, 1],
            mode="lines",
            name="Predicted (PINN)",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            name="Central body",
            marker=dict(size=14, symbol="star", color="orange"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def plot_data_efficiency(fractions, pinn_mse, baseline_mse, title: str = "Data efficiency: test MSE vs. training-data fraction", save_path: str | None = None):
    """
    fractions: list of training-data fractions used (e.g. [1.0, 0.6, 0.4, 0.2]).
    pinn_mse / baseline_mse: matching lists of each model's test MSE at that fraction.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, pinn_mse, "o-", label="PINN", linewidth=2, markersize=7)
    ax.plot(fractions, baseline_mse, "s-", label="Baseline MLP", linewidth=2, markersize=7)
    ax.set_xlabel("Fraction of training data used")
    ax.set_ylabel("Test MSE (log scale)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # so the chart reads left-to-right as "less data -->"
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig