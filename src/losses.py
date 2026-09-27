"""
losses.py
Loss functions for OrbitalNet, per the synopsis's PINN loss specification:

    L_total = L_data (MSE) + lambda * L_physics

where L_physics is the residual of Newton's two-body ODE, computed via
torch.autograd.grad with respect to the time input t.

Normalisation note
-------------------
Inputs (X = [x0, y0, vx0, vy0, t]) are standardised (zero mean, unit std)
before being fed to the network for stable training -- see train.py. Targets
(Y = [x, y, vx, vy]) are LEFT UNNORMALISED, since in canonical units
(GM = 1, semi-major axis ~1) they are already O(1) in scale. This means the
network's raw output is already in physical units, EXCEPT its derivative
w.r.t. the *normalised* time input t_norm must be rescaled back to a
derivative w.r.t. physical time t via the chain rule:

    t_norm = (t - mean) / std   =>   d/dt = (1/std) * d/dt_norm

`t_std` (a scalar tensor, the standard deviation used to normalise the time
column) must therefore be passed into the physics loss so the ODE residual
is evaluated in real physical units, matching GM = 1.
"""

from __future__ import annotations

import torch

GM = 1.0


def data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def physics_residual_loss(
    model: torch.nn.Module,
    X_physics_norm: torch.Tensor,
    t_std: torch.Tensor,
    gm: float = GM,
) -> torch.Tensor:
    """
    X_physics_norm: (N, 5) NORMALISED tensor [x0, y0, vx0, vy0, t].
    t_std: scalar tensor, the std used to normalise the time (5th) column.

    Computes the mean-squared residual of the two-body Newtonian ODE:
        dx/dt  - vx                = 0
        dy/dt  - vy                = 0
        dvx/dt + GM * x / r^3      = 0
        dvy/dt + GM * y / r^3      = 0
    using the model's OWN predicted (x, y, vx, vy) at these points and
    autograd derivatives with respect to the time input.
    """
    X_physics_norm = X_physics_norm.clone().requires_grad_(True)
    pred = model(X_physics_norm)  # physical-unit output: [x, y, vx, vy]
    x, y, vx, vy = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    ones = torch.ones_like(x)

    # d(output)/d(t_norm) for each of the 4 outputs
    dx_dtn = torch.autograd.grad(x, X_physics_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 4:5]
    dy_dtn = torch.autograd.grad(y, X_physics_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 4:5]
    dvx_dtn = torch.autograd.grad(vx, X_physics_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 4:5]
    dvy_dtn = torch.autograd.grad(vy, X_physics_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0][:, 4:5]

    # Chain rule: t_norm = (t - mean) / t_std  =>  d/dt = (1/t_std) * d/dt_norm
    dx_dt = dx_dtn / t_std
    dy_dt = dy_dtn / t_std
    dvx_dt = dvx_dtn / t_std
    dvy_dt = dvy_dtn / t_std

    r = torch.sqrt(x ** 2 + y ** 2 + 0.01) 
    r3 = r ** 3

    res_x = dx_dt - vx
    res_y = dy_dt - vy
    res_vx = dvx_dt + gm * x / r3
    res_vy = dvy_dt + gm * y / r3

    residual = torch.cat([res_x, res_y, res_vx, res_vy], dim=1)
    return torch.mean(residual ** 2)


def total_pinn_loss(
    model: torch.nn.Module,
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    X_physics: torch.Tensor,
    t_std: torch.Tensor,
    lam: float = 0.1,
    gm: float = GM,
):
    """
    Returns (total_loss_tensor, data_loss_value, physics_loss_value).
    """
    pred_data = model(X_data)
    l_data = data_loss(pred_data, Y_data)
    l_physics = physics_residual_loss(model, X_physics, t_std, gm=gm)
    total = l_data + lam * l_physics
    return total, l_data.item(), l_physics.item()
