"""Advection-Dispersion Equation (ADE) physics loss for the Gelman PINN.

The ADE with vertical dispersion and a lateral-loss sink term:

    ∂C/∂t + v·∂C/∂x - D·∂²C/∂x² - Dz·∂²C/∂z² + λ·C = 0

where:
    v   — seepage velocity (ft/yr)
    D   — longitudinal dispersion (ft²/yr)
    Dz  — vertical dispersion = D × 0.1  (ft²/yr); transverse dispersivity
          is ~10% of longitudinal per standard groundwater convention
    λ   — lateral (y-direction) loss coefficient (1/yr); a first-order sink
          that accounts for mass flux leaving the 1D transect corridor in the
          y-direction, which this model cannot explicitly resolve. This is
          distinct from vertical transport, which is already captured by the
          Dz·∂²C/∂z² term above. Motivated by the 2,200 kg mass-balance
          deficit reported by Lemke (2022).
    C   — normalized log-concentration (dimensionless)

All derivatives are computed via automatic differentiation (torch.autograd.grad)
and de-normalized using the physical scales before computing the residual.

Unit note:
    v and D come from model.get_physics_params() in ft/day and ft²/day and are
    converted to annual units by ×365.25 here. lambda comes in as 1/yr directly
    (no conversion needed).
"""

import torch


def ade_residual(
    model,
    col_input: torch.Tensor,
    depth_range: float,
    x_max: float,
    t_max: float,
) -> torch.Tensor:
    """Mean-squared ADE residual (with leakage) at collocation points.

    Args:
        model: DioxanePINN instance.
        col_input: [N, 3] tensor of (x_norm, z_norm, t_norm) points.
                   requires_grad is set internally — pass a plain tensor.
        depth_range: MAX_DEPTH - MIN_DEPTH in feet (for z de-normalization).
        x_max: X_MAX_DIST in feet (for x de-normalization).
        t_max: T_MAX_YEARS (for t de-normalization).

    Returns:
        Scalar mean-squared ADE residual, differentiable w.r.t. model params.
    """
    col_input = col_input.requires_grad_(True)

    v_day, D_day, lam = model.get_physics_params()
    v = v_day * 365.25    # ft/day  → ft/yr
    D = D_day * 365.25    # ft²/day → ft²/yr
    # lam (1/yr) — lateral loss, already in annual units, no conversion needed

    C = model(col_input)

    # First-order spatial and temporal derivatives
    # Chain rule: ∂C/∂x_phys = (∂C/∂x_norm) / x_max
    grads = torch.autograd.grad(
        C, col_input, torch.ones_like(C), create_graph=True
    )[0]
    C_x = grads[:, 0:1] / x_max
    C_z = grads[:, 1:2] / depth_range
    C_t = grads[:, 2:3] / t_max

    # Second-order derivatives for longitudinal and vertical dispersion
    C_xx = torch.autograd.grad(
        C_x, col_input, torch.ones_like(C_x), create_graph=True
    )[0][:, 0:1] / x_max

    C_zz = torch.autograd.grad(
        C_z, col_input, torch.ones_like(C_z), create_graph=True
    )[0][:, 1:2] / depth_range

    # Full ADE residual including lateral-loss sink:
    # ∂C/∂t + v·∂C/∂x - D·∂²C/∂x² - Dz·∂²C/∂z² + λ·C = 0
    # Note: Dz term handles vertical transport; λ·C is y-direction mass loss only.
    residual = C_t + v * C_x - D * C_xx - (D * 0.1) * C_zz + lam * C
    return torch.mean(residual ** 2)


def sample_collocation(n: int, device: torch.device) -> torch.Tensor:
    """Sample n uniform collocation points in the normalized domain [0,1]³.

    Returns a detached [n, 3] tensor; requires_grad is set by ade_residual.
    """
    return torch.rand(n, 3, device=device)
