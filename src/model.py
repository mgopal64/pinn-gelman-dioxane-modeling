"""DioxanePINN neural network architecture.

Architecture: 3-input (x_norm, z_norm, t_norm) → 4 × [64, Tanh] → [1, Sigmoid].
Physics parameters (all learnable, sigmoid-constrained):
    v       — seepage velocity (ft/day); bounds are unit-specific
    D       — longitudinal dispersion (ft²/day); bounds are unit-specific
    lambda  — lateral (y-direction) loss coefficient (1/yr); first-order sink
              for mass leaving the 1D transect corridor in the y-direction,
              which this model cannot explicitly resolve. Vertical transport
              is already captured by the Dz·∂²C/∂z² term in the ADE — this
              parameter is distinct from that. Motivated by the 2,200 kg
              mass-balance deficit reported by Lemke (2022).

Unit convention:
    v and D are stored/returned in ft/day and ft²/day respectively.
    Multiply by 365.25 to get annual units (done inside ade_residual).
    lambda is stored/returned in 1/yr directly (no day conversion needed).
"""

import torch
import torch.nn as nn


class DioxanePINN(nn.Module):
    """Physics-Informed Neural Network for 1,4-Dioxane advection-dispersion.

    Args:
        v_min, v_max:       Velocity bounds (ft/day). Default: Unit C3 bounds.
        d_min, d_max:       Dispersion bounds (ft²/day). Default: Unit C3 bounds.
        lam_min, lam_max:   Leakage coefficient bounds (1/yr).
    """

    def __init__(
        self,
        v_min: float = 0.41,
        v_max: float = 0.68,
        d_min: float = 2.73,
        d_max: float = 13.68,
        lam_min: float = 0.0,
        lam_max: float = 0.05,
    ):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.d_min = d_min
        self.d_max = d_max
        self.lam_min = lam_min
        self.lam_max = lam_max

        self.network = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.v_raw = nn.Parameter(torch.tensor([0.0]))
        self.D_raw = nn.Parameter(torch.tensor([0.0]))
        self.lambda_raw = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x_combined: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_combined: [N, 3] tensor with columns [x_norm, z_norm, t_norm].

        Returns:
            [N, 1] tensor of normalized log-concentrations in (0, 1).
        """
        return self.network(x_combined)

    def get_physics_params(self):
        """Return learnable physics parameters within their bounds.

        Returns:
            v   (ft/day)  — seepage velocity
            D   (ft²/day) — longitudinal dispersion
            lam (1/yr)    — vertical leakage coefficient
        """
        v = self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.v_raw)
        D = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(self.D_raw)
        lam = self.lam_min + (self.lam_max - self.lam_min) * torch.sigmoid(self.lambda_raw)
        return v, D, lam

    def get_annual_params(self) -> tuple[float, float, float]:
        """Return learned parameters in human-readable annual units.

        Returns:
            v_yr   (ft/yr)  — seepage velocity
            d_yr   (ft²/yr) — longitudinal dispersion
            lam_yr (1/yr)   — vertical leakage coefficient (already in 1/yr)
        """
        v, D, lam = self.get_physics_params()
        return v.item() * 365.25, D.item() * 365.25, lam.item()


def build_model(unit_cfg: dict, device: torch.device) -> DioxanePINN:
    """Instantiate a fresh DioxanePINN from a unit config dict."""
    v_min, v_max = unit_cfg["v_bounds"]
    d_min, d_max = unit_cfg["d_bounds"]
    lam_min, lam_max = unit_cfg.get("lambda_bounds", [0.0, 0.05])
    return DioxanePINN(v_min, v_max, d_min, d_max, lam_min, lam_max).to(device)


def load_model(unit_cfg: dict, device: torch.device) -> DioxanePINN:
    """Load a pre-trained DioxanePINN from the path in unit_cfg.

    Handles three checkpoint formats:
      - dict with 'model_state_dict' key (standard format)
      - raw state_dict (legacy unit_c3 format)
      - checkpoint without lambda_raw (pre-leakage weights): loads with
        strict=False so lambda_raw initializes to 0 (no leakage assumed).
    """
    model = build_model(unit_cfg, device)
    checkpoint = torch.load(unit_cfg["weights"], map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if "lambda_raw" in str(exc):
            model.load_state_dict(state_dict, strict=False)
            print(
                "Note: checkpoint predates the leakage parameter; "
                "lambda_raw initialized to 0 (maps to midpoint of lambda_bounds). "
                "Retrain to learn leakage from data."
            )
        else:
            raise
    model.eval()
    return model


def save_model(model: DioxanePINN, path: str, extra: dict | None = None) -> None:
    """Save model weights and learned physics parameters to path.

    Always saves as a dict so checkpoint format is consistent.
    """
    v_annual, d_annual, lam_annual = model.get_annual_params()
    payload = {
        "model_state_dict": model.state_dict(),
        "v_annual": v_annual,
        "d_annual": d_annual,
        "lambda_annual": lam_annual,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
