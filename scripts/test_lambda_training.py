"""100-epoch training smoke test to verify lambda learns from initialization."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from src.data_loader import load_data, df_to_tensors
from src.model import build_model
from src.physics import ade_residual, sample_collocation

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

unit_key = "unit_e"
unit_cfg = config["units"][unit_key]
norm     = config["normalization"]
device   = torch.device("cpu")

depth_range = unit_cfg["max_depth"] - unit_cfg["min_depth"]
x_max       = norm["x_max_dist"]
t_max       = norm["t_max_years"]

train_df, _ = load_data(config, unit_key)
inputs, C_tr = df_to_tensors(train_df, device)

model     = build_model(unit_cfg, device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Record initialization
v0, D0, lam0 = model.get_annual_params()
print(f"Init  :  v={v0:.3f} ft/yr  D={D0:.3f} ft²/yr  λ={lam0:.6f} /yr")
print(f"         (lambda_raw={model.lambda_raw.item():.4f}  → maps to midpoint of bounds)")
print()

col_input = sample_collocation(config["training"]["n_collocation"], device)

model.train()
for epoch in range(1, 101):
    optimizer.zero_grad()
    loss_d = torch.nn.functional.mse_loss(model(inputs), C_tr)
    loss_p = ade_residual(model, col_input, depth_range, x_max, t_max)
    loss   = loss_d + config["training"]["physics_weight"] * loss_p
    loss.backward()
    optimizer.step()
    col_input = sample_collocation(config["training"]["n_collocation"], device)

    if epoch % 25 == 0:
        v, D, lam = model.get_annual_params()
        print(f"epoch {epoch:3d}  loss={loss.item():.5f}  "
              f"v={v:.3f} ft/yr  D={D:.3f} ft²/yr  λ={lam:.6f} /yr")

v1, D1, lam1 = model.get_annual_params()
print(f"\nFinal :  v={v1:.3f} ft/yr  D={D1:.3f} ft²/yr  λ={lam1:.6f} /yr")
print(f"Δv={v1-v0:+.3f}  ΔD={D1-D0:+.3f}  Δλ={lam1-lam0:+.7f}")
print(f"lambda_raw: {lam0:.4f} → {model.lambda_raw.item():.4f}  "
      f"({'moved' if abs(model.lambda_raw.item()) > 1e-4 else 'STUCK — check grad flow'})")
