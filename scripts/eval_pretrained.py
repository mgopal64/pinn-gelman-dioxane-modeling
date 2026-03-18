"""Quick evaluation of pretrained weights + single-well LOWO sanity check."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from sklearn.metrics import r2_score, mean_squared_error

from src.data_loader import load_data, df_to_tensors
from src.model import load_model


def ppb(C_norm, c_max):
    return 10 ** (C_norm * c_max) - 1


def print_metrics(C_true_ppb, C_pred_ppb):
    r2_lin = r2_score(C_true_ppb, C_pred_ppb)
    r2_log = r2_score(
        np.log10(np.maximum(C_true_ppb, 0.01)),
        np.log10(np.maximum(C_pred_ppb, 0.01)),
    )
    rmse = np.sqrt(mean_squared_error(C_true_ppb, C_pred_ppb))
    mae  = np.mean(np.abs(C_pred_ppb - C_true_ppb))
    print(f"  R² (linear) : {r2_lin:.4f}")
    print(f"  R² (log)    : {r2_log:.4f}")
    print(f"  RMSE        : {rmse:.2f} ppb")
    print(f"  MAE         : {mae:.2f} ppb")
    return r2_lin


with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

c_max  = np.log10(config["normalization"]["c_max_raw"])
device = torch.device("cpu")

# ── Full-data evaluation ──────────────────────────────────────────────────────
print("\n=== FULL-DATA EVALUATION (pretrained weights) ===")
for unit_key in ("unit_e", "unit_c3"):
    unit_cfg = config["units"][unit_key]
    df, _    = load_data(config, unit_key)
    model    = load_model(unit_cfg, device)
    model.eval()

    inputs, _ = df_to_tensors(df, device)
    with torch.no_grad():
        C_pred_norm = model(inputs).cpu().numpy().flatten()

    C_true = ppb(df["C_norm"].values, c_max)
    C_pred = ppb(C_pred_norm, c_max)

    v_yr, d_yr, lam = model.get_annual_params()
    print(f"\n{unit_cfg['name']}  —  {len(df)} samples, {df['Bore'].nunique()} wells")
    print(f"  Learned: v={v_yr:.1f} ft/yr  D={d_yr:.1f} ft²/yr  λ={lam:.5f} /yr")
    print_metrics(C_true, C_pred)

# ── Single-well LOWO sanity check (Unit E, most-sampled well) ─────────────────
print("\n=== LOWO SANITY CHECK (Unit E, pretrained weights — no retraining) ===")
unit_cfg = config["units"]["unit_e"]
df_e, _  = load_data(config, "unit_e")

target_well = df_e["Bore"].value_counts().index[0]
test_df     = df_e[df_e["Bore"] == target_well]
print(f"Held-out well : {target_well}  ({len(test_df)} samples)")
print(f"Dist along transect : {test_df['along_dist_ft'].iloc[0]:.0f} ft downgradient")

model = load_model(unit_cfg, device)
model.eval()

inputs_t, _ = df_to_tensors(test_df, device)
with torch.no_grad():
    C_pred_norm = model(inputs_t).cpu().numpy().flatten()

C_true = ppb(test_df["C_norm"].values, c_max)
C_pred = ppb(C_pred_norm, c_max)

print_metrics(C_true, C_pred)
print(f"  Obs  range  : {C_true.min():.1f} – {C_true.max():.1f} ppb  (mean {C_true.mean():.1f})")
print(f"  Pred range  : {C_pred.min():.1f} – {C_pred.max():.1f} ppb  (mean {C_pred.mean():.1f})")
