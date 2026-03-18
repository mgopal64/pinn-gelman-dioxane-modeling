"""Leave-One-Well-Out (LOWO) cross-validation for the Gelman PINN.

For each well in the depth-band dataset:
  1. Train a fresh model on all other wells (Adam only, no L-BFGS for speed).
  2. Predict on the held-out well.
  3. Compute metrics in ppb space.

Metrics reported (from the LOWO validation notebook):
  - R² linear / R² log-space
  - RMSE (ppb) / Log-RMSE
  - Nash-Sutcliffe Efficiency (NSE)
  - Log bias / Median APE

Usage:
    python src/evaluate.py --unit unit_e
    python src/evaluate.py --unit unit_e --output figures/lowo_unit_e.csv
    python src/evaluate.py --unit unit_c3 --config configs/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import df_to_tensors, load_data
from src.model import build_model
from src.physics import ade_residual, sample_collocation


def compute_metrics(C_true_ppb: np.ndarray, C_pred_ppb: np.ndarray) -> dict:
    """Compute regression metrics in concentration (ppb) space.

    Args:
        C_true_ppb: Observed concentrations (ppb). Clipped at 0.01 for log safety.
        C_pred_ppb: Predicted concentrations (ppb).

    Returns:
        Dict with R2_linear, R2_log, RMSE_ppb, MAE_ppb, Log_RMSE, NSE,
        Log_Bias, Median_APE_pct keys.
    """
    C_true = np.maximum(C_true_ppb, 0.01)
    C_pred = np.maximum(C_pred_ppb, 0.01)
    log_true = np.log10(C_true)
    log_pred = np.log10(C_pred)

    r2_lin = float(r2_score(C_true, C_pred))
    r2_log = float(r2_score(log_true, log_pred))
    rmse = float(np.sqrt(mean_squared_error(C_true, C_pred)))
    mae = float(np.mean(np.abs(C_pred - C_true)))
    log_rmse = float(np.sqrt(mean_squared_error(log_true, log_pred)))
    log_bias = float(np.mean(log_pred - log_true))
    nse = float(
        1 - np.sum((C_true - C_pred) ** 2) / np.sum((C_true - np.mean(C_true)) ** 2)
    )
    median_ape = float(np.median(np.abs(C_pred - C_true) / C_true) * 100)

    return {
        "R2_linear": r2_lin,
        "R2_log": r2_log,
        "RMSE_ppb": rmse,
        "MAE_ppb": mae,
        "Log_RMSE": log_rmse,
        "NSE": nse,
        "Log_Bias": log_bias,
        "Median_APE_pct": median_ape,
    }


def run_lowo(config: dict, unit_key: str) -> pd.DataFrame:
    """Run LOWO cross-validation and return a per-well results DataFrame.

    Args:
        config:   Loaded config.yaml dict.
        unit_key: 'unit_e' or 'unit_c3'.

    Returns:
        DataFrame sorted by transect distance with one row per evaluated well.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    unit_cfg = config["units"][unit_key]
    lcfg = config["lowo"]
    norm = config["normalization"]
    c_max = np.log10(norm["c_max_raw"])
    depth_range = unit_cfg["max_depth"] - unit_cfg["min_depth"]
    x_max = norm["x_max_dist"]
    t_max = norm["t_max_years"]
    phys_w = lcfg["physics_weight"]

    train_df, _ = load_data(config, unit_key)
    well_list = train_df["Bore"].unique()
    print(f"\n--- LOWO on {len(well_list)} wells [{unit_cfg['name']}] ---\n")

    results = []
    for i, target_well in enumerate(well_list):
        print(f"[{i+1:2d}/{len(well_list)}] Hiding {target_well}...", end=" ", flush=True)

        train_sub = train_df[train_df["Bore"] != target_well]
        test_sub = train_df[train_df["Bore"] == target_well]

        if len(test_sub) < lcfg["min_test_samples"]:
            print("skipped (too few test points)")
            continue

        train_input, C_tr = df_to_tensors(train_sub, device)
        test_input, _ = df_to_tensors(test_sub, device)
        C_true_norm = test_sub["C_norm"].values

        # Fresh collocation per fold — no stale computation graphs
        col_input = sample_collocation(lcfg["n_collocation"], device)

        model = build_model(unit_cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lcfg["adam_lr"])

        model.train()
        for _ in range(lcfg["adam_epochs"]):
            optimizer.zero_grad()
            loss_d = torch.nn.functional.mse_loss(model(train_input), C_tr)
            loss_p = ade_residual(model, col_input, depth_range, x_max, t_max)
            (loss_d + phys_w * loss_p).backward()
            optimizer.step()
            # Re-sample each epoch: broader domain coverage, avoids graph accumulation
            col_input = sample_collocation(lcfg["n_collocation"], device)

        model.eval()
        with torch.no_grad():
            C_pred_norm = model(test_input).cpu().numpy().flatten()

        C_true_ppb = 10 ** (C_true_norm * c_max) - 1
        C_pred_ppb = 10 ** (C_pred_norm * c_max) - 1

        metrics = compute_metrics(C_true_ppb, C_pred_ppb)
        row = {
            "Well": target_well,
            "Dist_ft": float(test_sub["along_dist_ft"].iloc[0]),
            "Screen_ft": float(test_sub["Screen_Midpoint"].iloc[0]),
            "Samples": len(test_sub),
            "Mean_Obs_ppb": float(np.mean(np.maximum(C_true_ppb, 0.01))),
            **metrics,
        }
        results.append(row)

        print(
            f"Screen={row['Screen_ft']:.0f} ft | "
            f"Log-RMSE={metrics['Log_RMSE']:.3f} | "
            f"NSE={metrics['NSE']:.3f} | "
            f"R²(log)={metrics['R2_log']:.3f}"
        )

    res_df = pd.DataFrame(results).sort_values("Dist_ft").reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"LOWO Results [{unit_cfg['name']}]")
    print(f"{'='*60}")
    print(
        res_df[["Well", "Screen_ft", "Mean_Obs_ppb", "RMSE_ppb", "Log_RMSE", "NSE"]]
        .to_string(index=False, float_format="%.3f")
    )
    print(f"\nMedian Log-RMSE : {res_df['Log_RMSE'].median():.3f}")
    print(f"Mean Log-RMSE   : {res_df['Log_RMSE'].mean():.3f}")
    print(f"Median R² (log) : {res_df['R2_log'].median():.3f}")
    print(
        f"Wells < 0.5 Log-RMSE: "
        f"{(res_df['Log_RMSE'] < 0.5).sum()}/{len(res_df)}"
    )
    return res_df


def main():
    parser = argparse.ArgumentParser(
        description="LOWO cross-validation for the Gelman PINN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unit", choices=["unit_e", "unit_c3"], required=True,
        help="Aquifer unit to evaluate",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output", default=None,
        help="Save per-well results to this CSV path",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    res_df = run_lowo(config, args.unit)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        res_df.to_csv(args.output, index=False)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
