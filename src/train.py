"""Training pipeline for the Gelman PINN: Adam + L-BFGS refinement.

Loss terms:
    - Data loss:     MSE on training observations
    - IC loss:       MSE on 1986 initial-condition anchors
    - Physics loss:  Mean-squared ADE residual at collocation points (weight 5.0×)
    - BC loss:       Source boundary condition — model(x=0, z, t) ≈ 1.0

Usage:
    python src/train.py --unit unit_e
    python src/train.py --unit unit_c3 --adam-epochs 5000 --no-lbfgs
    python src/train.py --unit unit_e --output models/custom_unit_e.pth
    python src/train.py --unit unit_e --config configs/config.yaml
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import df_to_tensors, load_data
from src.model import build_model, save_model
from src.physics import ade_residual, sample_collocation


def _bc_tensors(n: int, device: torch.device) -> torch.Tensor:
    """Boundary condition inputs: source at x=0, random z and t."""
    return torch.cat(
        [
            torch.zeros(n, 1, device=device),
            torch.rand(n, 1, device=device),
            torch.rand(n, 1, device=device),
        ],
        dim=1,
    )


def _total_loss(
    model,
    train_input: torch.Tensor,
    C_train: torch.Tensor,
    col_input: torch.Tensor,
    ic_input: torch.Tensor,
    C_ic: torch.Tensor,
    bc_input: torch.Tensor,
    physics_weight: float,
    depth_range: float,
    x_max: float,
    t_max: float,
) -> torch.Tensor:
    loss_data = torch.nn.functional.mse_loss(model(train_input), C_train)
    loss_ic = torch.nn.functional.mse_loss(model(ic_input), C_ic)
    loss_phys = ade_residual(model, col_input, depth_range, x_max, t_max)
    # BC: concentration at the source boundary (x=0) should be near maximum (1.0)
    loss_bc = torch.nn.functional.mse_loss(
        model(bc_input), torch.ones(bc_input.shape[0], 1, device=bc_input.device)
    )
    return loss_data + loss_ic + physics_weight * loss_phys + loss_bc


def train(
    config: dict,
    unit_key: str,
    output_path: str | None = None,
    adam_epochs: int | None = None,
    run_lbfgs: bool = True,
) -> object:
    """Train a DioxanePINN for the specified aquifer unit.

    Args:
        config:       Loaded config.yaml dict.
        unit_key:     'unit_e' or 'unit_c3'.
        output_path:  Where to save the trained model. Defaults to
                      the path in config[units][unit_key][weights].
        adam_epochs:  Override config training.adam_epochs.
        run_lbfgs:    Whether to run the L-BFGS refinement stage.

    Returns:
        Trained DioxanePINN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    unit_cfg = config["units"][unit_key]
    tcfg = config["training"]
    norm = config["normalization"]

    # ── Data ────────────────────────────────────────────────────────────────
    train_df, ic_df = load_data(config, unit_key)
    train_input, C_train = df_to_tensors(train_df, device)
    ic_input, C_ic = df_to_tensors(ic_df, device)

    depth_range = unit_cfg["max_depth"] - unit_cfg["min_depth"]
    x_max = norm["x_max_dist"]
    t_max = norm["t_max_years"]
    phys_w = tcfg["physics_weight"]

    # ── Collocation & boundary tensors ──────────────────────────────────────
    col_input = sample_collocation(tcfg["n_collocation"], device)
    bc_input = _bc_tensors(100, device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(unit_cfg, device)
    n_adam = adam_epochs if adam_epochs is not None else tcfg["adam_epochs"]

    # ── Stage 1: Adam ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["adam_lr"])
    print(f"\nStage 1: Adam — {n_adam} epochs")
    model.train()
    for epoch in range(n_adam + 1):
        optimizer.zero_grad()
        loss = _total_loss(
            model, train_input, C_train, col_input,
            ic_input, C_ic, bc_input,
            phys_w, depth_range, x_max, t_max,
        )
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            v_yr, d_yr, lam_yr = model.get_annual_params()
            print(
                f"  Epoch {epoch:6d} | Loss {loss.item():.6f} "
                f"| v = {v_yr:.1f} ft/yr | D = {d_yr:.1f} ft²/yr | λ = {lam_yr:.5f} /yr"
            )

    # ── Stage 2: L-BFGS ──────────────────────────────────────────────────────
    if run_lbfgs:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=tcfg["lbfgs_max_iter"],
            history_size=tcfg["lbfgs_history_size"],
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            loss = _total_loss(
                model, train_input, C_train, col_input,
                ic_input, C_ic, bc_input,
                phys_w, depth_range, x_max, t_max,
            )
            loss.backward()
            return loss

        print("\nStage 2: L-BFGS refinement...")
        model.train()
        optimizer_lbfgs.step(closure)

    # ── Final report ─────────────────────────────────────────────────────────
    v_yr, d_yr, lam_yr = model.get_annual_params()
    print(f"\n{'='*50}")
    print(f"Converged physics parameters [{unit_cfg['name']}]")
    print(f"  Velocity (v):    {v_yr:.2f} ft/yr")
    print(f"  Dispersion (D):  {d_yr:.2f} ft²/yr")
    print(f"  Dispersivity:    {d_yr/v_yr:.2f} ft  (αL = D/v)")
    print(f"  Leakage (λ):     {lam_yr:.5f} /yr  (half-life ≈ {0.693/lam_yr:.1f} yr)" if lam_yr > 0 else f"  Leakage (λ):     0.0 /yr")
    print(f"{'='*50}")

    out_path = output_path or unit_cfg["weights"]
    save_model(model, out_path)
    print(f"Model saved → {out_path}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train the Gelman 1,4-Dioxane PINN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unit", choices=["unit_e", "unit_c3"], required=True,
        help="Aquifer unit to train (unit_e = 130-170 ft, unit_c3 = 50-90 ft)",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--adam-epochs", type=int, default=None,
        help="Override config training.adam_epochs",
    )
    parser.add_argument(
        "--no-lbfgs", action="store_true",
        help="Skip the L-BFGS refinement stage",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output .pth path (default: path from config)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(
        config,
        args.unit,
        output_path=args.output,
        adam_epochs=args.adam_epochs,
        run_lbfgs=not args.no_lbfgs,
    )


if __name__ == "__main__":
    main()
