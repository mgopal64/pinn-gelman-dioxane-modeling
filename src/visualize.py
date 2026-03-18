"""Visualization and forecast plotting for the Gelman PINN.

Plots available (--plot argument):
    forecast       — concentration vs distance curves for multiple years
    cross_section  — 2D depth × distance heatmap for one year
    surface        — 3D plume topography surface for one year
    temporal       — geographic scatter panels for three time slices
    heatmap        — folium interactive geographic heatmap (requires folium)
    all            — run forecast, cross_section, surface, temporal

Usage:
    python src/visualize.py --unit unit_e --plot forecast
    python src/visualize.py --unit unit_e --plot all --year 2040
    python src/visualize.py --unit unit_e --plot heatmap --year 2080
    python src/visualize.py --unit unit_e --plot forecast --output figures/forecast.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model


def _denorm_concentration(C_norm: np.ndarray, c_max_raw: int) -> np.ndarray:
    """Convert normalized log-concentration back to ppb."""
    c_log_max = np.log10(c_max_raw)
    return 10 ** (C_norm * c_log_max) - 1


def _inference_transect(
    model, year: int, z_norm: float, config: dict, n_pts: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference along the full transect at a fixed depth and year.

    Returns:
        x_feet: [n_pts] array of distances in feet.
        c_ppb:  [n_pts] array of concentrations in ppb.
    """
    t_max = config["normalization"]["t_max_years"]
    x_max = config["normalization"]["x_max_dist"]
    c_max_raw = config["normalization"]["c_max_raw"]
    t_start_year = int(config["data"]["t_start"][:4])

    x_norm = torch.linspace(0, 1, n_pts, device=device).view(-1, 1)
    z_fixed = torch.ones_like(x_norm) * z_norm
    t_fixed = torch.ones_like(x_norm) * ((year - t_start_year) / t_max)

    model.eval()
    with torch.no_grad():
        c_norm = model(torch.cat([x_norm, z_fixed, t_fixed], dim=1)).cpu().numpy()

    x_feet = x_norm.cpu().numpy().flatten() * x_max
    c_ppb = _denorm_concentration(c_norm.flatten(), c_max_raw)
    return x_feet, c_ppb


def plot_forecast(
    model,
    config: dict,
    unit_key: str,
    years: list[int] | None = None,
    output_path: str | None = None,
) -> None:
    """Concentration vs distance forecast curves for multiple years.

    Plots log-scale ppb on the y-axis with the Michigan regulatory limit
    (7.2 ppb) marked as a reference line.
    """
    device = next(model.parameters()).device
    unit_cfg = config["units"][unit_key]
    reg_limit = config["geography"]["regulatory_limit_ppb"]

    if years is None:
        years = [1990, 2024, 2040, 2056]

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(years)))

    plt.figure(figsize=(12, 7))
    for yr, color in zip(years, colors):
        x_feet, c_ppb = _inference_transect(model, yr, 0.5, config, 500, device)
        plt.plot(x_feet, c_ppb, label=str(yr), color=color, linewidth=2)

    plt.axhline(
        reg_limit, color="black", linewidth=1.5, linestyle="--",
        label=f"EGLE limit ({reg_limit} ppb)"
    )
    plt.fill_between(
        [0, config["normalization"]["x_max_dist"]], 0, reg_limit,
        color="gray", alpha=0.08
    )

    plt.yscale("log")
    plt.ylim(0.1, 1e5)
    plt.xlabel("Distance along transect (ft)")
    plt.ylabel("1,4-Dioxane concentration (ppb)")
    plt.title(f"Plume forecast — {unit_cfg['name']}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    else:
        plt.show()
    plt.close()


def plot_cross_section(
    model,
    config: dict,
    unit_key: str,
    year: int = 2024,
    output_path: str | None = None,
) -> None:
    """2D vertical cross-section heatmap: depth × transect distance."""
    device = next(model.parameters()).device
    unit_cfg = config["units"][unit_key]
    t_max = config["normalization"]["t_max_years"]
    x_max = config["normalization"]["x_max_dist"]
    t_start_year = int(config["data"]["t_start"][:4])
    c_max_raw = config["normalization"]["c_max_raw"]

    x_res, z_res = 200, 100
    x_norm = np.linspace(0, 1, x_res)
    z_norm = np.linspace(0, 1, z_res)
    X, Z = np.meshgrid(x_norm, z_norm)
    t_val = (year - t_start_year) / t_max

    inp = torch.tensor(
        np.stack([X.ravel(), Z.ravel(), np.full(X.size, t_val)], axis=1),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        C_pred = model(inp).cpu().numpy().reshape(z_res, x_res)

    min_d = unit_cfg["min_depth"]
    max_d = unit_cfg["max_depth"]
    x_feet = X * x_max
    z_feet = min_d + Z * (max_d - min_d)

    plt.figure(figsize=(15, 6))
    hm = plt.pcolormesh(x_feet, z_feet, C_pred, shading="gouraud", cmap="magma")
    plt.colorbar(hm, label="normalized log-concentration")
    plt.gca().invert_yaxis()
    plt.title(f"vertical cross-section — {unit_cfg['name']} ({year})")
    plt.xlabel("distance along transect (ft)")
    plt.ylabel("aquifer depth (ft)")
    plt.axvline(x=0.45 * x_max, color="white", linestyle="--", alpha=0.5, label="dip zone")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    else:
        plt.show()
    plt.close()


def plot_3d_surface(
    model,
    config: dict,
    unit_key: str,
    year: int = 2024,
    output_path: str | None = None,
) -> None:
    """3D surface plot of the plume topography at a given year."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    device = next(model.parameters()).device
    unit_cfg = config["units"][unit_key]
    t_max = config["normalization"]["t_max_years"]
    x_max = config["normalization"]["x_max_dist"]
    t_start_year = int(config["data"]["t_start"][:4])

    x_res, z_res = 100, 50
    x_norm = np.linspace(0, 1, x_res)
    z_norm = np.linspace(0, 1, z_res)
    X, Z = np.meshgrid(x_norm, z_norm)
    t_val = (year - t_start_year) / t_max

    inp = torch.tensor(
        np.stack([X.ravel(), Z.ravel(), np.full(X.size, t_val)], axis=1),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        C_pred = model(inp).cpu().numpy().reshape(z_res, x_res)

    min_d = unit_cfg["min_depth"]
    max_d = unit_cfg["max_depth"]
    x_feet = X * x_max
    z_feet = min_d + Z * (max_d - min_d)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x_feet, z_feet, C_pred, cmap="magma", edgecolor="none", alpha=0.85)

    ax.set_title(f"3d plume topography — {unit_cfg['name']} ({year})", fontsize=14)
    ax.set_xlabel("distance along transect (ft)", labelpad=10)
    ax.set_ylabel("aquifer depth (ft)", labelpad=10)
    ax.set_zlabel("normalized log-concentration", labelpad=10)
    ax.set_ylim(max_d, min_d)
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(surf, shrink=0.5, aspect=10, label="log-concentration")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved → {output_path}")
    else:
        plt.show()
    plt.close()


def plot_temporal_evolution(
    model,
    config: dict,
    unit_key: str,
    years: list[int] | None = None,
    output_path: str | None = None,
) -> None:
    """Geographic scatter panels showing plume evolution across time slices."""
    device = next(model.parameters()).device
    unit_cfg = config["units"][unit_key]
    c_max_raw = config["normalization"]["c_max_raw"]

    if years is None:
        years = [1986, 2024, 2056]

    source = config["coordinates"]["source"]
    target = config["coordinates"]["target"]

    fig, axes = plt.subplots(1, len(years), figsize=(7 * len(years), 8), sharey=True)
    if len(years) == 1:
        axes = [axes]

    sc = None
    for ax, year in zip(axes, years):
        x_feet, c_ppb = _inference_transect(model, year, 0.5, config, 500, device)
        x_norm_vals = x_feet / config["normalization"]["x_max_dist"]

        geo_e = source[0] + x_norm_vals * (target[0] - source[0])
        geo_n = source[1] + x_norm_vals * (target[1] - source[1])

        sc = ax.scatter(
            geo_e, geo_n, c=np.maximum(c_ppb, 1.0), cmap="jet",
            norm=LogNorm(vmin=1, vmax=1e5), s=30
        )
        ax.scatter(source[0], source[1], marker="X", color="red", s=200, label="source")
        ax.scatter(target[0], target[1], marker="*", color="green", s=250, label="intake")
        ax.set_title(f"year: {year}", fontsize=14, fontweight="bold")
        ax.set_xlabel("easting (ft)")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("northing (ft)")
    plt.suptitle(
        f"evolution of the 1,4-dioxane plume — {unit_cfg['name']}",
        fontsize=18, y=1.02
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="1,4-dioxane concentration (ppb)")
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")
    else:
        plt.show()
    plt.close()


def create_geographic_heatmap(
    model,
    config: dict,
    unit_key: str,
    year: int = 2056,
    output_path: str | None = None,
):
    """Create a folium interactive geographic heatmap overlay.

    Requires: folium, branca  (pip install folium branca)

    Args:
        output_path: Path for the output .html file. If None, returns the map object.
    """
    try:
        import folium
        from branca.element import Element
        from folium import plugins
    except ImportError:
        print("folium and branca are required: pip install folium branca")
        return None

    device = next(model.parameters()).device
    reg_limit = config["geography"]["regulatory_limit_ppb"]
    source_ll = config["geography"]["source_latlon"]
    target_ll = config["geography"]["target_latlon"]
    unit_cfg = config["units"][unit_key]

    _, c_ppb = _inference_transect(model, year, 0.5, config, 100, device)
    lats = np.linspace(source_ll[0], target_ll[0], 100)
    lons = np.linspace(source_ll[1], target_ll[1], 100)
    heat_data = [
        [float(lats[i]), float(lons[i]), float(c_ppb[i] / 1000)]
        for i in range(100) if c_ppb[i] > 1.0
    ]

    m = folium.Map(location=[42.310, -83.770], zoom_start=13, tiles="CartoDB positron")
    plugins.HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(m)

    # Mark the regulatory front
    front_idx = int(np.abs(c_ppb - reg_limit).argmin())
    if c_ppb[front_idx] >= reg_limit * 0.9:
        folium.Circle(
            location=[float(lats[front_idx]), float(lons[front_idx])],
            radius=250, color="red", fill=True, weight=2,
            popup=f"{reg_limit} ppb front ({year})"
        ).add_to(m)

    folium.Marker(
        source_ll, popup="gelman source",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    folium.Marker(
        target_ll, popup="barton pond intake",
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:50px;left:50px;width:175px;
                background:white;border:2px solid grey;z-index:9999;
                font-size:12px;padding:10px;">
      <b>{unit_cfg['name']} — {year}</b><br>
      <span style="color:red">■</span> high concentration<br>
      <span style="color:yellow">■</span> medium<br>
      <span style="color:blue">■</span> low (&gt;1.0 ppb)<br>
      <span style="border:2px solid red;border-radius:50%;
                   display:inline-block;width:8px;height:8px;"></span>
      {reg_limit} ppb front
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    if output_path:
        m.save(output_path)
        print(f"Saved → {output_path}")
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Gelman PINN forecast",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unit", choices=["unit_e", "unit_c3"], required=True,
        help="Aquifer unit to visualize",
    )
    parser.add_argument(
        "--plot",
        choices=["forecast", "cross_section", "surface", "temporal", "heatmap", "all"],
        default="forecast",
    )
    parser.add_argument(
        "--year", type=int, default=2024,
        help="Target year for single-year plots (cross_section, surface, heatmap)",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--output", default=None,
        help="Output file path (.png for plots, .html for heatmap). "
             "If omitted, displays interactively.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit_cfg = config["units"][args.unit]
    model = load_model(unit_cfg, device)

    v_yr, d_yr, lam_yr = model.get_annual_params()
    print(f"Loaded {unit_cfg['name']} | v = {v_yr:.1f} ft/yr | D = {d_yr:.1f} ft²/yr | λ = {lam_yr:.5f} /yr")

    def _out(suffix):
        if args.output:
            base, ext = os.path.splitext(args.output)
            return f"{base}_{suffix}{ext}" if args.plot == "all" else args.output
        return None

    if args.plot in ("forecast", "all"):
        plot_forecast(model, config, args.unit, output_path=_out("forecast"))

    if args.plot in ("cross_section", "all"):
        plot_cross_section(model, config, args.unit, year=args.year, output_path=_out("cross_section"))

    if args.plot in ("surface", "all"):
        plot_3d_surface(model, config, args.unit, year=args.year, output_path=_out("surface"))

    if args.plot in ("temporal", "all"):
        plot_temporal_evolution(model, config, args.unit, output_path=_out("temporal"))

    if args.plot == "heatmap":
        out = args.output or f"figures/{args.unit}_heatmap_{args.year}.html"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        create_geographic_heatmap(model, config, args.unit, year=args.year, output_path=out)


if __name__ == "__main__":
    main()
