"""Data loading, depth filtering, and normalization for the Gelman PINN.

Entry point: load_data(config, unit_key) → (train_df, ic_df)

Design notes:
- Screen_Midpoint is parsed from Screen_Interval strings (e.g. '140-155').
  Rows where the interval cannot be parsed are DROPPED rather than falling
  back to TotalDepth. TotalDepth is often the total borehole depth, not the
  screened interval, and contaminates depth-band filtering with wrong values.
- The IC (initial conditions) DataFrame uses a broader depth/buffer window
  than the unit training data. It anchors the model to the 1986 discovery-era
  spatial distribution across the full aquifer profile.
"""

import numpy as np
import pandas as pd
import torch


def _parse_screen_midpoint(interval) -> float:
    """Parse a Screen_Interval string like '140-155' to its midpoint (147.5).

    Returns NaN if the string is missing, 'unknown', or unparseable.
    """
    if pd.isna(interval) or str(interval).strip().lower() == "unknown":
        return np.nan
    try:
        parts = str(interval).split("-")
        return (float(parts[0]) + float(parts[1])) / 2
    except (ValueError, IndexError):
        return np.nan


def _project_to_transect(
    df: pd.DataFrame,
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project well coordinates onto the source→target transect.

    Returns:
        along: signed distance (ft) from source along the transect.
        perp:  perpendicular offset (ft) from the transect centerline.
    """
    line_vec = target - source
    line_unit = line_vec / np.linalg.norm(line_vec)
    points = df[["Easting", "Northing"]].values.astype(float)
    point_vec = points - source
    along = np.dot(point_vec, line_unit)
    proj = source + np.outer(along, line_unit)
    perp = np.linalg.norm(points - proj, axis=1)
    return along, perp


def _load_dataframe(configured_path: str) -> pd.DataFrame:
    """Load the monitoring data from the best available format.

    Resolution order (checked relative to the configured path's directory):
        1. <stem>.parquet  — preferred: 3× smaller, typed, no proprietary deps
        2. <stem>.csv.gz   — universal fallback, zero dependencies
        3. Configured path — original pkl; requires 'arcgis' if it was serialized
                             with ArcGIS objects. Run scripts/strip_arcgis.py first.

    To generate parquet/csv.gz from the original pkl, run once:
        python scripts/strip_arcgis.py
    """
    import os
    base = os.path.splitext(configured_path)[0]
    # strip a second extension if path ends in .csv.gz
    if base.endswith(".csv"):
        base = os.path.splitext(base)[0]

    parquet_path = base + ".parquet"
    csv_path = base + ".csv.gz"

    if os.path.exists(parquet_path):
        print(f"Loading {parquet_path}...")
        return pd.read_parquet(parquet_path)

    if os.path.exists(csv_path):
        print(f"Loading {csv_path}...")
        return pd.read_csv(csv_path, compression="gzip")

    # Fall back to the configured path (pkl or otherwise)
    print(f"Loading {configured_path}...")
    try:
        return pd.read_pickle(configured_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to load {configured_path}: {exc}\n"
            "The file was serialized in an ArcGIS environment. "
            "Run  python scripts/strip_arcgis.py  to produce a "
            "dependency-free parquet file, then retry."
        ) from exc


def load_raw(config: dict) -> pd.DataFrame:
    """Load the pickled monitoring DataFrame and add transect + screen columns.

    Adds columns:
        along_dist_ft, perp_dist_ft  — transect position
        Screen_Midpoint              — parsed from Screen_Interval

    Rows where Screen_Interval cannot be parsed are dropped.
    """
    df = _load_dataframe(config["data"]["pkl_path"])

    df["Easting"] = pd.to_numeric(df["Easting"], errors="coerce")
    df["Northing"] = pd.to_numeric(df["Northing"], errors="coerce")
    df.dropna(subset=["Easting", "Northing"], inplace=True)

    source = np.array(config["coordinates"]["source"], dtype=float)
    target = np.array(config["coordinates"]["target"], dtype=float)
    df["along_dist_ft"], df["perp_dist_ft"] = _project_to_transect(df, source, target)

    df["Screen_Midpoint"] = df.apply(
        lambda row: _parse_screen_midpoint(row["Screen_Interval"]), axis=1
    )
    n_before = len(df)
    df.dropna(subset=["Screen_Midpoint"], inplace=True)
    print(
        f"Screen interval filter: {n_before} → {len(df)} rows "
        f"({n_before - len(df)} dropped, unparseable Screen_Interval)."
    )

    df["SampleDate"] = pd.to_datetime(df["SampleDate"])
    return df


def filter_unit(df: pd.DataFrame, unit_cfg: dict, cfg: dict) -> pd.DataFrame:
    """Filter the raw DataFrame to a specific aquifer depth band."""
    min_d = unit_cfg["min_depth"]
    max_d = unit_cfg["max_depth"]
    buf = cfg["data"]["perp_buffer_ft"]
    min_val = cfg["data"]["min_value_ppb"]

    mask = (
        df["Screen_Midpoint"].between(min_d, max_d)
        & (df["perp_dist_ft"].abs() <= buf)
        & (df["along_dist_ft"] >= -100)
        & df["Value"].notna()
        & (df["Value"] > min_val)
    )
    if cfg["data"].get("drop_not_detected", True):
        mask &= df["Value_tx"] != "Not Detected"

    return df[mask].copy()


def normalize(df: pd.DataFrame, unit_cfg: dict, cfg: dict) -> pd.DataFrame:
    """Add normalized columns x_norm, z_norm, t_norm, C_norm to df.

    Normalization equations (from CLAUDE.md):
        x_norm = along_dist_ft / X_MAX_DIST
        z_norm = (depth_ft - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        t_norm = years_since_1986 / T_MAX_YEARS
        C_norm = log10(value + 1) / log10(C_MAX_RAW)
    """
    t_start = pd.Timestamp(cfg["data"]["t_start"])
    x_max = cfg["normalization"]["x_max_dist"]
    t_max = cfg["normalization"]["t_max_years"]
    c_max = np.log10(cfg["normalization"]["c_max_raw"])
    min_d = unit_cfg["min_depth"]
    depth_range = unit_cfg["max_depth"] - unit_cfg["min_depth"]

    df = df.copy()
    df["t_years"] = (df["SampleDate"] - t_start).dt.days / 365.25
    df["x_norm"] = df["along_dist_ft"] / x_max
    df["z_norm"] = (df["Screen_Midpoint"] - min_d) / depth_range
    df["t_norm"] = df["t_years"] / t_max
    df["C_norm"] = np.log10(df["Value"] + 1) / c_max
    return df


def load_ic_data(raw_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Extract 1986 initial-condition anchors from the raw DataFrame.

    Uses a broader depth/buffer window than unit training data (the "Discovery
    Era" window). Treats "Not Detected" readings as 0 ppb to anchor the clean
    outer boundaries of the plume.

    Returns a normalized DataFrame with x_norm, z_norm (broad window), t_norm=0,
    and C_norm columns.
    """
    ic_cfg = cfg["initial_conditions"]
    cutoff = pd.Timestamp(f"{ic_cfg['year_cutoff']}-01-01")

    ic_df = raw_df[
        (raw_df["SampleDate"] < cutoff)
        & (raw_df["perp_dist_ft"].abs() <= ic_cfg["perp_buffer_ft"])
        & raw_df["Screen_Midpoint"].between(ic_cfg["depth_min"], ic_cfg["depth_max"])
    ].copy()

    ic_df.loc[ic_df["Value_tx"] == "Not Detected", "Value"] = 0.0
    ic_df["Value"] = ic_df["Value"].fillna(0.0)

    x_max = cfg["normalization"]["x_max_dist"]
    c_max = np.log10(cfg["normalization"]["c_max_raw"])
    depth_range = ic_cfg["depth_max"] - ic_cfg["depth_min"]
    min_d = ic_cfg["depth_min"]

    ic_df["x_norm"] = ic_df["along_dist_ft"] / x_max
    ic_df["z_norm"] = (ic_df["Screen_Midpoint"] - min_d) / depth_range
    ic_df["t_norm"] = 0.0
    ic_df["C_norm"] = np.log10(ic_df["Value"] + 1) / c_max
    return ic_df


def load_data(config: dict, unit_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point: load, filter, and normalize data for one aquifer unit.

    Args:
        config: Loaded config.yaml dict.
        unit_key: 'unit_e' or 'unit_c3'.

    Returns:
        train_df: Normalized training DataFrame (depth-band filtered).
        ic_df:    Normalized 1986 IC anchors (broad discovery-era window).
    """
    raw_df = load_raw(config)
    unit_cfg = config["units"][unit_key]

    train_df = filter_unit(raw_df, unit_cfg, config)
    train_df = normalize(train_df, unit_cfg, config)
    ic_df = load_ic_data(raw_df, config)

    print(
        f"[{unit_cfg['name']}] "
        f"{len(train_df)} training samples from {train_df['Bore'].nunique()} wells | "
        f"{len(ic_df)} IC anchors"
    )
    return train_df, ic_df


def df_to_tensors(
    df: pd.DataFrame,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a normalized DataFrame to (input, target) tensors.

    Args:
        df: DataFrame with x_norm, z_norm, t_norm, C_norm columns.
        device: Target device.

    Returns:
        inp: [N, 3] float32 tensor of (x_norm, z_norm, t_norm).
        tgt: [N, 1] float32 tensor of C_norm.
    """
    inp = torch.tensor(
        df[["x_norm", "z_norm", "t_norm"]].values.astype(np.float32),
        device=device,
        dtype=torch.float32,
    )
    tgt = torch.tensor(
        df["C_norm"].values.astype(np.float32).reshape(-1, 1),
        device=device,
        dtype=torch.float32,
    )
    return inp, tgt
