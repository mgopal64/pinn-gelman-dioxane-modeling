"""Convert the arcgis-serialized merged_df.pkl to dependency-free formats.

The original pickle embeds two arcgis types:
    arcgis.features.geo._array.GeoArray  — the geometry column dtype
    arcgis.geometry._types.Point         — individual geometry values

Neither is used by the PINN (Easting/Northing are already regular float
columns). This script stubs those classes so the pickle can be unpickled
without arcgis installed, drops the SHAPE column, and saves:

    data/processed/merged_df.parquet    — primary (3× smaller, typed, fast)
    data/processed/merged_df.csv.gz     — fallback (zero dependencies)

The original merged_df.pkl is left untouched.

Usage:
    python scripts/strip_arcgis.py
    python scripts/strip_arcgis.py --input data/processed/merged_df.pkl
    python scripts/strip_arcgis.py --input data/processed/merged_df.pkl \\
                                   --out-dir data/processed
"""

import argparse
import os
import sys
import types

import numpy as np
import pandas as pd


# ── Minimal arcgis stubs ──────────────────────────────────────────────────────
# Injected into sys.modules before unpickling so pickle.find_class succeeds.
# The real arcgis GeoArray is a pandas ExtensionArray; we mimic just enough of
# the interface for the DataFrame BlockManager to reconstruct without errors.

class _PointStub(dict):
    """Stub for arcgis.geometry._types.Point (serialized as a dict subclass)."""
    pass


class _GeoExtDtype(pd.api.extensions.ExtensionDtype):
    name = "geometry"
    na_value = None
    type = object

    @classmethod
    def construct_array_type(cls):
        return _GeoArrayStub


class _GeoArrayStub(pd.api.extensions.ExtensionArray):
    """Stub for arcgis.features.geo._array.GeoArray.

    pickle uses __new__ + __setstate__ (not __init__), so we must not assume
    any attribute is set. The real class stores data in self.data.
    """
    dtype = _GeoExtDtype()

    def _items(self) -> list:
        for attr in ("data", "_data", "_values", "values", "_ndarray"):
            v = self.__dict__.get(attr)
            if v is not None:
                return list(v)
        return []

    def __len__(self) -> int:
        return len(self._items())

    def __getitem__(self, idx):
        items = self._items()
        if isinstance(idx, int):
            return items[idx]
        return _GeoArrayStub._from_list(np.asarray(items, dtype=object)[idx].tolist())

    def __setitem__(self, idx, val) -> None:
        pass  # absorb without error during block reconstruction

    def isna(self) -> np.ndarray:
        return np.zeros(len(self), dtype=bool)

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        items = np.asarray(self._items(), dtype=object)
        return _GeoArrayStub._from_list(items[indices].tolist())

    def copy(self):
        return _GeoArrayStub._from_list(self._items())

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls._from_list(list(scalars))

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_list(list(values))

    def _values_for_factorize(self):
        return np.array(self._items(), dtype=object), None

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls._from_list([x for a in to_concat for x in a._items()])

    @classmethod
    def _from_list(cls, lst: list) -> "_GeoArrayStub":
        obj = cls.__new__(cls)
        obj.data = lst
        return obj


def _inject_stubs() -> None:
    """Register stub modules in sys.modules before unpickling."""
    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _mod("arcgis")
    _mod("arcgis.geometry")
    _mod("arcgis.features")
    _mod("arcgis.features.geo")
    _mod("arcgis.geometry._types").Point = _PointStub
    _mod("arcgis.features.geo._array").GeoArray = _GeoArrayStub


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(input_path: str, out_dir: str) -> None:
    _inject_stubs()

    print(f"Loading {input_path} ...")
    df = pd.read_pickle(input_path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Drop the geometry column(s) — any column whose dtype is our stub dtype
    geo_cols = [c for c in df.columns if df[c].dtype.name == "geometry"]
    if geo_cols:
        df = df.drop(columns=geo_cols)
        print(f"  Dropped arcgis geometry column(s): {geo_cols}")
    else:
        print("  No geometry columns found — DataFrame is already clean.")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Parquet — primary output
    parquet_path = os.path.join(out_dir, "merged_df.parquet")
    df.to_parquet(parquet_path, index=False)
    size_kb = os.path.getsize(parquet_path) / 1024
    print(f"\nWrote parquet  → {parquet_path}  ({size_kb:.0f} KB)")

    # 2. CSV (gzip) — universal fallback
    csv_path = os.path.join(out_dir, "merged_df.csv.gz")
    df.to_csv(csv_path, index=False, compression="gzip")
    size_kb = os.path.getsize(csv_path) / 1024
    print(f"Wrote csv.gz   → {csv_path}  ({size_kb:.0f} KB)")

    orig_kb = os.path.getsize(input_path) / 1024
    print(f"\nOriginal pickle: {orig_kb:.0f} KB  →  parquet: {os.path.getsize(parquet_path)/1024:.0f} KB  "
          f"({(1 - os.path.getsize(parquet_path)/os.path.getsize(input_path))*100:.0f}% reduction)")
    print("\nDone. Update configs/config.yaml data.pkl_path → data/processed/merged_df.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip arcgis types from merged_df.pkl")
    parser.add_argument(
        "--input", default="data/processed/merged_df.pkl",
        help="Path to the arcgis-serialized pickle (default: data/processed/merged_df.pkl)",
    )
    parser.add_argument(
        "--out-dir", default="data/processed",
        help="Output directory for parquet and csv.gz files",
    )
    args = parser.parse_args()
    convert(args.input, args.out_dir)


if __name__ == "__main__":
    main()
