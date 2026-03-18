# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a **Physics-Informed Neural Network (PINN)** to model and forecast 1,4-Dioxane contamination migration at the Gelman Sciences site in Ann Arbor, Michigan. The PINN uses 40 years of monitoring well data (1986–2026) to predict the contaminant plume's movement toward the Barton Pond drinking water intake.

## Running the Code

All code lives in Jupyter notebooks — there is no traditional package structure. The primary runtime environment is **Google Colab** (GPU access, Google Drive for model weights). Local Jupyter is also supported.

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn matplotlib folium

# Launch notebooks locally
jupyter notebook notebooks/Gelman_PINN_Showcase.ipynb        # Inference & visualization
jupyter notebook notebooks/GelmanPINNLOWOValidation.ipynb    # LOWO cross-validation
jupyter notebook notebooks/GelmanPlumePINNPresentation.ipynb # Technical presentation
```

**Model weights** are not in the repo. They are stored on Google Drive at `/content/drive/MyDrive/gelman-pinn/`:
- `pinn_130_170_3input_final.pth` — Unit E (130–170 ft depth)
- `weights_50_90ft_v2.pth` — Unit C3 (50–90 ft depth)

## Architecture

### Neural Network (`DioxanePINN` class)
- **Inputs:** `(x, z, t)` — normalized distance along transect, depth, and time
- **Structure:** 4 hidden layers × 64 neurons, Tanh activations, sigmoid output → normalized concentration
- **Learnable physics parameters:** groundwater velocity `v` (ft/yr) and dispersion `D` (ft²/yr), bounded via sigmoid-constrained parameters

### Physics Constraint
The network is regularized by the 1D Advection-Dispersion Equation (ADE):
```
∂C/∂t + v·∂C/∂x - D·∂²C/∂x² - (D·0.1)·∂²C/∂z² = 0
```
Physics loss weight: **5.0×** relative to data loss.

### Two Aquifer Units
| Unit | Depth Band | Learned Velocity | Learned Dispersion |
|------|-----------|------------------|--------------------|
| E    | 130–170 ft | ~150 ft/yr       | ~2,500 ft²/yr     |
| C3   | 50–90 ft   | ~112.5 ft/yr     | ~1,450 ft²/yr     |

### Training Strategy
1. **Adam optimizer** (5,000–10,000 epochs) for global convergence
2. **L-BFGS refinement** for high-precision physics convergence
3. **10,000 collocation points** sampled randomly across the (x, z, t) domain to enforce ADE everywhere

### Data Normalization (critical — do not change)
```python
x_norm = along_distance_ft / 18178.0           # transect to Barton Pond
z_norm = (depth_ft - MIN_DEPTH) / DEPTH_RANGE
t_norm = years_since_1986 / 70.0               # 1986–2056 window
C_norm = log10(concentration + 1) / log10(212001)
```

## Data

- **File:** `data/processed/merged_df.pkl` (5.9 MB, pandas DataFrame)
- **Source:** 40 years of monitoring data from Washtenaw County & Michigan EGLE
- **Size:** ~26,682 samples after screen depth filtering; ~1,134 in Unit E
- **Depth filtering** is applied per notebook to select the relevant aquifer unit — this is a critical preprocessing step

## Key Constants
```python
X_MAX_DIST = 18178.0  # ft — source-to-Barton-Pond transect
T_MAX_YEARS = 70.0    # normalization window
T_START = 1986-01-01  # reference date
```

## Notebooks Summary
- **Gelman_PINN_Showcase.ipynb** — Loads pre-trained weights; produces 3D plume surface plots, vertical cross-sections, spatiotemporal animations, and geographic heatmaps. No training.
- **GelmanPINNLOWOValidation.ipynb** — Leave-One-Well-Out validation across 15 monitoring wells. Trains a fresh model per held-out well (~15 min/well).
- **GelmanPlumePINNPresentation.ipynb** — Full training pipeline plus regulatory analysis.
