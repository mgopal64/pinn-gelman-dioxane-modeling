# PINN-based Spatiotemporal Forecasting of 1,4-Dioxane Migration

**Physics-Informed Modeling of the Gelman Plume (Unit E & Unit C3 Aquifers)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Foreword

This project has a lot of personal meaning to me. When I first learned about it at UMich's CEE 365 class, I couldn't believe this environmental disaster isn't talked about more. 1,4-dioxane is a cancer-causing chemical, and Ann Arbor residents have had to close down their residential wells due to detections of it. I hope this model, on top of modeling its spread throughout my college town, will spread awareness of this issue.

---

## Project Overview

This project implements a **3-Input Physics-Informed Neural Network (PINN)** to model the migration of 1,4-Dioxane at the Gelman Sciences site in Ann Arbor, MI. By embedding the **Advection-Dispersion Equation (ADE)** directly into the loss function, the model bridges the "data deserts" between monitoring wells and the Huron River boundary.

Unlike traditional interpolation, the model ensures predictions are **physically constrained** by fluid dynamics. It explicitly models two distinct hydrostratigraphic units:

| Unit | Depth Range | Description |
|------|-------------|-------------|
| **Unit E** | 130–170 ft | Deep, high-velocity basal aquifer |
| **Unit C3** | 50–90 ft | Shallower unit posing risks to residential basements |

---

## Methodology

### 1. The 3-Input Architecture (x, z, t)

A **3-input Deep Neural Network** accounts for vertical variations (z) within the aquifer, allowing the model to differentiate between the plume's "core" and dispersed edges.

### 2. Physics Enforcement

The network is regularized by the 1D ADE with a vertical diffusion component and a horizontal leakage term, ensuring strictly physical behavior:

$$\frac{\partial C}{\partial t} + v \frac{\partial C}{\partial x} - D \frac{\partial^2 C}{\partial x^2} - (D \cdot 0.1)\frac{\partial^2 C}{\partial z^2} + \lambda C = 0$$

The $\lambda C$ term represents **lateral (y-direction) mass loss** — contaminant leaving the 1D transect laterally — which the 1D formulation cannot capture explicitly.

### 3. Inverse Physics Calibration

Instead of assuming hydraulic parameters, the PINN uses **Inverse Modeling** to learn site-specific aquifer properties from 40 years of data. All three parameters are bounded via sigmoid constraints configured in `configs/config.yaml`.

---

## Validation Metrics

Evaluated on full training data using pretrained weights:

| Metric | Unit E (130–170 ft) | Unit C3 (50–90 ft) |
|--------|---------------------|--------------------|
| Wells / Samples | 15 / 1134 | 2 / 100 |
| **R² (linear)** | **0.9006** | **0.7499** |
| **R² (log-space)** | **0.9220** | **0.7657** |
| RMSE | 154.47 ppb | 233.56 ppb |
| MAE | 73.44 ppb | 159.48 ppb |
| Learned v | 168.8 ft/yr | 151.5 ft/yr |
| Learned D | 2039.9 ft²/yr | 1067.0 ft²/yr |

**Leave-One-Well-Out (LOWO) — Unit E, MW-85 (138 samples, 7450 ft downgradient):**
- R² (linear): 0.7277 | R² (log): 0.8212 | RMSE: 301.06 ppb

*High precision relative to the 7.2 ppb regulatory limit.*

---

## Key Findings & Forecasts

### The 2080 Impact Horizon (Unit E)

For the deep Unit E aquifer, the model identifies **2080** as the critical year when the **7.2 ppb regulatory front** will reach the Barton Pond drinking water intake.

- **Safety Buffer:** The intake remains safe through the standard 2056 planning horizon.
- **Defensibility:** Based on the calibrated velocity of ~169 ft/yr, validated against 40 years of historical well data.

---

## Repository Structure

```
├── configs/
│   └── config.yaml              # All hyperparameters, unit bounds, normalization
├── data/
│   └── processed/
│       ├── merged_df.parquet    # Monitoring data — primary format (use this)
│       └── merged_df.csv.gz     # Fallback if pyarrow unavailable
├── models/                      # Pre-trained weights (not tracked by git)
│   ├── pinn_130_170_3input_final.pth
│   └── weights_50_90ft_v2.pth
├── notebooks/                   # Original Colab development notebooks (reference)
├── scripts/
│   ├── strip_arcgis.py          # One-time converter: legacy .pkl → parquet/csv.gz
│   └── eval_pretrained.py       # Quick evaluation script
├── src/
│   ├── model.py                 # DioxanePINN architecture + load/save helpers
│   ├── physics.py               # ADE residual loss, collocation sampling
│   ├── data_loader.py           # Loading, filtering, normalization
│   ├── train.py                 # Training pipeline (Adam + L-BFGS), CLI
│   ├── evaluate.py              # Leave-One-Well-Out validation, CLI
│   └── visualize.py             # Forecast and visualization plots, CLI
└── requirements.txt
```

---

## Data

The monitoring dataset (`merged_df.parquet`) is included in the repository. It contains 28,944 samples from 1986–2026 across both aquifer units. **Use the parquet file directly — no additional setup needed.**

```
data/process/merged_df.parquet   ← data_loader.py picks this up automatically
```

A `merged_df.csv.gz` is also provided as a zero-dependency fallback. The loader checks for parquet first, then csv.gz, automatically.

<details>
<summary>If you only have the original <code>merged_df.pkl</code> (legacy)</summary>

The original pickle was serialized in an ArcGIS environment and cannot be opened without the `arcgis` package. Run the conversion script once to produce the parquet and csv.gz files:

```bash
python scripts/strip_arcgis.py
```

This works without arcgis installed — it stubs the two arcgis types internally, drops the unused `SHAPE` geometry column, and writes both output formats.
</details>

---

## Setup & Usage

### Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run inference on pre-trained weights

```bash
# Forecast curves for Unit E (130–170 ft)
python src/visualize.py --unit unit_e --plot forecast

# All plots for a specific year
python src/visualize.py --unit unit_e --plot all --year 2056

# Interactive geographic heatmap (requires folium)
python src/visualize.py --unit unit_e --plot heatmap --year 2080
```

### Train from scratch

```bash
# Full training: Adam (10k epochs) + L-BFGS refinement
python src/train.py --unit unit_e

# Adam only, faster (e.g. for experimentation)
python src/train.py --unit unit_c3 --adam-epochs 5000 --no-lbfgs
```

### Run Leave-One-Well-Out validation

```bash
# Full LOWO across all wells in a unit
python src/evaluate.py --unit unit_e --output figures/lowo_unit_e.csv

# Quick pretrained evaluation + single-well LOWO sanity check
python scripts/eval_pretrained.py
```

---

## Data Attribution

This project is built upon **40 years of historical monitoring data** provided by:

- [Washtenaw County](https://www.washtenaw.org/)
- [Michigan Department of Environment, Great Lakes, and Energy (EGLE)](https://www.michigan.gov/egle)

Special recognition is given to the hydrogeologists and field technicians whose decades of field collection made this high-resolution modeling possible.

---

## About the Developer

**Manush Gopal**
*Computer Science & Environmental Engineering*

I am focused on the intersection of physical sciences and machine learning, developing "physics-aware" AI to solve high-stakes environmental challenges.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/manush-gopal/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/mgopal64)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
