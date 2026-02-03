# PINN-based Spatiotemporal Forecasting of 1,4-Dioxane Migration

**Physics-Informed Modeling of the Gelman Plume (Unit E & Unit C3 Aquifers)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgopal64/pinn-gelman-dioxane-modeling/blob/main/Gelman_PINN.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🗣️ Foreword

This project has a lot of personal meaning to me. When I first learned about it at UMich's CEE 365 class, I couldn't believe this environmental disaster isn't talked about more. 1,4-dioxane is a cancer-causing chemical, and Ann Arbor residents have had to close down their residential wells due to detections of it. I hope this model, on top of modeling its spread throughout my college town, will spread awareness of this issue.

---

## 📌 Project Overview

This project implements a **3-Input Physics-Informed Neural Network (PINN)** to model the migration of 1,4-Dioxane at the Gelman Sciences site in Ann Arbor, MI. By embedding the **Advection-Dispersion Equation (ADE)** directly into the loss function, the model bridges the "data deserts" between monitoring wells and the Huron River boundary.

Unlike traditional interpolation, this model ensures that predictions are **physically constrained** by fluid dynamics. It explicitly models two distinct hydrostratigraphic units:

| Unit | Depth Range | Description |
|------|-------------|-------------|
| **Unit E** | 130–170 ft | The deep, high-velocity basal aquifer |
| **Unit C3** | 50–90 ft | The shallower unit posing risks to residential basements |

---

## 🛠 Methodology

### 1. The 3-Input Architecture (x, z, t)

I upgraded from a standard 1D model to a **3-input Deep Neural Network** that accounts for vertical variations (z) within the aquifer. This allows the model to differentiate between the plume's "core" and its dispersed edges.

### 2. Physics Enforcement

The network is regularized by the 1D ADE with a vertical diffusion component, ensuring strictly physical behavior:

$$\frac{\partial C}{\partial t} + v \frac{\partial C}{\partial x} - D \frac{\partial^2 C}{\partial x^2} - (D \cdot 0.1)\frac{\partial^2 C}{\partial z^2} = 0$$

### 3. Inverse Physics Calibration

Instead of assuming hydraulic parameters, the PINN uses **Inverse Modeling** to "learn" the site-specific properties of the aquifer directly from 40 years of data:

| Parameter | Unit E | Unit C3 |
|-----------|--------|---------|
| **Velocity (v)** | ~150.2 ft/yr | ~112.5 ft/yr |
| **Transport** | High-speed | Slower migration |

---

## 📊 Key Findings & Forecasts

### 🚨 The 2080 Impact Horizon (Unit E)

For the deep Unit E aquifer, the model identifies **2080** as the critical year when the **7.2 ppb regulatory front** will reach the Barton Pond drinking water intake.

- **Safety Buffer:** The intake remains safe through the standard 2056 planning horizon.
- **Defensibility:** This finding is based on the calibrated velocity of 150.2 ft/yr, validated against 40 years of historical well data.

### 📉 Model Validation

The model achieves high fidelity to historical records, proving it isn't just "fitting curves" but learning the physics:

| Metric | Value |
|--------|-------|
| **Data R² Score** | 0.88 (Unit E) |
| **Physical RMSE** | ±0.79 ppb |

*High precision relative to the 7.2 ppb regulatory limit.*

---

## 💻 Interactive Dashboard

A **Streamlit dashboard** (`app.py`) is included to allow stakeholders (EGLE, City of Ann Arbor) to interactively visualize the plume's evolution.

### Features

- **Geographic Heatmap** — Overlays the plume on a live map of Ann Arbor to track the 7.2 ppb front
- **3D Topography** — Visualizes the plume's concentration gradient across depth (z) and distance (x)
- **Timeline Slider** — Move from 1986 (Discovery) to 2080 (Projected Impact)

---

## 📁 Repository Structure
```
├── app.py                          # Streamlit Interactive Dashboard
├── Gelman_PINN.ipynb               # Main Training & Analysis Notebook
├── merged_df.pkl                   # Processed Historical Monitoring Data
├── pinn_130_170_3input_final.pth   # Trained Weights: Unit E (Deep)
├── weights_50_90ft_v2.pth          # Trained Weights: Unit C3 (Shallow)
├── requirements.txt                # Dependencies (PyTorch, Streamlit, Folium)
└── README.md                       # Project Documentation
```

---

## 🏗 Setup & Usage

### Option 1: Run the Dashboard Locally

Visualize the 2080 forecasts on your own machine:
```bash
git clone https://github.com/mgopal64/pinn-gelman-dioxane-modeling.git
cd pinn-gelman-dioxane-modeling
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Train the Model (Colab)

Click the **"Open in Colab"** badge at the top of this README to access the full training pipeline, including the "Inverse Physics" calibration loop.

---

## 🤝 Data Attribution

This project is built upon **40 years of historical monitoring data** provided by:

- [Washtenaw County](https://www.washtenaw.org/)
- [Michigan Department of Environment, Great Lakes, and Energy (EGLE)](https://www.michigan.gov/egle)

Special recognition is given to the hydrogeologists and field technicians whose decades of field collection made this high-resolution modeling possible.

---

## 👤 About the Developer

**Manush Gopal**  
*Computer Science & Environmental Engineering*

I am focused on the intersection of physical sciences and machine learning, developing "physics-aware" AI to solve high-stakes environmental challenges.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/manush-gopal/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/mgopal64)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
