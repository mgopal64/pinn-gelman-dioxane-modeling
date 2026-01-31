# PINN-based Spatiotemporal Forecasting of 1,4-Dioxane Migration

**Modeling the Gelman Plume in the 130-170ft Aquifer unit**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mgopal64/pinn-gelman-dioxane-modeling/blob/main/GelmanPlumePINNPresentation.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

--- 

## Foreword

This project has a lot of personal meaning to me. When I first learned about it at UMich's CEE 365 class, I couldn't believe this environmental disaster isn't talked about more. 1,4-dioxane is a cancer-causing chemical, and Ann Arbor residents have had to close down their residential wells due to detections of it. I hope this model, on top of modeling its spread throughout my college town, will spread awareness of this issue. 

---

## 📌 Project Overview

This project implements a **Physics-Informed Neural Network (PINN)** to model the migration of 1,4-Dioxane at the Gelman Sciences site in Ann Arbor, MI. By embedding the Advection-Dispersion Equation (ADE) into a deep learning loss function, the model bridges "data deserts" between monitoring wells and the Huron River boundary.

Unlike traditional interpolation, this model ensures that the predicted plume movement is **physically constrained by fluid dynamics**, offering a high-fidelity forecast for long-term remediation planning.

---

## 🛠 Methodology


- **Hydrostratigraphic Isolation:** The analysis focuses on wells screened at 130-170 ft depth, corresponding to the **Intermediate Aquifer Zone** within the Unit E plume system—a high-conductivity glacial outwash unit that serves as a primary northeastward migration pathway toward the Huron River ([PLS Conceptual Site Model, 2014](https://www.michigan.gov/egle/-/media/Project/Websites/egle/Documents/Programs/RRD/Gelman/Selected-Documents/2014/pls-January-2014-Letter-Concerning-MW-103-Conceptual-Site-Model.pdf); [Loch-Caruso et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9835328/)).
  
- **Physics Enforcement:** The model is regularized using the 1D Advection-Dispersion equation, foundational to hydrology:

$$\frac{\partial C}{\partial t} + v \frac{\partial C}{\partial x} - D \frac{\partial^2 C}{\partial x^2} = 0$$

- **Optimization:** A dual-stage approach using Adam for initial convergence and L-BFGS for refining physical residuals and smoothing the concentration front.

- **Validation:** Spatial generalization was verified via Leave-One-Well-Out (LOWO) testing, achieving an R² of 0.88 and a physical precision of ±0.79 ppb.

---

## 📊 Key Results

### 2056 Plume Forecast
![2056 Forecast](forecast_2056.png)

The PINN forecast indicates that the **7.2 ppb regulatory threshold will not impact the Barton Pond/Huron River boundary** within the next century for **this depth unit**.

### Model Validation
![AE-1 Validation](validation_ae1.png)

Leave-One-Well-Out cross-validation confirms the model captures real contaminant transport dynamics, achieving **R² = 0.88** with physical precision of **±0.79 ppb**. Note that while not fitting the data 1-to-1, the trend line from the PINN model follows the underlying physics perfectly. This allows reliable prediction of the plume's edge.

- **100-Year Safety Horizon:** The PINN forecast indicates that the 7.2 ppb regulatory threshold will not impact the Barton Pond/Huron River boundary within the next century for this specific unit.

- **Learned Parameters:** The model autonomously calibrated site-specific velocity (*v*) and dispersion (*D*) coefficients that align with established hydrogeological findings.

---

## 📁 Repository Structure
```
├── Gelman_PINN.ipynb          # Main Jupyter/Colab notebook
├── merged_df.pkl              # Cleaned and processed site monitoring data
├── model_130_170ft_ploss5.pth # Pre-trained model weights for instant inference
├── requirements.txt           # Environment dependencies
└── README.md
```

---

## 🏗 Setup & Usage

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge above to run the notebook directly in your browser.

### Option 2: Local Installation
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
jupyter notebook Gelman_PINN.ipynb
```

### Running Modes

| Mode | Instructions |
|------|--------------|
| **Inference** | Default behavior—loads `model_130_170ft_ploss5.pth` and generates 2056 plume maps in seconds |
| **Training** | Set `RETRAIN_MODEL = True` in Section 3 to re-train or test a different depth band |

---

## 🤝 Data Attribution

This project is built upon 40 years of historical monitoring data provided by **Washtenaw County** and the **Michigan Department of Environment, Great Lakes, and Energy (EGLE)**. Special recognition is given to the hydrogeologists and field technicians whose decades of field collection made this high-resolution modeling possible.

---

## 👤 About the Developer

**Manush Gopal**  
Computer Science & Environmental Engineering

To learn more about my intersectional work:
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/manushgopal)

I am focused on the intersection of physical sciences and machine learning, developing "physics-aware" AI to solve high-stakes environmental challenges.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
