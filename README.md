# CAMH: Climate-Aware Multi-Hazard Framework for Grid-Failure Risk and Adaptive Operational Control

## Overview

The increasing penetration of renewable energy exposes modern power systems to compound climate hazards and heightened operational uncertainty. **CAMH** is a simulation-based, physics-informed decision-support framework that unifies **hazard perception**, **risk estimation**, and **operational control** in a single closed-loop architecture, rather than treating them as separate problems.

CAMH addresses three technical gaps in existing approaches:

- the temporal mismatch between **annual** hazard indices and **hourly** operational control;
- the absence of a **physics-informed, calibration-free** multi-hazard fusion method (i.e., one that does not require labelled fault/outage records);
- the lack of a **closed-loop** coupling between climate hazard forecasts and Model Predictive Control (MPC) objectives.

## Framework Architecture

CAMH is organised into three sequentially coupled layers:

| Layer | Function | Key methods |
|---|---|---|
| **Layer 1 — Hazard Perception** | Constructs wind, flood, and infrastructure-exposure hazard sub-indices from ERA5 reanalysis and grid operational data | Kinetic-energy wind proxy, Gaussian KDE storm-track density, physics-informed spatial/temporal regularisation |
| **Layer 2 — Multi-Hazard Fusion & Downscaling** | Fuses hazard dimensions into a single Multi-Hazard Index (MHI) and disaggregates it from annual to hourly resolution | Constrained Non-Negative Least Squares (NNLS), seasonal-amplitude temporal downscaling |
| **Layer 3 — Risk Estimation & Resilience-Aware Control** | Forecasts hazard evolution, maps it to generation degradation and a dynamic failure-risk indicator, and embeds this into MPC | Adaptive Kalman filter ensemble, Poisson failure-process risk indicator $P_{\mathrm{fail}}(t)$, resilience-aware MPC with ramp-rate/capacity constraints |

Two feedback paths keep the architecture closed-loop: an **operational feedback path** (control actions update the operating state feeding the next risk estimate) and a **control feedback path** (dispatch decisions feed back into the Kalman hazard estimator).

CAMH is a **decision-support / supervisory-control framework**: it produces advisory hazard, risk, and dispatch signals for use by system operators (TSO/DNO/aggregators), rather than a topology-resolved real-time dispatch controller with direct authority over grid assets.

## Key Results

Using historical UK power system and ERA5 reanalysis data (1979-2024), CAMH:

- reduces the **mean modelled grid-failure risk indicator** by ≈ **62%** (57-67% under parameter sensitivity) relative to a rule-based controller;
- lowers **peak residual operational stress** by **14-18%**;
- improves **operational flexibility utilisation** by **8-11%**;
- bounds control degradation under estimation error: a 10% hazard estimation error induces at most a **4.2%** increase in peak residual stress;
- the adaptive Kalman ensemble achieves **R² = 0.963** in hazard forecasting, outperforming AR, Gaussian Process, and deep learning (LSTM/GRU/TCN/Transformer) baselines evaluated on the same 46-point annual series.

> **Note on interpretation:** $P_{\mathrm{fail}}(t)$ is a **model-based resilience indicator** reflecting hazard-weighted operational stress. It is **not** an empirically calibrated outage probability, and results should be interpreted as simulation-based findings rather than real-world operational validation.

## Repository Structure

```
.
├── data/                 # ERA5, flood, and grid exposure indicators (or download scripts)
├── src/
│   ├── layer1_hazard/    # Wind/flood/exposure indicators, KDE, physics regularisation
│   ├── layer2_fusion/    # NNLS fusion, temporal downscaling
│   ├── layer3_control/   # Adaptive Kalman filter, resilience-aware MPC
│   └── utils/            # Shared utilities and metrics
├── notebooks/            # Reproducibility notebooks for figures and tables
├── results/              # Generated figures, tables, and simulation outputs
├── configs/              # Parameter configuration files (Table 1 of the paper)
└── README.md
```

## Data Sources

- **ERA5 reanalysis / C3S Atmosphere Data Store** — windstorm indicators, storm-track density: <https://cds.climate.copernicus.eu>
- **Northern Powergrid Open Data Portal** — renewable penetration, curtailment, feeder interruption data: <https://northernpowergrid.opendatasoft.com/pages/home>


## Contact

For questions about the code or paper, please open an issue or contact the corresponding author at muhammed.cavus@northumbria.ac.uk.
