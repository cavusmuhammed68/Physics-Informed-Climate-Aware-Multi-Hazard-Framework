# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:15:39 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
CIMH-DT — Integrated Part 7 + Part 8 (v5, Q1-ready)
================================================================================
Loads prior outputs (Parts 5–6), evaluates RBC, MPC and proposed CAMH-DT,
runs scenario-based robustness, and produces a unified 2×2 figure:
  • (a)–(c): Three scenarios (Mild-Stable, Gradual-Rise, Escalation)
             plotting curtailment proxy for RBC (red), MPC (blue), CAMH-DT (green, thicker)
  • (d): Summary improvements (CAMH-DT vs MPC & RBC)

Design: Matplotlib only. Serif (Times New Roman). 600 DPI. One shared legend.
Strict SoC bounds via headroom caps. Colours consistent with earlier figures.
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
from pathlib import Path
from scipy.special import erf

# -------------------------------- Paths -------------------------------
FEATURE = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new\feature_store")
ROOT    = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new\src\results")
FIGDIR  = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new\figures")
for p in (FEATURE, ROOT, FIGDIR): p.mkdir(parents=True, exist_ok=True)

# -------------------------- Plot styling -----------------------------
plt.style.use("default")
mpl.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True, "grid.alpha": 0.28,
    "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

# -------------------------- Load baseline ----------------------------
fcast_path   = ROOT / "CAMH_DT_Forecast_Panel.csv"
truth_path   = ROOT / "CAMH_DT_Fused_Panel_Advanced.csv"
if not fcast_path.exists():  raise FileNotFoundError("Run Part 5 first — forecast panel missing.")
if not truth_path.exists():  raise FileNotFoundError("Run Part 4 first — fused panel with ground truth missing.")

F = pd.read_csv(fcast_path).copy()
truth = pd.read_csv(truth_path)[["year","MHI"]]
F = F.merge(truth, on="year", how="left")
if F["MHI"].isna().any():
    raise RuntimeError("Ground truth could not be aligned with forecast years.")

# Prefer Kalman / calibrated forecast for baseline mean
for c in ["MHI_kalman","MHI_forecast_cal","MHI_forecast","MHI_raw"]:
    if c in F.columns:
        base_mu = F[c].astype(float).to_numpy(); chosen = c; break
else:
    raise KeyError("No forecast column found among expected options.")

yrs = F["year"].astype(int).to_numpy()
N   = len(yrs)
mc_sd = F.get("MC_std", pd.Series(np.full(N, np.nan))).astype(float).fillna(0.0).to_numpy()
mc_sd = np.maximum(mc_sd, 1e-9)
print(f"Using forecast column: {chosen}")

# ------------------------ Risk → Pressure mapping --------------------
A, B, C = 0.80, 1.50, 0.30
TAU_PCTL = 60
tau = float(np.nanpercentile(base_mu, TAU_PCTL))

def pressure_and_sigma(m: np.ndarray, s: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    d = m - tau
    p_pos = np.maximum(0.0, d)
    p_neg = np.maximum(0.0, -d)
    p  = A*p_pos + B*(d**2) + C*p_neg
    dp = A*(d>0).astype(float) + 2*B*d - C*(d<0).astype(float)
    sp = np.sqrt(np.maximum(1e-12, (dp**2)*(s**2)))
    return p, sp

# --------------------------- Scenario set ----------------------------
def smooth(x, k=5): return np.convolve(x, np.ones(k)/k, mode="same")

S1 = smooth(base_mu + 0.03*np.sin(np.linspace(0,4*np.pi,N)))     # Mild-Stable
S2 = smooth(base_mu + np.linspace(0, 0.8*np.std(base_mu), N))     # Gradual-Rise
S3 = smooth(base_mu + np.concatenate([np.zeros(int(0.55*N)),
                                      np.linspace(0,2.0*np.std(base_mu),
                                                   N-int(0.55*N))]))  # Escalation

SCEN = {"Mild-Stable": S1, "Gradual-Rise": S2, "Escalation": S3}

# --------------------------- Physical limits -------------------------
S_MIN, S_MAX = 0.15, 0.95
ETA_CH, ETA_DS = 0.92, 0.92
DR_MAX, V2G_MAX, CH_MAX, DS_MAX = 0.22, 0.18, 0.35, 0.30

def cap_actions_by_headroom(S_t: float, CH_t: float, DS_t: float) -> Tuple[float, float]:
    ch_cap = min(CH_MAX, max(0.0, (S_MAX - S_t)/ETA_CH))
    ds_cap = min(DS_MAX, max(0.0, (S_t - S_MIN)*ETA_DS))
    return min(CH_t, ch_cap), min(DS_t, ds_cap)

def propagate_soc_seq(CH: np.ndarray, DS: np.ndarray, S0: float) -> np.ndarray:
    H = len(CH); S = np.zeros(H+1, dtype=float)
    S[0] = float(np.clip(S0, S_MIN, S_MAX))
    for t in range(H):
        ch_t, ds_t = cap_actions_by_headroom(S[t], float(CH[t]), float(DS[t]))
        S[t+1] = S[t] + ETA_CH*ch_t - ds_t/ETA_DS
        S[t+1] = float(np.clip(S[t+1], S_MIN, S_MAX))
    return S

# ---------------------------- Controllers ----------------------------
def curtailment_proxy(m, s, DR, V2G, CH, DS, z=1.28):
    base, sp = pressure_and_sigma(m, s, tau)
    need = np.maximum(0.0, base + z*sp - DR - V2G - DS + CH)
    return need

def run_rbc(mhi: np.ndarray, sdev: np.ndarray, S0: float=0.25) -> pd.DataFrame:
    med = np.median(mhi)
    DR = np.clip(0.25*np.maximum(0, mhi - med), 0, DR_MAX)
    V2G= np.clip(0.20*np.maximum(0, mhi - med), 0, V2G_MAX)
    CH = np.clip(0.28*np.maximum(0, med - mhi), 0, CH_MAX)
    DS = np.clip(0.24*np.maximum(0, mhi - med), 0, DS_MAX)

    S = np.zeros_like(mhi); S[0] = float(np.clip(S0, S_MIN, S_MAX))
    CHc = np.zeros_like(CH); DSc = np.zeros_like(DS)
    for t in range(len(mhi)-1):
        CHc[t], DSc[t] = cap_actions_by_headroom(S[t], CH[t], DS[t])
        S[t+1] = S[t] + ETA_CH*CHc[t] - DSc[t]/ETA_DS
        S[t+1] = float(np.clip(S[t+1], S_MIN, S_MAX))
    CHc[-1], DSc[-1] = cap_actions_by_headroom(S[-1], CH[-1], DS[-1])

    curt = curtailment_proxy(mhi, sdev, DR, V2G, CHc, DSc, z=1.28)
    return pd.DataFrame({"year": yrs, "MHI_sim": mhi, "DR": DR, "V2G": V2G, "CH": CHc, "DS": DSc,
                         "S_state": S, "curtail_proxy": curt})

@dataclass
class MPCParams:
    H:int=6; w_curt:float=7.0; w_dr:float=1.0; w_v2g:float=0.9
    w_ch:float=0.30; w_ds:float=0.40; w_ramp:float=0.15; w_anti:float=2.0
    w_term:float=2.6; w_buf:float=1.4; z:float=1.28

def run_mpc(mhi: np.ndarray, sdev: np.ndarray, S0: float=0.25, p:MPCParams=MPCParams()) -> pd.DataFrame:
    H=p.H; S=float(np.clip(S0, S_MIN, S_MAX))
    rows=[]
    for t in range(N):
        seg = mhi[t:min(N, t+H)]
        press = np.maximum(0, seg - np.median(seg))
        weight = np.linspace(1.0, 1.6, len(seg))
        DR = np.clip(0.29*press*weight, 0, DR_MAX)
        V2G= np.clip(0.22*press*weight, 0, V2G_MAX)
        CH = np.clip(0.31*np.maximum(0, np.median(seg)-seg), 0, CH_MAX)
        DS = np.clip(0.25*press*weight, 0, DS_MAX)
        # prevent simultaneous charge/discharge
        if (DR[0]+V2G[0]+DS[0]) > CH[0]: CH[0] = 0.0
        S_next = float(np.clip(S + ETA_CH*CH[0] - DS[0]/ETA_DS, S_MIN, S_MAX))
        curt = curtailment_proxy(np.array([mhi[t]]), np.array([sdev[t]]), DR[:1], V2G[:1], CH[:1], DS[:1], z=p.z)[0]
        rows.append({"year": int(yrs[t]), "MHI_sim": float(mhi[t]),
                     "DR": float(DR[0]), "V2G": float(V2G[0]), "CH": float(CH[0]), "DS": float(DS[0]),
                     "S_state": S_next, "curtail_proxy": float(curt)})
        S = S_next
    return pd.DataFrame(rows)

# Proposed CAMH-DT controller (risk-aware + storage-favouring; guaranteed-SoC)
@dataclass
class CAMHParams:
    H:int=6; z:float=1.50
    w_term:float=3.2; w_buf:float=1.8
    alpha_dr:float=0.31; alpha_v2g:float=0.24; alpha_ds:float=0.27; alpha_ch:float=0.30

def run_camh_dt(mhi: np.ndarray, sdev: np.ndarray, S0: float=0.25, p:CAMHParams=CAMHParams()) -> pd.DataFrame:
    H=p.H; S=float(np.clip(S0, S_MIN, S_MAX))
    rows=[]
    for t in range(N):
        seg = mhi[t:min(N, t+H)]
        base, sp = pressure_and_sigma(seg, sdev[t:min(N, t+H)], tau)
        risk = np.maximum(0.0, base + p.z*sp)  # risk-uplifted pressure
        look = np.zeros_like(seg); look[:-1] = np.maximum(0.0, seg[1:]-tau)
        valS = p.w_term*np.mean(np.maximum(0.0, seg - tau)) + p.w_buf*np.mean(look)
        # slightly more aggressive discharging on high risk; storage favouring otherwise
        DR  = np.clip(p.alpha_dr  * risk, 0, DR_MAX)
        V2G = np.clip(p.alpha_v2g * risk, 0, V2G_MAX)
        DS  = np.clip(p.alpha_ds  * risk, 0, DS_MAX)
        CH  = np.clip(p.alpha_ch  * np.maximum(0.0, tau - seg), 0, CH_MAX)

        # never CH & DS together at step 0; favour CH if value-of-storage is high and SoC < mid
        if (DR[0]+V2G[0]+DS[0]) > CH[0] and (S < 0.55 or valS < 0):
            CH[0] = 0.0
        # SoC headroom caps
        CH[0], DS[0] = cap_actions_by_headroom(S, CH[0], DS[0])
        S_next = float(np.clip(S + ETA_CH*CH[0] - DS[0]/ETA_DS, S_MIN, S_MAX))
        curt = curtailment_proxy(np.array([mhi[t]]), np.array([sdev[t]]), DR[:1], V2G[:1], CH[:1], DS[:1], z=p.z)[0]
        rows.append({"year": int(yrs[t]), "MHI_sim": float(mhi[t]),
                     "DR": float(DR[0]), "V2G": float(V2G[0]), "CH": float(CH[0]), "DS": float(DS[0]),
                     "S_state": S_next, "curtail_proxy": float(curt)})
        S = S_next
    return pd.DataFrame(rows)

# --------------------------- Evaluate & Run ---------------------------
def eval_metrics(df: pd.DataFrame) -> Dict[str,float]:
    # Lower is better for curtailment; lower intensity & SoC volatility preferred.
    curtail = float(np.maximum(0.0, df["curtail_proxy"]).mean())
    intensity = float(df[["DR","V2G","CH","DS"]].abs().mean().mean())
    soc_vol = float(df["S_state"].std())
    return {"Curtail": curtail, "Intensity": intensity, "SoC_vol": soc_vol}

RESULTS = {}
for scen_name, series in SCEN.items():
    # approximate per-year stdev for this scenario (fallback to mc_sd)
    sdev = mc_sd if np.any(mc_sd > 0) else np.full_like(series, np.std(series - np.mean(series)))
    rbc = run_rbc(series, sdev, S0=0.25)
    mpc = run_mpc(series, sdev, S0=0.25)
    cam = run_camh_dt(series, sdev, S0=0.25)
    RESULTS[scen_name] = {"RBC": rbc, "MPC": mpc, "CAMH-DT": cam}

# ----------------------- Save per-scenario CSVs -----------------------
for scen_name, d in RESULTS.items():
    for k, df in d.items():
        out = ROOT / f"CAMH_DT_Integrated_{k}_{scen_name.replace(' ', '_')}.csv"
        df.to_csv(out, index=False)

# --------------------------- Figure 9 (2×2) --------------------------
fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.2))
axs = axs.flatten()

method_colors = {"RBC": "tab:red", "MPC": "tab:blue", "CAMH-DT": "tab:green"}
method_lw = {"RBC": 1.6, "MPC": 1.6, "CAMH-DT": 2.2}  # CAMH-DT highlighted
scen_list = list(RESULTS.keys())

# (a)–(c): scenario panels — plot Curtailment proxy (lower is better)
for i, scen_name in enumerate(scen_list[:3]):
    ax = axs[i]
    for method in ["RBC", "MPC", "CAMH-DT"]:
        df = RESULTS[scen_name][method]
        ax.plot(df["year"], df["curtail_proxy"],
                color=method_colors[method],
                lw=method_lw[method],
                label=method)
    ax.set_title(f"({chr(ord('a') + i)}) {scen_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Curtailment Proxy")
    ax.grid(True, alpha=0.28)
    #  Legend inside subplot (upper-right corner)
    ax.legend(frameon=True, fontsize=11, loc="upper right")

# (d): Summary improvements — bars of average curtailment (lower is better)
axd = axs[3]
width = 0.24
x = np.arange(len(scen_list[:3]))
avg_curt = {
    m: [RESULTS[s][m]["curtail_proxy"].mean() for s in scen_list[:3]]
    for m in ["RBC", "MPC", "CAMH-DT"]
}

# Plot grouped bars
axd.bar(x - width, avg_curt["RBC"], width, color=method_colors["RBC"], label="RBC")
axd.bar(x, avg_curt["MPC"], width, color=method_colors["MPC"], label="MPC")
axd.bar(x + width, avg_curt["CAMH-DT"], width, color=method_colors["CAMH-DT"], label="CAMH-DT")

axd.set_xticks(x)
axd.set_xticklabels(scen_list[:3])
axd.set_ylabel("Mean Curtailment Proxy")
axd.set_title("(d) Summary — Lower is Better")
axd.grid(True, alpha=0.28)

#  Legend inside the bar chart panel — single row (1×3)
axd.legend(frameon=True, fontsize=11, loc="upper left",
           ncol=3, columnspacing=1.0, handletextpad=0.6, labelspacing=0.4)

# ----------------------- Layout and Save -----------------------
fig.suptitle("Figure 9 — Integrated Method Comparison (RBC, MPC, CAMH-DT) — 2×2 Panel",
             fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.97])

fig9_path = FIGDIR / "Figure9_Integrated_CIMHDT_Comparison.png"
fig.savefig(fig9_path, bbox_inches="tight")
plt.close(fig)
print("Figure saved:", fig9_path)

# -------------------- Policy Impact Table -------------------
summary_rows = []
for s in scen_list[:3]:
    for m in ["RBC", "MPC", "CAMH-DT"]:
        v = float(RESULTS[s][m]["curtail_proxy"].mean())
        summary_rows.append({"Scenario": s, "Method": m, "MeanCurtail": v})

summary_df = pd.DataFrame(summary_rows)
summary_out = ROOT / "CAMH_DT_Integrated_Summary_Curtailment.csv"
summary_df.to_csv(summary_out, index=False)
print("Saved:", summary_out)
print("\n Integrated Part 7 + Part 8 completed successfully.")
