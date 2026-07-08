# -*- coding: utf-8 -*-
"""
CIMH-DT — Part 6 (v4, Q1-ready)
===============================================================================
This module turns Kalman-fused MHI from Part 5 into **operational controls**.
It is written as a *stand‑alone research artifact* with rich comments & plotting.
Matplotlib only (no seaborn). Serif typography + high DPI for publication.

COMMON ASSUMPTIONS
------------------
• You already ran Part 5 and produced: ROOT/CAMH_DT_Forecast_Panel.csv
  (must include: year, MHI_kalman, MC_std).

• Storage state-of-charge (SoC) must respect **hard bounds**:
      S_MIN = 0.15   ≤  S_t  ≤   S_MAX = 0.95
  Charging/Discharging efficiencies are ETA_CH and ETA_DS.

• Baseline *risk pressure* is mapped from MHI via an asymmetric
  affine–quadratic function  p(m) = a·(m-τ)^+ + b·(m-τ)^2 + c·(τ-m)^+.
  Uncertainty uplift uses the delta method: Var[p] ≈ (∂p/∂m)^2 Var[m].
===============================================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# -------------------------- Paths (Windows) --------------------------
FEATURE = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Upcoming Papers\TSG - 1 Grid Failure (Submitted)\UK datasets\Results_new\feature_store")
ROOT    = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Upcoming Papers\TSG - 1 Grid Failure (Submitted)\UK datasets\Results_new\results")
FIGDIR  = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Upcoming Papers\TSG - 1 Grid Failure (Submitted)\UK datasets\Results_new\figures")
for p in (FEATURE, ROOT, FIGDIR): p.mkdir(parents=True, exist_ok=True)


# -------------------------- Plot styling -----------------------------
plt.style.use("default")
mpl.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": True, "grid.alpha": 0.28,
    "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

# -------------------------- Load forecasts ----------------------------
panel_path = ROOT / "CAMH_DT_Forecast_Panel.csv"
if not panel_path.exists():
    raise FileNotFoundError("Run Part 5 first — forecast panel missing.")
panel = pd.read_csv(panel_path)
if "MHI_kalman" not in panel.columns:
    raise KeyError("Expected 'MHI_kalman' missing in forecast panel produced by Part 5.")

years = panel["year"].astype(int).values
mhi   = panel["MHI_kalman"].astype(float).values
sigma = panel.get("MC_std", pd.Series(np.zeros_like(mhi))).astype(float).fillna(0.0).values

# Mild causal smoothing (reduces chattering; does not leak far future)
if len(mhi) >= 3:
    mhi_s = mhi.copy()
    mhi_s[1:-1] = 0.2*mhi[:-2] + 0.6*mhi[1:-1] + 0.2*mhi[2:]
    mhi = mhi_s

# ------------------------ Risk → Pressure mapping ---------------------
A, B, C = 0.80, 1.50, 0.30          # affine–quadratic weights
TAU_PCTL = 60                        # default threshold percentile
tau = float(np.nanpercentile(mhi, TAU_PCTL))

def pressure_and_sigma(m: np.ndarray, s: np.ndarray, tau: float) -> Tuple[np.ndarray,np.ndarray]:
    """Asymmetric affine–quadratic pressure and its uncertainty uplift (delta method)."""
    d = m - tau
    relu_pos = np.maximum(0.0, d)
    relu_neg = np.maximum(0.0, -d)
    p  = A*relu_pos + B*(d**2) + C*relu_neg
    dp = A*(d>0).astype(float) + 2*B*d - C*(d<0).astype(float)
    sp = np.sqrt(np.maximum(1e-12, (dp**2) * (s**2)))
    return p, sp

# SOC limits (strict)
S_MIN, S_MAX = 0.15, 0.95
ETA_CH, ETA_DS = 0.92, 0.92

# Device limits
DR_MAX, V2G_MAX = 0.22, 0.18
SCH_MAX, SDS_MAX = 0.35, 0.30

# Convenience plotting helpers
def twinx(ax):
    ax2 = ax.twinx()
    ax2.grid(False)
    return ax2

# =====================================================================
#  C) Storage strategies (2×2): same MPC, four storage policy variants
# =====================================================================
@dataclass
class MPCParams:
    H: int = 6
    w_curt: float = 6.5; w_dr: float = 1.0; w_v2g: float = 0.9
    w_ch: float = 0.35; w_ds: float = 0.45; w_ramp: float = 0.15; w_anti: float = 2.0
    w_soc: float = 0.2; w_term: float = 2.4; w_buf: float = 1.2; z: float = 1.28
    DR_MAX: float = DR_MAX; V2G_MAX: float = V2G_MAX; SCH_MAX: float = SCH_MAX; SDS_MAX: float = SDS_MAX
    S_MIN: float = S_MIN; S_MAX: float = S_MAX; ETA_CH: float = ETA_CH; ETA_DS: float = ETA_DS

def solve_segment(m_seg, s_seg, S0, p: MPCParams):
    H=len(m_seg); x=np.zeros(5*H)
    def unpack(x):
        DR=x[:H]; V2G=x[H:2*H]; CH=x[2*H:3*H]; DS=x[3*H:4*H]; CURT=x[4*H:5*H]
        return DR,V2G,CH,DS,CURT
    def project(x):
        DR,V2G,CH,DS,CURT=unpack(x)
        DR=np.clip(DR,0,p.DR_MAX); V2G=np.clip(V2G,0,p.V2G_MAX)
        CH=np.clip(CH,0,p.SCH_MAX); DS=np.clip(DS,0,p.SDS_MAX); CURT=np.clip(CURT,0,None)
        S=np.zeros(H+1); S[0]=np.clip(S0,p.S_MIN,p.S_MAX)
        for t in range(H):
            S[t+1]=np.clip(S[t]+p.ETA_CH*CH[t]-DS[t]/p.ETA_DS,p.S_MIN,p.S_MAX)
            if S[t] + p.ETA_CH*CH[t] - DS[t]/p.ETA_DS < p.S_MIN:
                DS[t]=p.ETA_DS*(S[t]+p.ETA_CH*CH[t]-p.S_MIN); DS[t]=max(0.0,DS[t]); S[t+1]=p.S_MIN
            if S[t] + p.ETA_CH*CH[t] - DS[t]/p.ETA_DS > p.S_MAX:
                CH[t]=(p.S_MAX-S[t]+DS[t]/p.ETA_DS)/p.ETA_CH; CH[t]=max(0.0,min(CH[t],p.SCH_MAX)); S[t+1]=p.S_MAX
        return np.concatenate([DR,V2G,CH,DS,CURT])
    base,bsp=pressure_and_sigma(m_seg,s_seg,tau); pos=np.zeros(H); pos[:-1]=np.maximum(0.0,m_seg[1:]-tau)
    def obj(x):
        DR,V2G,CH,DS,CURT=unpack(x)
        def ramp(z): d=np.diff(z,prepend=z[0]); return np.sum(d*d)
        S=np.zeros(H+1); S[0]=np.clip(S0,p.S_MIN,p.S_MAX)
        for t in range(H): S[t+1]=np.clip(S[t]+p.ETA_CH*CH[t]-DS[t]/p.ETA_DS,p.S_MIN,p.S_MAX)
        need=np.maximum(0.0, base + p.z*bsp - DR - V2G - DS + CH)
        return (p.w_curt*np.sum((CURT-need)**2)+p.w_dr*np.sum(DR**2)+p.w_v2g*np.sum(V2G**2)+
                p.w_ch*np.sum(CH**2)+p.w_ds*np.sum(DS**2)+p.w_ramp*(ramp(DR)+ramp(V2G)+ramp(CH)+ramp(DS))+
                p.w_anti*np.sum(CH*DS)+p.w_soc*np.sum(S**2) - p.w_term*np.mean(np.maximum(0.0,m_seg-tau))*S[-1] - p.w_buf*np.sum(S[1:]*pos))
    def grad(x):
        g=np.zeros_like(x); eps=1e-4; fx=obj(x)
        for i in range(len(x)): x2=x.copy(); x2[i]+=eps; g[i]=(obj(x2)-fx)/eps
        return g
    for _ in range(80):
        g=grad(x); t=0.6; fx=obj(x)
        for _ in range(8):
            xn=project(x - t*g)
            if obj(xn)<=fx-1e-4*t*np.dot(g,g): x=xn; break
            t*=0.5
    DR,V2G,CH,DS,CURT=unpack(x); S=np.clip(S0 + p.ETA_CH*CH[0]-DS[0]/p.ETA_DS,p.S_MIN,p.S_MAX)
    return DR,V2G,CH,DS,CURT,S

def run_mpc(p: MPCParams, S0=0.20) -> pd.DataFrame:
    H=p.H; T=len(mhi); rows=[]; soc=S0
    for t in range(T):
        DR,V2G,CH,DS,CURT,soc_next=solve_segment(mhi[t:min(T,t+H)],sigma[t:min(T,t+H)],soc,p)
        base,bsp=pressure_and_sigma(np.array([mhi[t]]),np.array([sigma[t]]),tau)
        need=max(0.0,float(base + p.z*bsp - DR[0]-V2G[0]-DS[0]+CH[0]))
        rows.append({"year":int(years[t]),"MHI":mhi[t],"DR":DR[0],"V2G":V2G[0],"CH":CH[0],"DS":DS[0],"S_state":float(soc_next),"curtail_proxy":need})
        soc=float(soc_next)
    return pd.DataFrame(rows)

# Four storage strategies
strategies = {
    "Tight bounds": (S_MIN, S_MAX),
    "Loose bounds": (0.20, 0.90),
    "High charge eff": (S_MIN, S_MAX),
    "High discharge eff": (S_MIN, S_MAX),
}
outputs: Dict[str,pd.DataFrame] = {}
for name,(smin,smax) in strategies.items():
    p = MPCParams()
    if name=="Loose bounds":
        p.S_MIN, p.S_MAX = smin, smax
    if name=="High charge eff":
        p.ETA_CH = 0.97
    if name=="High discharge eff":
        p.ETA_DS = 0.97
    outputs[name] = run_mpc(p, S0=0.20)

# 2×2 figure (Figure 6E-C)
# --------------------------- Figure 6E-C ------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))
axs = axs.flatten()

# Collect legend handles and labels (for one shared legend below)
handles_for_legend, labels_for_legend = [], []

for ax, (name, dfv) in zip(axs, outputs.items()):
    # Left-axis: MHI, CH, DS (consistent colours)
    lMHI, = ax.plot(dfv["year"], dfv["MHI"], color="black", lw=1.6, label="MHI")
    lCH,  = ax.plot(dfv["year"], dfv["CH"],  color="tab:orange", lw=1.3, label="CH")
    lDS,  = ax.plot(dfv["year"], dfv["DS"],  color="tab:red",    lw=1.3, label="DISC")

    # Right-axis: SoC (orange dashed) + Curtailment (blue solid)
    ax2 = twinx(ax)
    lSOC,  = ax2.plot(dfv["year"], dfv["S_state"], lw=1.2, linestyle="--",
                      color="tab:blue", label="SoC")
    lCURT, = ax2.plot(dfv["year"], dfv["curtail_proxy"], lw=1.2,
                      color="tab:blue", label="Curtailment")

    # Titles, labels, axis formatting
    ax.set_title(name, fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Level", fontsize=14)
    ax2.set_ylabel("SoC / Curtailment", fontsize=14, color="tab:blue")

    ax.tick_params(axis="y", labelsize=12, colors="black")
    ax2.tick_params(axis="y", labelsize=12, colors="tab:blue")

    # Collect legend entries once
    if not labels_for_legend:
        handles_for_legend = [lMHI, lCH, lDS, lSOC, lCURT]
        labels_for_legend  = [h.get_label() for h in handles_for_legend]

# Shared legend below all subplots (2 rows × 3 columns)
fig.legend(handles_for_legend, labels_for_legend,
           loc="lower center", ncol=5, fontsize=14, frameon=True,
           columnspacing=1.2, handletextpad=0.8, labelspacing=0.6)

# Figure title and spacing
fig.suptitle("Figure 6E-C — Storage strategy variants (2×2)", fontsize=18)
fig.tight_layout(rect=[0, 0.10, 1, 0.96])   # space at bottom for legend

# Save and close
pth = FIGDIR / "Figure6E_C_Storage_2x2.png"
fig.savefig(pth, bbox_inches="tight")
plt.close(fig)
print("Saved figure:", pth)

