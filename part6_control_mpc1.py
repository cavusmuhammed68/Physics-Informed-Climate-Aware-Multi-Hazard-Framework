# -*- coding: utf-8 -*-
"""
CIMH-DT — Part 6 (Advanced Control Layer, v3 — active storage)
---------------------------------------------------------------
Key changes to activate charging/discharging:
  • Stronger value-of-storage: terminal + rolling look-ahead buffer value
  • Lower effort weight for storage vs DR/V2G, separate w_ch and w_ds
  • Non-zero initial SOC (S0=0.20) to allow immediate discharge when risk high
  • Slightly earlier threshold (τ at 55th pct) → more frequent actions
  • Robust pressure & chance as in v2; anti-simultaneity kept

Outputs & inputs same as v2.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
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
df = pd.read_csv(panel_path)
if "MHI_kalman" not in df.columns:
    raise KeyError("Expected 'MHI_kalman' missing. Ensure Part 5 (adaptive) produced it.")

years = df["year"].astype(int).values
mhi   = df["MHI_kalman"].astype(float).values
sigma = df.get("MC_std", pd.Series(np.zeros_like(mhi))).astype(float).fillna(0.0).values

# light smoothing
mhi_smooth = mhi.copy()
if len(mhi) >= 3:
    mhi_smooth[1:-1] = 0.2*mhi[:-2] + 0.6*mhi[1:-1] + 0.2*mhi[2:]
mhi = mhi_smooth

# ----------------------- Pressure mapping ----------------------------
a, b, c = 0.80, 1.50, 0.30
tau = float(np.nanpercentile(mhi, 55))  # slightly earlier trigger

def pressure_and_sigma(m: np.ndarray, s: np.ndarray, tau: float) -> Tuple[np.ndarray,np.ndarray]:
    d = m - tau
    relu_pos = np.maximum(0.0, d)
    relu_neg = np.maximum(0.0, -d)
    p = a*relu_pos + b*(d**2) + c*relu_neg
    dp = a*(d>0).astype(float) + 2*b*d - c*(d<0).astype(float)
    sp = np.sqrt(np.maximum(1e-12, (dp**2) * (s**2)))
    return p, sp

base_p, base_sp = pressure_and_sigma(mhi, sigma, tau)

# -------------------------- RBC (unchanged from v2) -------------------
DR_MAX, V2G_MAX = 0.22, 0.18
SCH_MAX, SDS_MAX, S_MAX = 0.35, 0.30, 1.20
ETA_CH, ETA_DS = 0.92, 0.92

rbc_DR  = np.clip(0.35 * np.maximum(0.0, mhi - tau), 0, DR_MAX)
rbc_V2G = np.clip(0.30 * np.maximum(0.0, mhi - tau), 0, V2G_MAX)
rbc_CH  = np.clip(0.25 * np.maximum(0.0, tau - mhi), 0, SCH_MAX)
rbc_DS  = np.clip(0.22 * np.maximum(0.0, mhi - tau), 0, SDS_MAX)

S = np.zeros_like(mhi)
for t in range(len(mhi)-1):
    S[t+1] = np.clip(S[t] + ETA_CH*rbc_CH[t] - rbc_DS[t]/ETA_DS, 0, S_MAX)

curt_rbc = np.maximum(0.0, (base_p + 1.28*base_sp) - rbc_DR - rbc_V2G - rbc_DS + rbc_CH)

rbc_df = pd.DataFrame({
    "year": years, "MHI": mhi, "tau": tau, "pressure": base_p,
    "DR": rbc_DR, "V2G": rbc_V2G, "CH": rbc_CH, "DS": rbc_DS,
    "S_state": S, "curtail_proxy": curt_rbc
})

# -------------------------- MPC (v3 with active storage) --------------
from dataclasses import dataclass

@dataclass
class MPCParams:
    H: int = 6
    w_curt: float = 6.0     # stronger curtailment penalty
    w_dr: float = 1.0
    w_v2g: float = 0.9
    w_ch: float = 0.35      # cheaper to charge than DR/V2G
    w_ds: float = 0.45
    w_ramp: float = 0.15
    w_anti: float = 2.0
    w_soc: float = 0.2
    w_term: float = 3.0     # stronger terminal value
    w_buf: float = 1.2      # rolling buffer value for near-future risk
    z: float = 1.28
    DR_MAX: float = DR_MAX; V2G_MAX: float = V2G_MAX
    SCH_MAX: float = SCH_MAX; SDS_MAX: float = SDS_MAX
    S_MAX: float = S_MAX; ETA_CH: float = ETA_CH; ETA_DS: float = ETA_DS

mpc = MPCParams()

def line_search(x, grad, project, f, step=0.6):
    fx = f(x); t = step
    for _ in range(10):
        xn = project(x - t*grad)
        if f(xn) <= fx - 1e-4*t*np.dot(grad, grad):
            return xn, t
        t *= 0.5
    return project(x - t*grad), t

def solve_segment(m_seg, s_seg, S0, p: MPCParams):
    H = len(m_seg)
    x = np.zeros(5*H)
    def unpack(x):
        DR=x[0:H]; V2G=x[H:2*H]; CH=x[2*H:3*H]; DS=x[3*H:4*H]; CURT=x[4*H:5*H]
        return DR,V2G,CH,DS,CURT
    def project(x):
        DR,V2G,CH,DS,CURT = unpack(x)
        DR=np.clip(DR,0,p.DR_MAX); V2G=np.clip(V2G,0,p.V2G_MAX)
        CH=np.clip(CH,0,p.SCH_MAX); DS=np.clip(DS,0,p.SDS_MAX)
        CURT=np.clip(CURT,0,None)
        # storage feasibility
        S=np.zeros(H+1); S[0]=np.clip(S0,0,p.S_MAX)
        for t in range(H):
            S[t+1]=np.clip(S[t]+p.ETA_CH*CH[t]-DS[t]/p.ETA_DS,0,p.S_MAX)
            if S[t]+p.ETA_CH*CH[t]-DS[t]/p.ETA_DS<0:
                DS[t]=p.ETA_DS*(S[t]+p.ETA_CH*CH[t]); S[t+1]=0.0
        return np.concatenate([DR,V2G,CH,DS,CURT])
    base,bsp = pressure_and_sigma(m_seg, s_seg, tau)
    # precompute future positive risk for buffer value
    pos_future = np.zeros(H)
    pos_future[:-1] = np.maximum(0.0, m_seg[1:] - tau)
    pos_future[-1]  = pos_future[-2] if H>1 else 0.0
    def obj(x):
        DR,V2G,CH,DS,CURT = unpack(x)
        def ramp(z): d=np.diff(z,prepend=z[0]); return np.sum(d*d)
        S=np.zeros(H+1); S[0]=np.clip(S0,0,p.S_MAX)
        for t in range(H):
            S[t+1]=np.clip(S[t]+p.ETA_CH*CH[t]-DS[t]/p.ETA_DS,0,p.S_MAX)
        look = p.w_buf*np.sum(S[1:]*pos_future)    # rolling buffer value
        term = p.w_term*np.mean(np.maximum(0.0, m_seg-tau))*S[-1]
        chance = p.z*bsp
        need = np.maximum(0.0, base+chance - DR - V2G - DS + CH)
        return (p.w_curt*np.sum((CURT-need)**2)
                + p.w_dr*np.sum(DR**2) + p.w_v2g*np.sum(V2G**2)
                + p.w_ch*np.sum(CH**2) + p.w_ds*np.sum(DS**2)
                + p.w_ramp*(ramp(DR)+ramp(V2G)+ramp(CH)+ramp(DS))
                + p.w_anti*np.sum(CH*DS) + p.w_soc*np.sum(S**2)
                - look - term)   # subtract value-of-storage terms
    def grad(x):
        g=np.zeros_like(x); eps=1e-4; fx=obj(x)
        for i in range(len(x)):
            x2=x.copy(); x2[i]+=eps; g[i]=(obj(x2)-fx)/eps
        return g
    for _ in range(90):
        g=grad(x)
        x,_=line_search(x,g,project,obj,step=0.6)
    DR,V2G,CH,DS,CURT = unpack(x)
    S=np.zeros(H+1); S[0]=np.clip(S0,0,p.S_MAX)
    for t in range(H):
        S[t+1]=np.clip(S[t]+p.ETA_CH*CH[t]-DS[t]/p.ETA_DS,0,p.S_MAX)
    return DR,V2G,CH,DS,CURT,S[1]

# Rolling horizon with **non-zero initial SOC**
H = mpc.H; T=len(mhi)
rows=[]; S0=0.20
for t in range(T):
    seg_m=mhi[t:min(T,t+H)]; seg_s=sigma[t:min(T,t+H)]
    DR,V2G,CH,DS,CURT,S1=solve_segment(seg_m,seg_s,S0,mpc)
    base,bsp = pressure_and_sigma(np.array([mhi[t]]), np.array([sigma[t]]), tau)
    need = max(0.0, float(base + mpc.z*bsp - DR[0] - V2G[0] - DS[0] + CH[0]))
    rows.append({"year":int(years[t]),"MHI":mhi[t],"DR":DR[0],"V2G":V2G[0],
                 "CH":CH[0],"DS":DS[0],"curtail_proxy":need,
                 "S_state":float(np.clip(S0 + ETA_CH*CH[0] - DS[0]/ETA_DS,0,S_MAX)),
                 "tau":tau})
    S0=float(np.clip(S0 + ETA_CH*CH[0] - DS[0]/ETA_DS,0,S_MAX))

mpc_df=pd.DataFrame(rows)

# -------------------------- Save & Figures ----------------------------
rbc_out = ROOT / "CAMH_DT_Control_Summary.csv"
mpc_out = ROOT / "CAMH_DT_MPC_Trajectory.csv"
rbc_df.assign(strategy="RBC").to_csv(rbc_out, index=False)
mpc_df.assign(strategy="MPC").to_csv(mpc_out, index=False)

# Figures
figA, axA = plt.subplots(figsize=(9.6,6.0))
axA.plot(years, mhi, color="black", lw=2, label="MHI (Kalman)")
axA.axhline(tau, ls="--", color="gray", lw=1.0, label="Threshold τ")
axA.plot(years, rbc_df["curtail_proxy"], color="tab:blue", lw=1.8, label="RBC curtailment proxy")
axA.set_xlabel("Year"); axA.set_ylabel("Index / Proxy")
axA.set_title("Figure 6A — RBC (advanced)")
axA.legend(frameon=True); figA.tight_layout()
figA_path = FIGDIR / "Figure6A_RBC_Advanced_v3.png"
figA.savefig(figA_path); plt.close(figA)

figB, axB = plt.subplots(figsize=(10.2,6.2))
axB.plot(mpc_df["year"], mpc_df["DR"],  label="DR")
axB.plot(mpc_df["year"], mpc_df["V2G"], label="V2G")
axB.plot(mpc_df["year"], mpc_df["CH"],  label="Storage charge")
axB.plot(mpc_df["year"], mpc_df["DS"],  label="Storage discharge")
axB.set_xlabel("Year"); axB.set_ylabel("Control level")
axB.set_title("Figure 6B — MPC (v3): controls with active storage")
axB.legend(frameon=True); figB.tight_layout()
figB_path = FIGDIR / "Figure6B_MPC_v3_Controls.png"
figB.savefig(figB_path); plt.close(figB)

figC, axC = plt.subplots(figsize=(10.2,6.0))
axC.plot(mpc_df["year"], mpc_df["S_state"], lw=1.8, label="Storage state")
axC2 = axC.twinx()
axC2.plot(mpc_df["year"], mpc_df["curtail_proxy"], lw=1.8, color="tab:red", label="Curtailment proxy")
axC.set_xlabel("Year"); axC.set_ylabel("Storage state")
axC2.set_ylabel("Curtailment proxy")
axC.set_title("Figure 6C — MPC (v3): storage trajectory & curtailment proxy")
axC.legend(loc="upper left", frameon=True); axC2.legend(loc="upper right", frameon=True)
figC.tight_layout()
figC_path = FIGDIR / "Figure6C_MPC_v3_States.png"
figC.savefig(figC_path); plt.close(figC)

print("Saved:", rbc_out, "|", mpc_out)
print("Figures:", figA_path.name, figB_path.name, figC_path.name)
