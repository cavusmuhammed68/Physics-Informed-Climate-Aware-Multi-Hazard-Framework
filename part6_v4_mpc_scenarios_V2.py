# -*- coding: utf-8 -*-
"""
CIMH-DT — Part 6 (v4-stable, headroom-safe)
===============================================================================
Kalman-fused MHI → operational MPC controller.
SoC strictly confined to [0.15, 0.95] under all scenarios via stepwise headroom caps.
Matplotlib only. Serif typography + high DPI for publication.
===============================================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# -------------------------- Paths (Windows) --------------------------
FEATURE = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new\feature_store")
ROOT    = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new\src\results")
FIGDIR  = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new\figures")
for p in (FEATURE, ROOT, FIGDIR):
    p.mkdir(parents=True, exist_ok=True)

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
if "year" not in panel.columns or "MHI_kalman" not in panel.columns:
    raise KeyError("Expected columns ['year','MHI_kalman'] missing in the forecast panel.")

years = panel["year"].astype(int).to_numpy()
mhi   = panel["MHI_kalman"].astype(float).to_numpy()
sigma = (panel["MC_std"].astype(float).fillna(0.0).to_numpy()
         if "MC_std" in panel.columns else np.zeros_like(mhi, dtype=float))

# Mild causal smoothing (reduces chattering; no far-future leak)
if len(mhi) >= 3:
    mhi[1:-1] = 0.2*mhi[:-2] + 0.6*mhi[1:-1] + 0.2*mhi[2:]

# ------------------------ Risk → Pressure mapping ---------------------
A, B, C = 0.80, 1.50, 0.30          # affine–quadratic weights
TAU_PCTL = 60                        # default threshold percentile
tau = float(np.nanpercentile(mhi, TAU_PCTL))

def pressure_and_sigma(m: np.ndarray, s: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """Asymmetric affine–quadratic pressure and its uncertainty uplift (delta method)."""
    d = m - tau
    relu_pos = np.maximum(0.0, d)
    relu_neg = np.maximum(0.0, -d)
    p  = A*relu_pos + B*(d**2) + C*relu_neg
    dp = A*(d > 0).astype(float) + 2*B*d - C*(d < 0).astype(float)
    sp = np.sqrt(np.maximum(1e-12, (dp**2) * (s**2)))
    return p, sp

# ----------------------------- Limits --------------------------------
S_MIN, S_MAX = 0.15, 0.95
ETA_CH, ETA_DS = 0.92, 0.92
DR_MAX, V2G_MAX = 0.22, 0.18
SCH_MAX, SDS_MAX = 0.35, 0.30

def twinx(ax):
    ax2 = ax.twinx(); ax2.grid(False); return ax2

# =============================================================================
#  A) MPC scenario suite (2×2): four distinct behaviours of the SAME controller
# =============================================================================
@dataclass
class MPCParams:
    H: int = 6
    w_curt: float = 6.0
    w_dr: float = 1.0
    w_v2g: float = 0.9
    w_ch: float = 0.35
    w_ds: float = 0.45
    w_ramp: float = 0.15
    w_anti: float = 2.0
    w_term: float = 2.2
    w_buf: float = 1.0
    z: float = 1.28
    # limits/efficiencies
    DR_MAX: float = DR_MAX; V2G_MAX: float = V2G_MAX
    SCH_MAX: float = SCH_MAX; SDS_MAX: float = SDS_MAX
    S_MIN: float = S_MIN; S_MAX: float = S_MAX
    ETA_CH: float = ETA_CH; ETA_DS: float = ETA_DS

# ------------------------ Headroom helpers ----------------------------
def cap_actions_by_headroom(S_t: float, CH_t: float, DS_t: float, p: MPCParams) -> Tuple[float, float]:
    """
    Enforce per-step SoC feasibility via headroom-based caps:
      CH_t ≤ min(SCH_MAX, max(0,(S_MAX - S_t)/ETA_CH))
      DS_t ≤ min(SDS_MAX, max(0,(S_t - S_MIN)*ETA_DS))
    Returns (CH_t_capped, DS_t_capped).
    """
    ch_cap = min(p.SCH_MAX, max(0.0, (p.S_MAX - S_t) / p.ETA_CH))
    ds_cap = min(p.SDS_MAX, max(0.0, (S_t - p.S_MIN) * p.ETA_DS))
    return min(CH_t, ch_cap), min(DS_t, ds_cap)

def propagate_soc_sequence(CH: np.ndarray, DS: np.ndarray, S0: float, p: MPCParams) -> np.ndarray:
    """Deterministic SoC rollout using physical caps each step (guarantees bounds)."""
    H = len(CH)
    S = np.zeros(H+1, dtype=float)
    S[0] = float(np.clip(S0, p.S_MIN, p.S_MAX))
    for t in range(H):
        # apply the same caps here to mirror the feasible physics exactly
        ch_t, ds_t = cap_actions_by_headroom(S[t], CH[t], DS[t], p)
        S[t+1] = S[t] + p.ETA_CH*ch_t - ds_t/p.ETA_DS
        # small numerical safety
        if S[t+1] < p.S_MIN: S[t+1] = p.S_MIN
        if S[t+1] > p.S_MAX: S[t+1] = p.S_MAX
    return S

# ------------------------ MPC solver ---------------------------------
def solve_segment(m_seg: np.ndarray, s_seg: np.ndarray, S0: float, p: MPCParams):
    """
    Projected-gradient MPC with strong feasibility:
    - Every projection step caps CH/DS using SoC headroom for each time index.
    - Objective SoC propagation uses the same rule (consistent physics).
    """
    H = len(m_seg)
    x = np.zeros(5*H, dtype=float)

    def unpack(xv):
        DR  = xv[:H]
        V2G = xv[H:2*H]
        CH  = xv[2*H:3*H]
        DS  = xv[3*H:4*H]
        CURT= xv[4*H:5*H]
        return DR, V2G, CH, DS, CURT

    def project(xv: np.ndarray) -> np.ndarray:
        DR, V2G, CH, DS, CURT = unpack(xv)

        # Primary box constraints
        DR   = np.clip(DR,   0.0, p.DR_MAX)
        V2G  = np.clip(V2G,  0.0, p.V2G_MAX)
        CH   = np.clip(CH,   0.0, p.SCH_MAX)
        DS   = np.clip(DS,   0.0, p.SDS_MAX)
        CURT = np.clip(CURT, 0.0, None)

        # Headroom-based caps to guarantee SoC feasibility step-by-step
        S = np.zeros(H+1, dtype=float)
        S[0] = float(np.clip(S0, p.S_MIN, p.S_MAX))
        for t in range(H):
            CH[t], DS[t] = cap_actions_by_headroom(S[t], CH[t], DS[t], p)
            S[t+1] = S[t] + p.ETA_CH*CH[t] - DS[t]/p.ETA_DS
            # numerical clip (should be strictly inside by construction)
            S[t+1] = float(np.clip(S[t+1], p.S_MIN, p.S_MAX))

        return np.concatenate([DR, V2G, CH, DS, CURT])

    base, bsp = pressure_and_sigma(m_seg, s_seg, tau)
    pos_future = np.zeros(H, dtype=float)
    if H > 1:
        pos_future[:-1] = np.maximum(0.0, m_seg[1:] - tau)

    def obj(xv: np.ndarray) -> float:
        DR, V2G, CH, DS, CURT = unpack(xv)

        def ramp(z: np.ndarray) -> float:
            dz = np.diff(z, prepend=z[0]); return float(np.sum(dz*dz))

        # Use the *same* headroom physics to compute SoC for the cost
        S = propagate_soc_sequence(CH, DS, S0, p)

        term = p.w_term * float(np.mean(np.maximum(0.0, m_seg - tau))) * S[-1]
        look = p.w_buf * float(np.sum(S[1:] * pos_future))
        need = np.maximum(0.0, base + p.z*bsp - DR - V2G - DS + CH)

        return (p.w_curt * float(np.sum((CURT - need)**2))
                + p.w_dr  * float(np.sum(DR**2))
                + p.w_v2g * float(np.sum(V2G**2))
                + p.w_ch  * float(np.sum(CH**2))
                + p.w_ds  * float(np.sum(DS**2))
                + p.w_ramp * (ramp(DR) + ramp(V2G) + ramp(CH) + ramp(DS))
                + p.w_anti * float(np.sum(CH * DS))
                + 0.2 * float(np.sum(S**2))
                - term - look)

    def grad(xv: np.ndarray) -> np.ndarray:
        eps = 1e-4
        g = np.zeros_like(xv)
        fx = obj(xv)
        for i in range(len(xv)):
            x2 = xv.copy(); x2[i] += eps
            g[i] = (obj(x2) - fx) / eps
        return g

    # Projected gradient with backtracking
    for _ in range(60):
        g  = grad(x)
        fx = obj(x)
        t  = 0.5
        for _ in range(6):
            xn = project(x - t*g)
            if obj(xn) <= fx - 1e-4 * t * float(np.dot(g, g)):
                x = xn; break
            t *= 0.5

    DR, V2G, CH, DS, CURT = unpack(x)
    # Next-step SoC using the same headroom rule (hard guarantee)
    S_seq = propagate_soc_sequence(CH[:1], DS[:1], S0, p)
    S_next = float(S_seq[-1])
    return DR, V2G, CH, DS, CURT, S_next

# ------------------------ MPC rollout --------------------------------
def run_mpc(params: MPCParams, S0: float = 0.20) -> pd.DataFrame:
    H = params.H; T = len(mhi)
    rows = []; soc = float(np.clip(S0, params.S_MIN, params.S_MAX))

    for t in range(T):
        DR, V2G, CH, DS, CURT, soc_next = solve_segment(
            mhi[t:min(T, t+H)], sigma[t:min(T, t+H)], soc, params
        )
        # Defensive clip (should already be inside)
        soc_next = float(np.clip(soc_next, params.S_MIN, params.S_MAX))

        base, bsp = pressure_and_sigma(np.array([mhi[t]]), np.array([sigma[t]]), tau)
        need = max(0.0, float(base + params.z*bsp - DR[0] - V2G[0] - DS[0] + CH[0]))

        rows.append({
            "year": int(years[t]),
            "MHI": float(mhi[t]),
            "DR": float(DR[0]), "V2G": float(V2G[0]),
            "CH": float(CH[0]), "DS": float(DS[0]),
            "curtail_proxy": float(need),
            "S_state": soc_next, "tau": float(tau)
        })
        soc = soc_next

    df = pd.DataFrame(rows)
    # Final hard check
    if not df["S_state"].between(params.S_MIN - 1e-12, params.S_MAX + 1e-12).all():
        raise AssertionError("SoC bounds violated — this should be impossible with headroom caps.")
    return df

# -------------------------- Scenario definitions ---------------------
variants: Dict[str, MPCParams] = {
    "Base": MPCParams(),
    "High-risk": MPCParams(w_curt=9.0, w_term=2.6, z=1.5),
    "Storage-favouring": MPCParams(w_ch=0.25, w_ds=0.35, w_term=3.2, w_buf=1.8),
    "V2G-heavy": MPCParams(w_v2g=0.6, w_dr=1.2, w_curt=7.0),
}

# ------------------------------ Run ---------------------------------
results = {name: run_mpc(cfg, S0=0.20) for name, cfg in variants.items()}

# Save per-variant CSV
for name, dfv in results.items():
    out = ROOT / f"CAMH_DT_MPC_{name.replace(' ', '_')}.csv"
    dfv.assign(variant=name).to_csv(out, index=False)

# ------------------------------ Figure -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))
axs = axs.flatten()

for ax, (name, dfv) in zip(axs, results.items()):
    # Main curves
    ax.plot(dfv["year"], dfv["MHI"], color="black", lw=1.6, label="MHI")
    ax.plot(dfv["year"], dfv["DR"],  lw=1.3, label="DR")
    ax.plot(dfv["year"], dfv["V2G"], lw=1.3, label="V2G")
    ax.plot(dfv["year"], dfv["CH"],  lw=1.3, label="CH")
    ax.plot(dfv["year"], dfv["DS"],  lw=1.3, label="DS")

    # Secondary y-axis for SoC and Curtailment
    ax2 = twinx(ax)
    ax2.plot(dfv["year"], dfv["curtail_proxy"], lw=1.2,
             color="tab:blue", label="Curtailment")

    # Titles and labels
    ax.set_title(name, fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Level", fontsize=14)

    # Merge legends from both y-axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combined legend: 3 columns × 2 rows, fontsize 14
    ax2.legend(lines1 + lines2, labels1 + labels2,
               frameon=True, fontsize=14, loc="upper right",
               ncol=3, labelspacing=0.6, columnspacing=1.0,
               borderpad=0.7, handletextpad=0.8)

# Figure title and layout
fig.suptitle("Figure 6E-A — MPC behaviour across four scenarios (2×2)", fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = FIGDIR / "Figure6E_A_MPC_2x2.png"
fig.savefig(fig_path)
plt.close(fig)
print("Saved figure:", fig_path)









# ------------------------------ Figure -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))
axs = axs.flatten()

# Collect legend handles from all subplots
all_lines, all_labels = [], []

# ------------------------------ Figure -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))
axs = axs.flatten()

# Collect legend handles from all subplots
all_lines, all_labels = [], []

for ax, (name, dfv) in zip(axs, results.items()):
    # Main (left-axis) signals with consistent colour scheme
    l1, = ax.plot(dfv["year"], dfv["MHI"], color="black", lw=1.6, label="MHI")
    l2, = ax.plot(dfv["year"], dfv["DR"],  color="tab:green",  lw=1.3, label="DR")
    l3, = ax.plot(dfv["year"], dfv["V2G"], color="tab:purple", lw=1.3, label="V2G")
    l4, = ax.plot(dfv["year"], dfv["CH"],  color="tab:orange", lw=1.3, label="CH")
    l5, = ax.plot(dfv["year"], dfv["DS"],  color="tab:red",    lw=1.3, label="DISC")

    # Secondary (right-axis) signals
    ax2 = ax.twinx()
    l6, = ax2.plot(dfv["year"], dfv["S_state"], lw=1.2, linestyle="--",
                   color="tab:blue", label="SoC")
    l7, = ax2.plot(dfv["year"], dfv["curtail_proxy"], lw=1.2,
                   color="tab:blue", label="Curtailment")

    # Axis titles and labels
    ax.set_title(name, fontsize=16)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Level", fontsize=14)
    ax2.set_ylabel("SoC / Curtailment", fontsize=14, color="tab:blue")

    # Tick styling for clarity
    ax.tick_params(axis="y", labelsize=12, colors="black")
    ax2.tick_params(axis="y", labelsize=12, colors="tab:blue")

    # Save one set of handles for the shared legend
    if not all_labels:
        all_lines = [l1, l2, l3, l4, l5, l6, l7]
        all_labels = [ln.get_label() for ln in all_lines]

# Shared legend below the figure (2 rows × 3 columns)
fig.legend(all_lines, all_labels,
           loc="lower center",
           ncol=7, fontsize=14, frameon=True,
           columnspacing=1.2, handletextpad=0.8, labelspacing=0.6)

# Adjust layout to leave space for the legend
fig.suptitle("Figure 6E-A — MPC behaviour across four scenarios (2×2)", fontsize=18)
fig.tight_layout(rect=[0, 0.10, 1, 0.96])  # leave bottom space for legend

# Save and close
fig_path = FIGDIR / "Figure6E_A_MPC_2x2.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
print("Saved figure:", fig_path)


