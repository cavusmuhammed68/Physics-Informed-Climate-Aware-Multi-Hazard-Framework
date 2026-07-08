# -*- coding: utf-8 -*-
"""
CAMH-DT — Part 4: Multi-Hazard Fusion, Physics-informed ML & Predictive Feedback Control
----------------------------------------------------------------------------------------
Integrates multiple hazard indicators (wind, flood, energy losses),
builds a unified Multi-Hazard Index (MHI), applies hybrid machine learning
with physical regularisation, and introduces predictive control and resilience metrics.

Author: <Your Name>
Affiliation: Northumbria University
Licence: MIT / CC-BY (as required by publisher)
"""

from __future__ import annotations
import os, sys, json, logging, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger():
    log = logging.getLogger("camh_dt.part4")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log

LOGGER = setup_logger()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a numeric 'year' column regardless of source schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["year"])
    df = df.copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        candidates = [c for c in df.columns if c.lower() in
                      ["yr","time","timestamp","datetime","date","year"]]
        if candidates:
            c = candidates[0]
            df["year"] = pd.to_datetime(df[c], errors="coerce").dt.year
        elif isinstance(df.index, pd.DatetimeIndex):
            df["year"] = df.index.year
        else:
            df["year"] = datetime.now().year
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    return df

def safe_read_parquet(fname, root):
    p = Path(root) / fname
    if not p.exists():
        LOGGER.warning(f"Missing {fname}")
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        csv = p.with_suffix(".csv")
        if csv.exists(): return pd.read_csv(csv)
        return pd.DataFrame()

def normalise(x):
    x = pd.Series(x, dtype=float)
    m, s = x.mean(), x.std(ddof=0)
    if s == 0 or np.isnan(s): return pd.Series(np.zeros_like(x))
    return (x - m) / s

def smooth(x, win=7, poly=2):
    try:
        return pd.Series(savgol_filter(x.fillna(method="ffill"), win, poly))
    except Exception:
        return x.rolling(win, min_periods=1, center=True).mean()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path("./feature_store")
FIGROOT = Path("./figures"); FIGROOT.mkdir(exist_ok=True)
RESULTS = Path("./results");  RESULTS.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Load datasets
# -----------------------------------------------------------------------------
LOGGER.info("Loading feature_store datasets ...")
datasets = {
    "wind": safe_read_parquet("c3s_wind_ind.parquet", ROOT),
    "risk": safe_read_parquet("ukc11_indicator_risk.parquet", ROOT),
    "loss": safe_read_parquet("ukc11_indicator_losses.parquet", ROOT),
    "curt": safe_read_parquet("curtailment_events.parquet", ROOT)
}

for k, df in datasets.items():
    if df.empty:
        LOGGER.warning(f"{k} dataset empty.")
        continue
    df = ensure_year_column(df)
    datasets[k] = df
    LOGGER.info(f"{k}: {len(df)} rows | columns={list(df.columns)[:6]}")

# -----------------------------------------------------------------------------
# Aggregate yearly means
# -----------------------------------------------------------------------------
def yearly(df):
    """
    Aggregates numeric indicators by year.
    Handles duplicate 'year' columns safely.
    """
    if df.empty:
        return pd.DataFrame(columns=["year", "index"])
    
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    num = df.select_dtypes(include="number").copy()
    
    # Eğer 'year' zaten numeric setin içinde varsa — yeniden ekleme
    if "year" not in num.columns:
        num["year"] = df["year"]
    
    # groupby sırasında 'year' zaten varsa, reset_index'te yeniden yaratma
    grouped = num.groupby("year", as_index=False).mean(numeric_only=True)
    
    # Varsayılan “index” sütununu belirle
    if "index" not in grouped.columns:
        # ikinci sütunu “index” olarak yeniden adlandır
        if len(grouped.columns) > 1:
            grouped = grouped.rename(columns={grouped.columns[1]: "index"})
        else:
            grouped["index"] = np.nan
    
    return grouped[["year", "index"]]


wind_y = yearly(datasets["wind"])
risk_y = yearly(datasets["risk"])
loss_y = yearly(datasets["loss"])

# -----------------------------------------------------------------------------
# Unified hazard panel
# -----------------------------------------------------------------------------
years = sorted(set(wind_y.year) | set(risk_y.year) | set(loss_y.year))
panel = pd.DataFrame({"year": years})
panel["W_hat"] = panel["year"].map(dict(zip(wind_y.year, wind_y["index"])))
panel["F_hat"] = panel["year"].map(dict(zip(risk_y.year, risk_y["index"])))
panel["E_hat"] = panel["year"].map(dict(zip(loss_y.year, loss_y["index"])))
panel[["W_hat","F_hat","E_hat"]] = panel[["W_hat","F_hat","E_hat"]].apply(normalise)
panel["MHI_raw"] = panel[["W_hat","F_hat","E_hat"]].mean(axis=1)
panel["MHI"] = smooth(panel["MHI_raw"], win=5)

panel["Hazard_Volatility"] = panel[["W_hat","F_hat","E_hat"]].std(axis=1)
LOGGER.info(f"Unified hazard panel built for {len(panel)} years.")

# -----------------------------------------------------------------------------
# Physics-informed + ML models
# -----------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

# Remove rows that are entirely empty
train = panel.dropna(subset=["W_hat","F_hat","E_hat","MHI"], how="all")

X = train[["W_hat","F_hat","E_hat"]].values
y = train["MHI"].values

# Fill missing values (mean imputation)
imp = SimpleImputer(strategy="mean")
X_filled = imp.fit_transform(X)

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Train models
ridge = RidgeCV(alphas=np.logspace(-4,4,80)).fit(X_scaled, y)
rf = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42).fit(X_scaled, y)
gb = GradientBoostingRegressor(n_estimators=250, random_state=42).fit(X_scaled, y)

# Predictions
ridge_pred = ridge.predict(X_scaled)
rf_pred = rf.predict(X_scaled)
gb_pred = gb.predict(X_scaled)

# Scores
r2_ridge = r2_score(y, ridge_pred)
r2_rf = r2_score(y, rf_pred)
r2_gb = r2_score(y, gb_pred)

LOGGER.info(f"Model fits — Ridge R²={r2_ridge:.3f}, RF R²={r2_rf:.3f}, GB R²={r2_gb:.3f}")


# -----------------------------------------------------------------------------
# Predictive control & feedback
# -----------------------------------------------------------------------------
panel["y_pred"] = ridge.predict(scaler.transform(panel[["W_hat","F_hat","E_hat"]].fillna(0)))
panel["error"] = panel["MHI"] - panel["y_pred"]

panel["control_signal"] = -0.25 * panel["error"].diff().fillna(0)
panel["MHI_controlled"] = smooth(panel["MHI"] + panel["control_signal"], win=5)

horizon = 3
panel["MPC_forecast"] = panel["y_pred"].rolling(window=horizon, min_periods=1).mean()
panel["MPC_adjusted"] = smooth(0.7*panel["MHI_controlled"] + 0.3*panel["MPC_forecast"])

panel["Resilience_Score"] = 1 - np.abs(panel["error"])
panel["Uncertainty"] = panel[["W_hat","F_hat","E_hat"]].std(axis=1)

# -----------------------------------------------------------------------------
# PCA (latent hazard modes)
# -----------------------------------------------------------------------------
if len(panel) > 5:
    pca = PCA(n_components=2)
    pc = pca.fit_transform(panel[["W_hat","F_hat","E_hat"]].fillna(0))
    panel["PC1"], panel["PC2"] = pc[:,0], pc[:,1]
    expl = pca.explained_variance_ratio_
    LOGGER.info(f"PCA explained variance: PC1={expl[0]*100:.1f}%, PC2={expl[1]*100:.1f}%")
else:
    panel["PC1"]=panel["PC2"]=np.nan
    LOGGER.warning("Too few records for PCA.")

# -----------------------------------------------------------------------------
# Correlation matrix
# -----------------------------------------------------------------------------
corr = panel[["W_hat","F_hat","E_hat","MHI","Resilience_Score"]].corr()
corr.to_csv(RESULTS/"correlation_matrix.csv")

# -----------------------------------------------------------------------------
# Figures (8 subplots, 600 dpi)
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi":600,"font.size":13})
fig,ax=plt.subplots(4,2,figsize=(10,12))

# (a)
ax[0,0].plot(panel.year,panel.W_hat,label="Wind")
ax[0,0].plot(panel.year,panel.F_hat,label="Flood")
ax[0,0].plot(panel.year,panel.E_hat,label="Energy")
ax[0,0].set_title("(a) Normalised Hazard Indices")
ax[0,0].set_xlabel("Year"); ax[0,0].set_ylabel("Index"); ax[0,0].legend()

# (b)
ax[0,1].plot(panel.year,panel.MHI,'k-',label="Raw MHI")
ax[0,1].plot(panel.year,panel.MHI_controlled,'r--',label="Controlled")
ax[0,1].plot(panel.year,panel.MPC_adjusted,'g-',alpha=0.7,label="MPC Adjusted")
ax[0,1].set_title("(b) Multi-Hazard Index Dynamics")
ax[0,1].legend(); ax[0,1].set_xlabel("Year")

# (c) Model predictions vs actual
ax[1,0].scatter(y, ridge_pred, s=40, alpha=0.7, label=f"Ridge R²={r2_ridge:.2f}")
ax[1,0].scatter(y, rf_pred, s=40, alpha=0.6, label=f"RF R²={r2_rf:.2f}")
ax[1,0].scatter(y, gb_pred, s=40, alpha=0.6, label=f"GB R²={r2_gb:.2f}")
ax[1,0].plot([min(y), max(y)], [min(y), max(y)], 'k--')
ax[1,0].set_title("(c) Model Fit Comparison")
ax[1,0].set_xlabel("Actual MHI")
ax[1,0].set_ylabel("Predicted MHI")
ax[1,0].legend()

# (d)
imp = rf.feature_importances_
ax[1,1].bar(["Wind","Flood","Energy"],imp)
ax[1,1].set_title("(d) Random-Forest Feature Importance")

# (e)
if panel.PC1.notna().any():
    sc=ax[2,0].scatter(panel.PC1,panel.PC2,c=panel.MHI,cmap="viridis",s=40)
    fig.colorbar(sc,ax=ax[2,0],label="MHI")
ax[2,0].set_title("(e) PCA Latent Hazard Space")
ax[2,0].set_xlabel("PC1"); ax[2,0].set_ylabel("PC2")

# (f)
ax[2,1].plot(panel.year,panel.control_signal,'m-',lw=1.2)
ax[2,1].set_title("(f) Control Signal Evolution")
ax[2,1].set_xlabel("Year"); ax[2,1].set_ylabel("Signal")

# (g)
ax[3,0].plot(panel.year,panel.Resilience_Score,'b-',lw=1.3)
ax[3,0].set_title("(g) Resilience Score Over Time")
ax[3,0].set_xlabel("Year"); ax[3,0].set_ylabel("Resilience (1-|error|)")

# (h)
im=ax[3,1].imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)
ax[3,1].set_xticks(range(len(corr.columns)))
ax[3,1].set_xticklabels(corr.columns,rotation=45)
ax[3,1].set_yticks(range(len(corr.columns)))
ax[3,1].set_yticklabels(corr.columns)
ax[3,1].set_title("(h) Correlation Heatmap")
fig.colorbar(im,ax=ax[3,1],shrink=0.8)

plt.tight_layout()
ts=datetime.now().strftime("%Y%m%d_%H%M")
fig.savefig(FIGROOT/f"CAMH_DT_Part4_Results_{ts}.png",dpi=600)
LOGGER.info("Figures saved successfully.")

# -----------------------------------------------------------------------------
# Export final datasets
# -----------------------------------------------------------------------------
panel.to_csv(RESULTS/"CAMH_DT_Fused_Panel_Advanced.csv",index=False)
LOGGER.info("✅ Part 4 (Advanced Edition) complete — all results exported.")



# ----------------------------------------------------------------------
# Figure (4 panels, 2×2, high-resolution, British academic style)
# ----------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 13
})

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# (a) Multi-Hazard Index Dynamics
ax[0,0].plot(panel.year, panel.MHI, 'k-', lw=1.6, label="MHI (Raw)")
ax[0,0].plot(panel.year, panel.MHI_controlled, 'r--', lw=1.6, label="Controlled MHI")
ax[0,0].plot(panel.year, panel.MPC_adjusted, 'g-', lw=1.6, alpha=0.75, label="MPC-Adjusted MHI")
ax[0,0].set_title("(a) Multi-Hazard Index Dynamics")
ax[0,0].set_xlabel("Year")
ax[0,0].set_ylabel("Index (normalised)")
ax[0,0].legend(frameon=True)

# (b) Model–Fit Comparison
ax[0,1].scatter(y, ridge_pred, s=40, alpha=0.7, label=f"Ridge (R² = {r2_ridge:.2f})")
ax[0,1].scatter(y, rf_pred, s=40, alpha=0.6, label=f"RF (R² = {r2_rf:.2f})")
ax[0,1].scatter(y, gb_pred, s=40, alpha=0.6, label=f"GB (R² = {r2_gb:.2f})")
ax[0,1].plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=1.0)
ax[0,1].set_title("(b) Model–Fit Comparison")
ax[0,1].set_xlabel("Actual MHI")
ax[0,1].set_ylabel("Predicted MHI")
ax[0,1].legend(frameon=True)

# (c) Control Signal Evolution
ax[1,0].plot(panel.year, panel.control_signal, 'm-', lw=1.4)
ax[1,0].set_title("(c) Control Signal Evolution")
ax[1,0].set_xlabel("Year")
ax[1,0].set_ylabel("Control Signal (dimensionless)")

# (d) Resilience Score Over Time
ax[1,1].plot(panel.year, panel.Resilience_Score, 'b-', lw=1.4)
ax[1,1].set_title("(d) Resilience Score Over Time")
ax[1,1].set_xlabel("Year")
ax[1,1].set_ylabel("Resilience (1 − |error|)")

fig.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M")
fig.savefig(FIGROOT / f"CAMH_DT_Part4_Results_Reduced_{ts}.png", dpi=600)
LOGGER.info("✅ Reduced 2×2 figure saved successfully.")



plt.rcParams.update({
    "figure.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16
})

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# (a) Multi-Hazard Index Dynamics
ax[0,0].plot(panel.year, panel.MHI, 'k-', lw=1.6, label="MHI (Raw)")
ax[0,0].plot(panel.year, panel.MHI_controlled, 'r--', lw=1.6, label="Controlled MHI")
ax[0,0].plot(panel.year, panel.MPC_adjusted, 'g-', lw=1.6, alpha=0.75, label="MPC-Adjusted MHI")
ax[0,0].set_title("(a)", fontweight="bold")
ax[0,0].set_xlabel("Year")
ax[0,0].set_ylabel("Index (normalised)")
ax[0,0].legend(frameon=True)

# (b) Model–Fit Comparison
ax[0,1].scatter(y, ridge_pred, s=40, alpha=0.7, label=f"Ridge (R² = {r2_ridge:.2f})")
ax[0,1].scatter(y, rf_pred, s=40, alpha=0.6, label=f"RF (R² = {r2_rf:.2f})")
ax[0,1].scatter(y, gb_pred, s=40, alpha=0.6, label=f"GB (R² = {r2_gb:.2f})")
ax[0,1].plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=1.0)
ax[0,1].set_title("(b)", fontweight="bold")
ax[0,1].set_xlabel("Actual MHI")
ax[0,1].set_ylabel("Predicted MHI")
ax[0,1].legend(frameon=True)

# (c) Control Signal Evolution
ax[1,0].plot(panel.year, panel.control_signal, 'm-', lw=1.4)
ax[1,0].set_title("(c)", fontweight="bold")
ax[1,0].set_xlabel("Year")
ax[1,0].set_ylabel("Control Signal (dimensionless)")

# (d) Resilience Score Over Time
ax[1,1].plot(panel.year, panel.Resilience_Score, 'b-', lw=1.4)
ax[1,1].set_title("(d)", fontweight="bold")
ax[1,1].set_xlabel("Year")
ax[1,1].set_ylabel("Resilience (1 − |error|)")

fig.tight_layout()
ts = datetime.now().strftime("%Y%m%d_%H%M")
fig.savefig(FIGROOT / f"CAMH_DT_Part4_Results_Reduced_{ts}.png", dpi=600)
LOGGER.info("✅ Reduced 2×2 figure saved successfully.")


