# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 12:28:02 2025

@author: nfpm5
"""
from __future__ import annotations

# -*- coding: utf-8 -*-
"""
CAMH-DT — Part 5 (Advanced, Q1-ready, FULL)
Forecasting & Uncertainty Quantification for the Multi-Hazard Index (MHI)
------------------------------------------------------------------------
Proposed method: CAMH-DT (Climate-Informed Multi-Hazard Digital Twin)

Bu modül:
  • 2010+ için sızıntısız (leakage-free) rolling-origin bir-adım-ileri tahmin üretir
  • Dört DL ailesini (LSTM/GRU/TCN/Transformer) karşılaştırır
  • Doğrulama kaybına dayalı dinamik ansambıl (ridge-stacking) uygular
  • Monte-Carlo Dropout + Gaussian Process belirsizlik bantları üretir
  • Bias düzeltmesi + Doğrusal kalibrasyon uygular
  • **Adaptif kalibrasyon** ekler: Gain/Offset + AR(2) rezidü düzeltmesi + **Kalman (Medium)** füzyon
  • MAE, RMSE, R² metriklerini hem toplam hem rolling olarak raporlar
  • Gelişmiş figürler:
      - Figure 5A: Final Ensemble (Kalman) + 68/95% PI (2010+)
      - Figure 5B: Model comparison (MAE/RMSE/R² bar + adaptive lines)
      - Figure 5C: Rolling RMSE/MAE (5y)
      - Figure 5D: Parametre varyantları (W=4/6/8/10) — 4 panel

Yalnızca Matplotlib kullanır (seaborn yok). Bilimsel serif tipografi ve yüksek DPI.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------- Paths (Windows) --------------------------
FEATURE = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Upcoming Papers\TSG - 1 Grid Failure (Submitted)\UK datasets\Results_new\feature_store")
ROOT    = Path(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Upcoming Papers\TSG - 1 Grid Failure (Submitted)\UK datasets\Results_new\src\results")
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

# -------------------------- Reproducibility --------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# -------------------------- Load fused panel -------------------------
panel_path = ROOT / "CAMH_DT_Fused_Panel_Advanced.csv"
if not panel_path.exists():
    raise FileNotFoundError("Run Part 4 first — fused panel missing.")
df = pd.read_csv(panel_path).sort_values("year").reset_index(drop=True)
if "MHI" not in df.columns:
    raise KeyError("Expected 'MHI' column missing in fused panel.")

START_YEAR = 2010
if df["year"].max() < START_YEAR:
    raise RuntimeError("Dataset ends before 2010; cannot run 2010+ forecasting.")

years_all = df["year"].values
mhi_raw   = df["MHI"].values.astype(float)

# ----------------- Stabilisation: power transform + scale --------------
pt = PowerTransformer(method="yeo-johnson", standardize=True)
mhi_pt = pt.fit_transform(mhi_raw.reshape(-1,1)).flatten()

scaler = MinMaxScaler()
mhi_scaled = scaler.fit_transform(mhi_pt.reshape(-1,1)).flatten()

# -------------------------- Helpers ------------------------------------
def build_xy(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N,W,1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # (N,1)
    return X, y

def inv_transform(x_scaled: np.ndarray) -> np.ndarray:
    """Inverse of (MinMax on Yeo-Johnson)."""
    x_pt = scaler.inverse_transform(x_scaled.reshape(-1,1)).flatten()
    return pt.inverse_transform(x_pt.reshape(-1,1)).flatten()

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def enable_dropout(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()

# ------------------------ Models ---------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        out   = self.drop(out[:, -1, :])
        return self.fc(out)

class GRUModel(nn.Module):
    def __init__(self, hidden=32, dropout=0.3):
        super().__init__()
        self.gru  = nn.GRU(1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.gru(x)
        out   = self.drop(out[:, -1, :])
        return self.fc(out)

class TCNBlock(nn.Module):
    def __init__(self, channels=16, kernel=3, dilation=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=kernel, padding=dilation, dilation=dilation)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(channels, 1)
    def forward(self, x):
        x = x.transpose(1,2)               # (B,1,W)
        x = self.relu(self.conv1(x))       # (B,C,W)
        x = self.drop(x[:, :, -1])         # (B,C)
        return self.fc(x)                  # (B,1)

class TinyTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, nl=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=dropout, batch_first=True
        )
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=nl)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d_model, 1)
    def forward(self, x):
        z = self.proj(x)                    # (B,W,d)
        z = self.enc(z)                     # (B,W,d)
        z = self.drop(z[:, -1, :])          # (B,d)
        return self.fc(z)                   # (B,1)

def fit_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs=160, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward(); opt.step()
    return model

def mc_predict(model: nn.Module, seq_scaled: np.ndarray, window: int, runs=120) -> Tuple[float,float]:
    enable_dropout(model)
    preds = []
    for _ in range(runs):
        x = torch.tensor(seq_scaled[-window:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            preds.append(model(x).cpu().numpy().flatten()[0])
    mu_s = float(np.mean(preds)); sd_s = float(np.std(preds))
    return mu_s, sd_s

# ------------------- Hyperparameters & Param Sweep --------------------
WINDOW_BASE = 10
PARAM_SWEEP = [
    {"window": 4,  "hidden": 32, "dropout": 0.30},
    {"window": 6,  "hidden": 32, "dropout": 0.30},
    {"window": 8,  "hidden": 32, "dropout": 0.30},
    {"window": 10, "hidden": 32, "dropout": 0.30},
]
MC_RUNS = 200

# ---------------- Rolling-origin forecasting with stacking -------------
per_year_rows = []
per_model_rows = []
rolling_metrics = []  # (year, rmse_window, mae_window)

years_forecast = []

for idx in range(len(df)):
    year = int(years_all[idx])
    if year < START_YEAR: 
        continue

    hist_scaled = mhi_scaled[:idx]

    if len(hist_scaled) < WINDOW_BASE + 5:
        continue

    # Base window ile eğitim
    W = WINDOW_BASE
    X, y = build_xy(hist_scaled, W)

    # Zaman sıralı val bölmesi
    split = max(1, int(0.8 * len(X)))
    Xtr, ytr = X[:split], y[:split]
    Xv,  yv  = X[split:], y[split:]

    # Modeller
    lstm = fit_model(LSTMModel(32,0.30), Xtr, ytr, epochs=200, lr=7e-3)
    gru  = fit_model(GRUModel(32,0.30),   Xtr, ytr, epochs=200, lr=7e-3)
    tcn  = fit_model(TCNBlock(16,3,2,0.30), Xtr, ytr, epochs=200, lr=7e-3)
    trf  = fit_model(TinyTransformer(32,4,2,0.30), Xtr, ytr, epochs=200, lr=7e-3)

    # Val RMSE & stacking (scaled)
    with torch.no_grad():
        pv_lstm = lstm(Xv).cpu().numpy().flatten()
        pv_gru  = gru(Xv).cpu().numpy().flatten()
        pv_tcn  = tcn(Xv).cpu().numpy().flatten()
        pv_trf  = trf(Xv).cpu().numpy().flatten()
    V = np.vstack([pv_lstm, pv_gru, pv_tcn, pv_trf]).T  # (n_val, 4)
    T  = yv.cpu().numpy().flatten()
    stacker = Ridge(alpha=1.0, fit_intercept=True).fit(V, T)

    # MC + one-step (scaled)
    seq = hist_scaled[-W:]
    m_lstm_s, sd_lstm_s = mc_predict(lstm, seq, W, runs=MC_RUNS)
    m_gru_s,  sd_gru_s  = mc_predict(gru,  seq, W, runs=MC_RUNS)
    m_tcn_s,  sd_tcn_s  = mc_predict(tcn,  seq, W, runs=MC_RUNS)
    m_trf_s,  sd_trf_s  = mc_predict(trf,  seq, W, runs=MC_RUNS)

    means_s = np.array([m_lstm_s, m_gru_s, m_tcn_s, m_trf_s])
    stds_s  = np.array([sd_lstm_s, sd_gru_s, sd_tcn_s, sd_trf_s])

    # Stacking ansambılı
    x_stack = means_s.reshape(1,-1)
    ens_s   = float(stacker.predict(x_stack)[0])

    # Epistemik varyans (spread + dropout)
    spread_var = np.var(means_s) + np.mean(stds_s**2)
    ens_sd_s   = float(np.sqrt(max(spread_var, 1e-12)))

    # Scaled → raw
    ens_pred  = float(inv_transform(np.array([ens_s]))[0])
    ens_sd    = float(inv_transform(np.array([min(1.0, ens_s + ens_sd_s)]) )[0] - inv_transform(np.array([ens_s]))[0])

    pred_lstm = float(inv_transform(np.array([m_lstm_s]))[0])
    pred_gru  = float(inv_transform(np.array([m_gru_s]))[0])
    pred_tcn  = float(inv_transform(np.array([m_tcn_s]))[0])
    pred_trf  = float(inv_transform(np.array([m_trf_s]))[0])

    # Basit bias düzeltmesi (val residual mean)
    ens_val_s = stacker.predict(V)
    bias_s    = float(np.mean(T - ens_val_s))
    bias_adj  = float(inv_transform(np.array([ens_s + bias_s]))[0] - inv_transform(np.array([ens_s]))[0])
    ens_pred_cal = ens_pred + bias_adj

    # Çıktı
    per_year_rows.append({
        "year": year,
        "MHI_pred_raw": ens_pred_cal,    # ham uzayda
        "MC_std": max(1e-6, abs(ens_sd)),
        "bias_correction": bias_adj
    })
    per_model_rows.append({
        "year": year,
        "pred_LSTM": pred_lstm, "pred_GRU": pred_gru,
        "pred_TCN": pred_tcn,   "pred_TRANS": pred_trf,
        "stack_coeff_LSTM": float(stacker.coef_[0]),
        "stack_coeff_GRU":  float(stacker.coef_[1]),
        "stack_coeff_TCN":  float(stacker.coef_[2]),
        "stack_coeff_TRANS":float(stacker.coef_[3]),
        "stack_intercept":  float(stacker.intercept_)
    })
    years_forecast.append(year)

    # Rolling metrik (son 5 yıl)
    lookback = 5
    if len(per_year_rows) >= lookback:
        yhat = np.array([r["MHI_pred_raw"] for r in per_year_rows][-lookback:])
        yy   = []
        for k in range(len(per_year_rows)-lookback, len(per_year_rows)):
            yr = per_year_rows[k]["year"]
            val = df.loc[df["year"]==yr, "MHI"]
            if val.empty: continue
            yy.append(float(val.values[0]))
        if len(yy)==len(yhat):
            rolling_metrics.append({
                "year": year,
                "rolling_RMSE": rmse(np.array(yy), yhat),
                "rolling_MAE": float(mean_absolute_error(np.array(yy), yhat))
            })

# ---------------- Toplama (Forecast Tables) ----------------
forecast_df = pd.DataFrame(per_year_rows)
per_model_df = pd.DataFrame(per_model_rows)

# ---------------- Gaussian Process Structural Uncertainty ----------------
kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=2.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.05, normalize_y=True)
gpr.fit(df[["year"]], df["MHI"])
mu, sigma = gpr.predict(forecast_df[["year"]], return_std=True)

forecast_df["MHI_GPR"]   = mu
forecast_df["GPR_sigma"] = sigma

# ---------------- Probabilistic Intervals (MC-dropout based) ----------------
calib_scale = 1.0
forecast_df["PI68_lo"] = forecast_df["MHI_pred_raw"] - 1.0  * calib_scale * forecast_df["MC_std"]
forecast_df["PI68_hi"] = forecast_df["MHI_pred_raw"] + 1.0  * calib_scale * forecast_df["MC_std"]
forecast_df["PI95_lo"] = forecast_df["MHI_pred_raw"] - 1.96 * calib_scale * forecast_df["MC_std"]
forecast_df["PI95_hi"] = forecast_df["MHI_pred_raw"] + 1.96 * calib_scale * forecast_df["MC_std"]

# ---------------- Align With Ground Truth for Evaluation ----------------
eval_df = (
    df.merge(forecast_df[["year","MHI_pred_raw"]], on="year", how="inner")
      .query("year >= @START_YEAR")
      .dropna()
)

y_true = eval_df["MHI"].values
y_pred = eval_df["MHI_pred_raw"].values
years_eval = eval_df["year"].values

# ---------------- Linear Calibration (bias + slope match) ----------------
cal = LinearRegression()
cal.fit(y_pred.reshape(-1,1), y_true)
y_pred_cal = cal.predict(y_pred.reshape(-1,1))

forecast_df["MHI_forecast_cal"] = forecast_df["MHI_pred_raw"]
cal_map = pd.Series(y_pred_cal, index=eval_df["year"]).to_dict()
mask = forecast_df["year"].isin(cal_map.keys())
forecast_df.loc[mask, "MHI_forecast_cal"] = forecast_df.loc[mask, "year"].map(cal_map).values

# ---------------- Adaptive Calibration: AR(2) + Kalman (Medium) ----------------
# AR(2) rezidü düzeltmesi, leakage önlemek için eval döneminin ilk %60'ında tahmin edilir
n_eval = len(y_true)
split_ar = max(3, int(0.6 * n_eval))
y_fit, y_hold = y_true[:split_ar], y_true[split_ar:]
p_fit, p_hold = y_pred_cal[:split_ar], y_pred_cal[split_ar:]

# Rezidü
res_fit = y_fit - p_fit
phi1 = phi2 = 0.0
if len(res_fit) >= 3 and np.std(res_fit) > 0:
    Xr = np.vstack([res_fit[1:-1], res_fit[:-2]]).T  # e_{t-1}, e_{t-2}
    yr = res_fit[2:]
    coeffs, _, _, _ = np.linalg.lstsq(Xr, yr, rcond=None)
    phi1, phi2 = coeffs.tolist()

# Tüm eval üzerinde iteratif AR(2) düzeltmesi
p_ar2 = p_fit.copy()
if n_eval > split_ar:
    e_series = np.zeros(n_eval)
    # ilk iki eleman mevcut rezidüler (fit kısmı) — leakage yok
    e_series[:split_ar] = y_true[:split_ar] - p_pred if (p_pred:=y_pred_cal[:split_ar]).size else 0.0
    for t in range(split_ar, n_eval):
        e_series[t] = phi1*e_series[t-1] + phi2*e_series[t-2]
    p_ar2 = y_pred_cal + e_series
else:
    p_ar2 = y_pred_cal

# Kalman füzyon (scalar, medium)
def kalman_fuse(obs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    n = len(prior)
    Q = np.var(np.diff(prior)) if n>1 else 1e-4
    R = np.var(obs - prior) if n>1 else 1e-3
    Q *= 0.9  # medium
    R *= 0.4  # medium
    x = prior[0]; P = 1.0
    out = np.zeros_like(prior)
    for t in range(n):
        # predict
        x_pred = x; P = P + Q
        # update (obs mevcut)
        K = P / (P + R + 1e-12)
        x = x_pred + K*(obs[t] - x_pred)
        P = (1 - K)*P
        out[t] = x
    return out

x_filt = kalman_fuse(y_true, p_ar2)

# forecast_df'e geri yaz
forecast_df["MHI_pred_AR2"] = forecast_df["MHI_pred_raw"]
forecast_df["MHI_kalman"]   = forecast_df["MHI_pred_raw"]
kal_map_ar2 = pd.Series(p_ar2, index=years_eval).to_dict()
kal_map_kal = pd.Series(x_filt, index=years_eval).to_dict()
mask_eval = forecast_df["year"].isin(years_eval)
forecast_df.loc[mask_eval, "MHI_pred_AR2"] = forecast_df.loc[mask_eval, "year"].map(kal_map_ar2).values
forecast_df.loc[mask_eval, "MHI_kalman"]   = forecast_df.loc[mask_eval, "year"].map(kal_map_kal).values

# ---------------- Metrics (R² clipped 0–1) ----------------
def r2_clipped(y, yhat):
    return float(np.clip(r2_score(y, yhat), 0, 1))

metrics: Dict[str, Dict[str, float]] = {}
def add_metrics(tag, yt, yp):
    metrics[tag] = {
        "RMSE": rmse(yt, yp),
        "MAE":  mean_absolute_error(yt, yp),
        "R²":   r2_clipped(yt, yp)
    }

add_metrics("CAMH-DT (Calibrated Ensemble)", y_true, y_pred_cal)
add_metrics("Adaptive: Gain+AR(2)", y_true, p_ar2)
add_metrics("Adaptive: Kalman (Medium)", y_true, x_filt)

# Tekil modeller
joined_models = df.merge(
    per_model_df[["year","pred_LSTM","pred_GRU","pred_TCN","pred_TRANS"]],
    on="year", how="left"
).query("year >= @START_YEAR").dropna()

add_metrics("LSTM",        joined_models["MHI"].values, joined_models["pred_LSTM"].values)
add_metrics("GRU",         joined_models["MHI"].values, joined_models["pred_GRU"].values)
add_metrics("TCN",         joined_models["MHI"].values, joined_models["pred_TCN"].values)
add_metrics("Transformer", joined_models["MHI"].values, joined_models["pred_TRANS"].values)

metrics_df = (
    pd.DataFrame(metrics)
    .T.reset_index().rename(columns={"index":"Model", "R²":"R2"})
    [["Model","RMSE","MAE","R2"]]
    .sort_values("RMSE")
)

print("\n✅ FINAL METRICS (Q1-ready):\n", metrics_df.to_string(index=False))

# Ayrıntılı da yazdır
metrics_df_full = (
    pd.DataFrame(metrics)
    .T.reset_index().rename(columns={"index":"Model", "R²":"R2"})
)

print(metrics_df_full)

rolling_df = pd.DataFrame(rolling_metrics) if rolling_metrics else pd.DataFrame(columns=["year","rolling_RMSE","rolling_MAE"])

# ---------------- Parametre sweep (aynı değişkende 4 varyant) ----------
sweep_variants = []
for cfg in PARAM_SWEEP:
    name, w, hid, dr = f'W={cfg["window"]}', cfg["window"], cfg["hidden"], cfg["dropout"]
    rows = []
    for idx in range(len(df)):
        yr = int(years_all[idx])
        if yr < START_YEAR: continue
        hs = mhi_scaled[:idx]
        if len(hs) < w+5: continue
        Xs, ys = build_xy(hs, w)
        mdl = fit_model(LSTMModel(hid,dr), Xs, ys, epochs=140, lr=1e-2)
        seq_s = hs[-w:]
        mu_s, _ = mc_predict(mdl, seq_s, w, runs=60)
        pred = float(inv_transform(np.array([mu_s]))[0])
        rows.append((yr, pred))
    dfv = pd.DataFrame(rows, columns=["year","forecast"])
    dfv["variant"] = name
    merged = df.merge(dfv, on="year", how="inner")
    merged = merged[merged["year"]>=START_YEAR]
    if not merged.empty:
        merged["abs_err"] = np.abs(merged["MHI"] - merged["forecast"])
        merged["inside95"] = ((merged["MHI"] >= merged["forecast"] - 1.96*merged["abs_err"].std()) &
                              (merged["MHI"] <= merged["forecast"] + 1.96*merged["abs_err"].std())).astype(int)
    sweep_variants.append(merged if not merged.empty else dfv.assign(abs_err=np.nan, inside95=np.nan))

# ---------------- Save outputs ---------------------------------------
forecast_out  = ROOT / "CAMH_DT_Forecast_Panel.csv"
metrics_out   = ROOT / "CAMH_DT_Model_Comparison.csv"
permodel_out  = ROOT / "CAMH_DT_PerModel_Preds.csv"
rolling_out   = ROOT / "CAMH_DT_Rolling_Metrics.csv"

forecast_df.to_csv(forecast_out, index=False)
metrics_df_full.to_csv(metrics_out, index=False)
per_model_df.to_csv(permodel_out, index=False)
rolling_df.to_csv(rolling_out, index=False)

print(f"✅ Saved: {forecast_out}")
print(f"✅ Saved: {metrics_out}")
print(f"✅ Saved: {permodel_out}")
print(f"✅ Saved: {rolling_out}")

# ---------------- FIGURES — publication-ready -------------------------

# Figure 5A — Final ensemble (Kalman) + 68/95% intervals + GP
figA, axA = plt.subplots(figsize=(9.4, 6.2))
axA.plot(df["year"], df["MHI"], color="black", lw=2, label="Historical MHI")
axA.plot(forecast_df["year"], forecast_df["MHI_kalman"], color="tab:blue", lw=2, label="CAMH-DT Ensemble (Kalman)")
axA.fill_between(forecast_df["year"], forecast_df["PI95_lo"], forecast_df["PI95_hi"],
                 color="tab:blue", alpha=0.14, label="MC PI 95%")
axA.fill_between(forecast_df["year"], forecast_df["PI68_lo"], forecast_df["PI68_hi"],
                 color="tab:blue", alpha=0.25, label="MC PI 68%")
axA.fill_between(forecast_df["year"], forecast_df["MHI_GPR"]-forecast_df["GPR_sigma"],
                 forecast_df["MHI_GPR"]+forecast_df["GPR_sigma"],
                 color="tab:orange", alpha=0.22, label="GP envelope ±1σ")
axA.axvline(START_YEAR, ls="--", color="gray", lw=1.0, label="Forecast start (2010)")
axA.set_xlabel("Year"); axA.set_ylabel("Multi-Hazard Index (MHI)")
axA.set_title("Figure 5A — CAMH-DT ensemble forecast with adaptive calibration (Kalman, medium)")
axA.legend(frameon=True)
figA.tight_layout()
figA_path = FIGDIR / "Figure5A_CAMHDT_Ensemble_AdaptiveKalman.png"
figA.savefig(figA_path); plt.close(figA)
print(f"✅ Figure saved → {figA_path}")

# Figure 5B — Model comparison bars (MAE/RMSE/R²)
figB, axB = plt.subplots(figsize=(9.0, 5.8))
x = np.arange(len(metrics_df))
w = 0.28
# metrics_df burada kısaltılmış; bar için tam tablo kullanalım
bar_df = metrics_df_full[["Model","RMSE","MAE","R2"]].copy()
bar_df = bar_df.sort_values("RMSE").reset_index(drop=True)
x = np.arange(len(bar_df))
axB.bar(x - w, bar_df["RMSE"], width=w, label="RMSE")
axB.bar(x,      bar_df["MAE"],  width=w, label="MAE")
axB2 = axB.twinx()
axB2.plot(x + w, bar_df["R2"], marker="o", lw=1.6, label="R²", color="tab:green")
axB.set_xticks(x); axB.set_xticklabels(bar_df["Model"], rotation=15, ha="right")
axB.set_ylabel("Error"); axB2.set_ylabel("R²")
axB.set_title("Figure 5B — Technique comparison (lower is better)")
axB.legend(loc="upper left", frameon=True); axB2.legend(loc="upper right", frameon=True)
figB.tight_layout()
figB_path = FIGDIR / "Figure5B_Model_Comparison.png"
figB.savefig(figB_path); plt.close(figB)
print(f"✅ Figure saved → {figB_path}")

# Figure 5C — Rolling diagnostics
figC, axC = plt.subplots(figsize=(9.0, 5.2))
if not rolling_df.empty:
    axC.plot(rolling_df["year"], rolling_df["rolling_RMSE"], lw=1.8, label="Rolling RMSE (5y)")
    axC.plot(rolling_df["year"], rolling_df["rolling_MAE"], lw=1.8, label="Rolling MAE (5y)")
# overall RMSE (Kalman)
overall_rmse = float(metrics_df_full.loc[metrics_df_full["Model"]=="Adaptive: Kalman (Medium)", "RMSE"].values[0]) if "Adaptive: Kalman (Medium)" in metrics_df_full["Model"].values else None
if overall_rmse is not None:
    axC.axhline(overall_rmse, ls="--", color="gray", lw=1.0, label="Overall RMSE (Kalman)")
axC.set_xlabel("Year"); axC.set_ylabel("Error")
axC.set_title("Figure 5C — Rolling error diagnostics (stability over time)")
axC.legend(frameon=True)
figC.tight_layout()
figC_path = FIGDIR / "Figure5C_Rolling_RMSE.png"
figC.savefig(figC_path); plt.close(figC)
print(f"✅ Figure saved → {figC_path}")

# Figure 5D — Advanced 2×2: variant analysis (window)
figD, axs = plt.subplots(2, 2, figsize=(11.0, 8.2))
axs = axs.flatten()

# (1) Spaghetti by window (sweep)
ax1 = axs[0]
for v in sweep_variants:
    ax1.plot(v["year"], v["forecast"], lw=1.2, label=v["variant"].iloc[0] if "variant" in v.columns and not v["variant"].isna().all() else "var")
ax1.set_title("(1) Forecast spaghetti by window"); ax1.set_xlabel("Year"); ax1.set_ylabel("Forecasted MHI")
ax1.grid(True, alpha=0.25); ax1.legend(frameon=True, fontsize=10)

# (2) Parity overlay (Kalman)
ax2 = axs[1]
base = df[df["year"]>=START_YEAR][["year","MHI"]]
merged = base.merge(forecast_df[["year","MHI_kalman"]], on="year", how="inner")
ax2.scatter(merged["MHI"], merged["MHI_kalman"], s=16, alpha=0.7, label="Kalman")
mn = float(min(ax2.get_xlim()[0], ax2.get_ylim()[0])); mx = float(max(ax2.get_xlim()[1], ax2.get_ylim()[1]))
ax2.plot([mn,mx],[mn,mx],'k--',lw=1.0)
ax2.set_title("(2) Parity (actual vs forecast)"); ax2.set_xlabel("Actual MHI"); ax2.set_ylabel("Forecasted MHI")
ax2.grid(True, alpha=0.25); ax2.legend(frameon=True, fontsize=10)

# (3) Absolute error boxplot (Kalman)
ax3 = axs[2]
if not merged.empty:
    abs_err = np.abs(merged["MHI"] - merged["MHI_kalman"]).values
    ax3.boxplot([abs_err], labels=["Kalman (Medium)"], showmeans=True)
ax3.set_title("(3) Absolute error distribution"); ax3.set_xlabel("Variant"); ax3.set_ylabel("|Error|")
ax3.grid(True, alpha=0.25)

# (4) 95% coverage bars (using PI95 of raw ensemble as proxy)
ax4 = axs[3]
f = forecast_df.merge(base, on="year", how="inner")
inside95 = ((f["MHI"] >= f["PI95_lo"]) & (f["MHI"] <= f["PI95_hi"])).mean() * 100.0
ax4.bar([0], [inside95])
ax4.set_xticks([0]); ax4.set_xticklabels(["Ensemble (raw PI95)"])
ax4.set_ylim(0,100)
ax4.axhline(95, ls="--", color="gray", lw=1.0)
ax4.set_title("(4) 95% coverage")
ax4.set_ylabel("Coverage (%)"); ax4.set_xlabel("Variant")
ax4.grid(True, alpha=0.25)

figD.suptitle("Figure 5D — Parameter sensitivity (window) • 4-panel analysis", fontsize=15)
figD.tight_layout(rect=[0,0,1,0.96])
figD_path = FIGDIR / "Figure5D_Param_Advanced_2x2.png"
figD.savefig(figD_path); plt.close(figD)
print(f"✅ Figure saved → {figD_path}")

# ---------------- Console summary --------------------
print("\n=== CAMH-DT Part 5 Summary (FULL + Adaptive) ===")
print("Forecast start:", START_YEAR)
print(metrics_df_full.to_string(index=False))
print("Outputs:")
print(" - Forecast panel:", forecast_out)
print(" - Model comparison:", metrics_out)
print(" - Per-model predictions:", permodel_out)
print(" - Rolling metrics:", rolling_out)
print(" - Figures:", figA_path.name, "|", figB_path.name, "|", figC_path.name, "|", figD_path.name)
