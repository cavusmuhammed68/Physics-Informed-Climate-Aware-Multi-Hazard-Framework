# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 23:17:43 2025

@author: nfpm5
"""

# ==============================================================
# CAMH-DT UK GRID-FAILURE POSSIBILITY — MAP + TEMPORAL (Q1-ready)
# ==============================================================

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- Paths -----------------------------
BASE_DIR   = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Upcoming Papers\TSG - 1 Grid Failure (Submitted)\UK datasets"
RESULTS_DIR= os.path.join(BASE_DIR, "Results_CAMH_DT")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES = {
    "c3s"    : "C3S_Operational_Windstorm_TIER_1_INDICATORS_ANNUAL_v1.2.xlsx",
    "curt"   : "curtailment-events-site-specific.csv",
    "feeders": "npg-ehv-feeders.csv",
    "risk"   : "UKC11_indicator_risk_all.csv",
    "loss"   : "UKC11_indicator_losses_all.csv",
    "tier2"  : "UKC11_tier2_all.csv",
}

# --------------------------- Plot style --------------------------
mpl.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25
})

def savefig(fig, name):
    out = os.path.join(RESULTS_DIR, name)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"[saved] {out}")

def read_file(name):
    p = os.path.join(BASE_DIR, name)
    if not os.path.exists(p):
        print(f"[warn] missing: {name}"); return pd.DataFrame()
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(p, engine="openpyxl")
    return pd.read_csv(p, low_memory=False)

def detect_col(df, subs):
    for c in df.columns:
        if any(s in c.lower() for s in subs): return c
    return None

def scale01(x):
    x = pd.to_numeric(x, errors="coerce")
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn: return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)

# ==============================================================
# 1) Load data
# ==============================================================
D = {k: read_file(v) for k, v in FILES.items()}

# ==============================================================
# 2) Hazard: C3S → yearly hazard index (normalised)
# ==============================================================
c3s = D["c3s"].copy()
if not c3s.empty:
    ycol = detect_col(c3s, ["year","date"])
    if ycol is None: raise KeyError("Year/Date column not found in C3S file.")
    if "date" in ycol.lower():
        c3s["Year"] = pd.to_datetime(c3s[ycol], errors="coerce").dt.year
    else:
        c3s["Year"] = pd.to_numeric(c3s[ycol], errors="coerce")
    c3s = c3s.dropna(subset=["Year"])
    numcols = c3s.select_dtypes("number").columns.tolist()
    if "Year" in numcols: numcols.remove("Year")
    c3s["Hazard_Index"] = scale01(c3s[numcols].mean(axis=1))
else:
    raise ValueError("C3S dataset is empty.")

# ==============================================================
# 3) Multi-hazard context (UKC11): compute MHI mean level
# ==============================================================
risk, loss, tier2 = D["risk"], D["loss"], D["tier2"]
if risk.empty or loss.empty or tier2.empty:
    print("[warn] UKC11 tables missing → MHI defaults to 0.5")
    MHI_mean = 0.5
else:
    r = scale01(risk.select_dtypes("number").mean(axis=1))
    l = scale01(loss.select_dtypes("number").mean(axis=1))
    t = scale01(tier2.select_dtypes("number").mean(axis=1))
    MHI_mean = np.nanmean((r + l + t)/3.0)

# ==============================================================
# 4) Curtailment (robust ISO-8601 with timezone)
# ==============================================================
curt = D["curt"].copy()
if curt.empty: raise ValueError("Curtailment file is empty.")
# we know from your logs the column is "Start time UTC"
tcol = "Start time UTC" if "Start time UTC" in curt.columns else detect_col(curt, ["start","date","datetime","time"])
vcol = detect_col(curt, ["mw","mwh","curtail","value"])
if tcol is None or vcol is None:
    raise KeyError(f"Could not detect time/value columns in curtailment table. Columns: {list(curt.columns)}")

curt["_ts"] = pd.to_datetime(curt[tcol], utc=True, errors="coerce").dt.tz_convert(None)
curt[vcol]   = pd.to_numeric(curt[vcol], errors="coerce")
curt = curt.dropna(subset=["_ts", vcol])
curt["Year"] = curt["_ts"].dt.year.astype(int)
curt_y = curt.groupby("Year")[vcol].sum().reset_index()
curt_y["Curtail_Norm"] = scale01(curt_y[vcol])

# ==============================================================
# 5) Feeders — parse coordinates from Geo Point / Geo Shape
# ==============================================================
feed = D["feeders"].copy()
if feed.empty: raise ValueError("Feeders file is empty.")

def parse_geo_point(s):
    if pd.isna(s): return np.nan, np.nan
    s = str(s).replace("POINT","").replace("("," ").replace(")"," ").replace(","," ")
    parts = [p for p in s.split() if p.replace('.','',1).replace('-','',1).isdigit()]
    if len(parts) < 2: return np.nan, np.nan
    a, b = float(parts[0]), float(parts[1])
    # decide order by UK ranges
    if 49 <= a <= 61 and -12 <= b <= 4:   # (lat, lon)
        return a, b
    if 49 <= b <= 61 and -12 <= a <= 4:   # (lon, lat) → swap
        return b, a
    return np.nan, np.nan

def parse_geo_shape(s):
    if pd.isna(s): return np.nan, np.nan
    m = re.findall(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)", str(s))
    if not m: return np.nan, np.nan
    x, y = map(float, m[0])
    # WKT is usually (lon lat)
    if 49 <= y <= 61 and -12 <= x <= 4: return y, x
    if 49 <= x <= 61 and -12 <= y <= 4: return x, y
    return np.nan, np.nan

if "Geo Point" in feed.columns:
    latlon = feed["Geo Point"].apply(lambda v: pd.Series(parse_geo_point(v)))
elif "Geo Shape" in feed.columns:
    latlon = feed["Geo Shape"].apply(lambda v: pd.Series(parse_geo_shape(v)))
else:
    raise KeyError(f"No Geo Point/Geo Shape column. Columns: {list(feed.columns)}")

feed["lat"], feed["lon"] = latlon[0], latlon[1]
feed = feed.dropna(subset=["lat","lon"])
if feed.empty: raise ValueError("No valid feeder coordinates could be parsed.")

# exposure proxy: you can replace with a true attribute if present
feed["Exposure"] = MinMaxScaler().fit_transform(np.arange(len(feed)).reshape(-1,1)).ravel()

# ==============================================================
# 6) CAMH-DT fusion  α·MHI + β·Exposure + γ·Curtailment
# ==============================================================
α, β, γ = 0.55, 0.30, 0.15
curt_mean = float(curt_y["Curtail_Norm"].mean()) if not curt_y.empty else 0.0

# feeder-level probability (spatial weights)
feed["grid_fail_prob"] = np.clip(α*MHI_mean + β*feed["Exposure"] + γ*curt_mean, 0, 1)

# temporal evolution (yearly)
combo = pd.merge(
    c3s[["Year","Hazard_Index"]],
    curt_y[["Year","Curtail_Norm"]],
    on="Year", how="outer"
).fillna(0.0)
combo = combo.sort_values("Year")
combo["Failure_Possibility"] = np.clip(α*combo["Hazard_Index"] + γ*combo["Curtail_Norm"], 0, 1)

# ==============================================================
# 7) Spatial field on UK grid (proper smoothing + extent)
#    Nadaraya–Watson: smooth numerator & denominator then divide
# ==============================================================
UK_EXTENT = [-12, 4, 49, 61]   # lon_min, lon_max, lat_min, lat_max
NX = NY = 240                  # resolution
lon_edges = np.linspace(UK_EXTENT[0], UK_EXTENT[1], NX+1)
lat_edges = np.linspace(UK_EXTENT[2], UK_EXTENT[3], NY+1)

# numerator: sum(prob) per bin; denominator: count per bin
num, _, _ = np.histogram2d(feed["lon"], feed["lat"],
                           bins=[lon_edges, lat_edges],
                           weights=feed["grid_fail_prob"])
den, _, _ = np.histogram2d(feed["lon"], feed["lat"],
                           bins=[lon_edges, lat_edges],
                           weights=np.ones(len(feed)))

# smooth both → avoids tiny blocky patch & preserves support
SIGMA = 1.8
num_s = gaussian_filter(num, SIGMA, mode="nearest")
den_s = gaussian_filter(den, SIGMA, mode="nearest")
grid  = np.divide(num_s, den_s, out=np.full_like(num_s, np.nan), where=den_s>0)

# robust colour limits
vmin = np.nanpercentile(grid, 2)
vmax = np.nanpercentile(grid, 98)
norm = Normalize(vmin=vmin, vmax=vmax)

# ==============================================================
# 8) Figure: Map (top) + Temporal evolution (bottom)
# ==============================================================
fig = plt.figure(figsize=(8.5, 10))
gs  = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.28)

# --- MAP ---
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
# draw raster directly over UK, not as a small rectangle
img = ax1.imshow(grid.T, origin="lower", cmap="turbo", norm=norm,
                 extent=UK_EXTENT, transform=ccrs.PlateCarree(), interpolation="bilinear")
ax1.coastlines(linewidth=0.6)
ax1.add_feature(cfeature.BORDERS, linewidth=0.4)
ax1.add_feature(cfeature.LAND, facecolor="#eeeeee", alpha=0.5, zorder=0)
ax1.add_feature(cfeature.OCEAN, facecolor="#f9f9f9", zorder=0)
ax1.set_extent(UK_EXTENT)
ax1.set_title("CAMH-DT Estimated Grid-Failure Possibility over the UK", pad=10, weight="bold")

# embedded colourbar inside map
cax = inset_axes(ax1, width="32%", height="3%", loc="lower right", borderpad=1.2)
cb  = plt.colorbar(img, cax=cax, orientation="horizontal")
cb.set_label("Grid-failure possibility (0–1)", fontsize=9)
cb.ax.tick_params(labelsize=8)
cb.outline.set_visible(False)

# --- TEMPORAL EVOLUTION ---
ax2 = fig.add_subplot(gs[1])
if not combo.empty:
    ax2.plot(combo["Year"], combo["Failure_Possibility"], color="firebrick", lw=2)
    lo = np.maximum(0, combo["Failure_Possibility"]*0.92)
    hi = np.minimum(1, combo["Failure_Possibility"]*1.08)
    ax2.fill_between(combo["Year"], lo, hi, color="firebrick", alpha=0.15, lw=0)
ax2.set_xlabel("Year")
ax2.set_ylabel("Mean grid-failure possibility")
ax2.set_title("Temporal evolution of mean grid-failure possibility", fontsize=13)
ax2.grid(True, ls="--", alpha=0.35)
if not combo.empty:
    ax2.set_xlim(combo["Year"].min(), combo["Year"].max())
    ax2.set_ylim(0, max(0.05, float(np.nanmax(combo["Failure_Possibility"]))*1.1))

fig.tight_layout()
savefig(fig, "fig_CAMHDT_UK_GridFailure_Map_Temporal.png")
