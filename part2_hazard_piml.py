# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:53:13 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
CAMH-DT — Part 2: Satellite-Informed Hazard Characterisation (Physics-Informed)
Generates sub-indices Ŵ (wind) and F̂ (flood), and Figures 1–2.

Figures:
    Figure 1: C3S Windstorm Indicators — Annual Trends (a–d)
    Figure 2: ERA5 Storm-Track Density Maps (a–d)

Output:
    feature_store/hazard_indices.parquet
    figures/Figure1_C3S_Windstorm_Indicators.png
    figures/Figure2_ERA5_StormTracks.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Optional cartopy for geo maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import zscore

mpl.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
})

ROOT = Path("./feature_store")
FIGDIR = Path("./figures")
FIGDIR.mkdir(exist_ok=True, parents=True)


# -------------------------------------------------------------------------
# Load processed data from Part 1
# -------------------------------------------------------------------------
wind_path = ROOT / "c3s_wind_ind.parquet"
tracks_path = ROOT / "era5_storm_tracks.parquet"
if not wind_path.exists() or not tracks_path.exists():
    raise FileNotFoundError("Run part1_ingest.py first to generate feature_store parquet files.")

df_wind = pd.read_parquet(wind_path)
df_tracks = pd.read_parquet(tracks_path)

# -------------------------------------------------------------------------
# Physics-Informed smoothing for C3S wind indicators
# -------------------------------------------------------------------------

def physics_informed_smooth(series, window=5, poly=2):
    """Savitzky–Golay smoothing with energy-proportional regularisation."""
    s = series.copy().astype(float)
    s = s.fillna(s.interpolate(limit_direction='both'))
    # physical consistency: wind energy ∝ v^3
    energy = np.power(s, 3)
    smooth_e = savgol_filter(energy, window_length=window, polyorder=poly, mode="interp")
    smooth_v = np.cbrt(np.maximum(smooth_e, 0))
    return smooth_v


def normalise(series):
    s = (series - series.min()) / (series.max() - series.min())
    return s


# Identify indicator columns (excluding metadata)
indicator_cols = [c for c in df_wind.columns if c not in {"year", "source"}]

# Create smoothed and normalised versions
wind_proc = df_wind[["year"]].copy()
for col in indicator_cols:
    wind_proc[col + "_smooth"] = physics_informed_smooth(df_wind[col])
    wind_proc[col + "_norm"] = normalise(wind_proc[col + "_smooth"])

# Derive composite storm severity indicator
wind_proc["storm_severity_index"] = wind_proc[[c for c in wind_proc.columns if c.endswith("_norm")]].mean(axis=1)
wind_proc["storm_severity_index_smooth"] = savgol_filter(wind_proc["storm_severity_index"], 7, 2)
wind_proc["source"] = "C3S_Wind"

# -------------------------------------------------------------------------
# Plot Figure 1 — C3S Windstorm Indicators (a–d)
# -------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
axs = axs.flatten()

cols = indicator_cols[:4] if len(indicator_cols) >= 4 else indicator_cols
titles = ["(a) Max wind-gust index",
          "(b) Severe-storm days",
          "(c) Mean wind-speed anomaly",
          "(d) Composite storm-severity index"]

for i, (ax, col, title) in enumerate(zip(axs, cols + [None]*(4-len(cols)), titles)):
    if col:
        y = wind_proc[col + "_smooth"]
        ax.plot(wind_proc["year"], y, lw=1.8, color="tab:blue")
        ax.fill_between(wind_proc["year"], y*0.95, y*1.05, alpha=0.2, color="tab:blue")
    else:
        ax.axis("off")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised index")
    ax.grid(True, alpha=0.3)
fig.suptitle("C3S Windstorm Indicators — Annual Trends", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
f1_path = FIGDIR / "Figure1_C3S_Windstorm_Indicators.png"
fig.savefig(f1_path, dpi=600)
plt.close(fig)

print(f"✅ Figure 1 saved: {f1_path.resolve()}")

# -------------------------------------------------------------------------
# ERA5 Storm Tracks — compute storm density maps
# -------------------------------------------------------------------------

df_tracks = df_tracks.dropna(subset=["lat", "lon"])
lon_bins = np.linspace(-180, 180, 72)
lat_bins = np.linspace(-90, 90, 36)
H, xedges, yedges = np.histogram2d(df_tracks["lon"], df_tracks["lat"],
                                   bins=[lon_bins, lat_bins])
density = H.T  # transpose for correct orientation

# -------------------------------------------------------------------------
# Figure 2 — ERA5 Storm Track Density Maps (a–d)
# -------------------------------------------------------------------------
if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(11, 7))
    panels = [221, 222, 223, 224]
    titles = [
        "(a) Global track density",
        "(b) Log-scaled global density",
        "(c) UK-focused domain (−12°–4°, 49°–61° N)",
        "(d) Seasonal composites (DJF, MAM, JJA, SON)"
    ]

    # (a) raw density
    ax1 = plt.subplot(panels[0], projection=proj)
    ax1.set_title(titles[0])
    pcm = ax1.pcolormesh(xedges, yedges, density,
                         cmap="viridis", transform=proj)
    ax1.add_feature(cfeature.COASTLINE, lw=0.4)
    ax1.set_global()
    plt.colorbar(pcm, ax=ax1, shrink=0.7, label="Track count")

    # (b) log-scaled
    ax2 = plt.subplot(panels[1], projection=proj)
    ax2.set_title(titles[1])
    pcm2 = ax2.pcolormesh(xedges, yedges, np.log1p(density),
                          cmap="plasma", transform=proj)
    ax2.add_feature(cfeature.COASTLINE, lw=0.4)
    ax2.set_global()
    plt.colorbar(pcm2, ax=ax2, shrink=0.7, label="log(1+count)")

    # (c) UK zoom
    ax3 = plt.subplot(panels[2], projection=proj)
    ax3.set_extent([-12, 4, 49, 61])
    ax3.set_title(titles[2])
    pcm3 = ax3.pcolormesh(xedges, yedges, np.log1p(density),
                          cmap="YlOrRd", transform=proj)
    ax3.add_feature(cfeature.COASTLINE, lw=0.6)
    plt.colorbar(pcm3, ax=ax3, shrink=0.8, label="log(1+count)")

    # (d) seasonal composites
    ax4 = plt.subplot(panels[3], projection=proj)
    ax4.set_extent([-12, 4, 49, 61])
    ax4.set_title(titles[3])
    seasons = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    for s, months in seasons.items():
        subset = df_tracks[df_tracks["month"].isin(months)]
        ax4.scatter(subset["lon"], subset["lat"], s=1.0, alpha=0.15, label=s, transform=proj)
    ax4.add_feature(cfeature.COASTLINE, lw=0.4)
    ax4.legend(loc="lower left", fontsize=9, frameon=True)
    fig.suptitle("ERA5 Storm-Track Density Maps", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f2_path = FIGDIR / "Figure2_ERA5_StormTracks.png"
    fig.savefig(f2_path, dpi=600)
    plt.close(fig)
    print(f"✅ Figure 2 saved: {f2_path.resolve()}")
else:
    print("⚠️ Cartopy not installed; skipping geographic maps. Install with `pip install cartopy`.")

# -------------------------------------------------------------------------
# Derived sub-index export (Ŵ and F̂ placeholders)
# -------------------------------------------------------------------------
wind_proc.rename(columns={"storm_severity_index_smooth": "W_hat"}, inplace=True)
# F̂ (flood) placeholder: will be produced in Part 3
wind_proc["F_hat"] = np.nan
out_path = ROOT / "hazard_indices.parquet"
wind_proc.to_parquet(out_path, index=False)
print(f"✅ Hazard sub-indices saved: {out_path.resolve()}")
