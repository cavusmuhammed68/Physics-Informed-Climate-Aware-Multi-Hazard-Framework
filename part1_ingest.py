# -*- coding: utf-8 -*-
"""
CAMH-DT — Part 1: Data Ingestion, Configuration & Provenance
Q1-ready ingestion pipeline for climate-aware multi-hazard digital twin studies.

Author: <your name>
Affiliation: Northumbria University
Licence: MIT (or as required by the journal)

This module:
    - Reads all raw datasets (C3S/ERA5, UKC11, curtailment, embedded capacity, EHV feeders)
    - Normalises schemas and units; harmonises time (year/season/month/week)
    - Performs basic validation & quality checks (nulls, ranges, outliers)
    - Generates a data dictionary (column-level metadata) and a quality report
    - Writes cleaned tables to a feature store (parquet) with stable names
    - Records full provenance (file hashes, sizes, modified times, source path)
"""

from __future__ import annotations

import os
import io
import re
import sys
import json
import yaml
import math
import time
import hashlib
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional geospatial support
try:
    import geopandas as gpd  # type: ignore
    _HAS_GPD = True
except Exception:
    _HAS_GPD = False
    warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "project_name": "CAMH-DT",
    "random_seed": 42,
    "timezone": "Europe/London",
    "io": {
        # User-provided Windows path (primary)
        "primary_root": r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Satellite Paper\UK datasets\Results_new",
        # Fallback for this execution environment (uploaded files)
        "fallback_root": "/mnt/data",
        "output_root": "./feature_store",
        "fig_root": "./figures",
        "results_root": "./results",
    },
    "files": {
        "c3s_wind_ind": "C3S_Operational_Windstorm_TIER_1_INDICATORS_ANNUAL_v1.2.xlsx",
        "era5_tracks": "C3S_StormTracks_ERA5_1979_2021_clean.csv",
        "ukc11_risk": "UKC11_indicator_risk.csv",
        "ukc11_losses": "UKC11_indicator_losses.csv",
        "ukc11_tier2": "UKC11_tier2.csv",
        "curtailment": "curtailment-events-site-specific.csv",
        "embedded_capacity": "embedded-capacity-register-part-2.csv",
        "ehv_feeders": "npg-ehv-feeders.csv",
    },
    "outputs": {
        "c3s_wind_ind": "c3s_wind_ind.parquet",
        "era5_tracks": "era5_storm_tracks.parquet",
        "ukc11_risk": "ukc11_indicator_risk.parquet",
        "ukc11_losses": "ukc11_indicator_losses.parquet",
        "ukc11_tier2": "ukc11_tier2.parquet",
        "curtailment": "curtailment_events.parquet",
        "embedded_capacity": "embedded_capacity.parquet",
        "ehv_feeders": "npg_ehv_feeders.parquet",
        "data_dictionary": "data_dictionary.json",
        "quality_report": "quality_report.json",
        "provenance": "provenance_manifest.json",
        "combined_calendar": "calendar_index.parquet",
    },
    "schema": {
        # Minimal expected columns per file (flexible; will warn not fail)
        "c3s_wind_ind": ["year"],
        "era5_tracks": ["storm_id", "datetime", "lat", "lon"],
        "ukc11_risk": ["area_code", "year"],
        "ukc11_losses": ["area_code", "year"],
        "ukc11_tier2": ["area_code", "year"],
        "curtailment": ["site_id", "timestamp"],
        "embedded_capacity": ["generator_id", "installed_capacity_mw", "lat", "lon"],
        "ehv_feeders": ["feeder_id", "lat", "lon"],
    },
    "time_index": {
        "freqs": ["A", "Q", "M", "W"],
        "start_year": 1979,
        "end_year": 2025
    },
    "numeric_outlier": {
        "zscore_threshold": 4.0,
        "iqr_multiplier": 3.0
    }
}


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("camh_dt.part1")
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


LOGGER = setup_logging()


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load YAML config if provided; otherwise returns DEFAULT_CONFIG.
    """
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
        # Deep-merge (user overrides defaults)
        cfg = DEFAULT_CONFIG.copy()
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
        LOGGER.info("Loaded user config from %s", config_path)
        return cfg
    LOGGER.info("Using DEFAULT_CONFIG")
    return DEFAULT_CONFIG


def detect_root_paths(cfg: Dict) -> Tuple[Path, Path, Path]:
    """
    Resolve primary and fallback roots; ensure output directories exist.
    """
    primary = Path(cfg["io"]["primary_root"])
    fallback = Path(cfg["io"]["fallback_root"])
    output_root = Path(cfg["io"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    Path(cfg["io"]["fig_root"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["io"]["results_root"]).mkdir(parents=True, exist_ok=True)

    if primary.exists():
        LOGGER.info("Primary root found: %s", primary)
        return primary, fallback, output_root
    LOGGER.warning("Primary root not found; using fallback: %s", fallback)
    return fallback, fallback, output_root


def smart_path(root: Path, fname: str, fallback: Path) -> Path:
    """
    Prefer file under `root`; if missing, try in `fallback`.
    """
    p = root / fname
    if p.exists():
        return p
    q = fallback / fname
    return q


def file_digest(path: Path, chunk_size: int = 1 << 20) -> Dict[str, str | int]:
    """
    Compute SHA256 hash plus size/modtime for provenance.
    """
    sha = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                size += len(block)
                sha.update(block)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        return {"sha256": sha.hexdigest(), "bytes": size, "modified": mtime}
    except Exception as e:
        LOGGER.warning("Provenance failed for %s: %s", path, e)
        return {"sha256": "NA", "bytes": -1, "modified": "NA"}


def try_parse_datetime(x) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(x, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def to_float(x) -> Optional[float]:
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() in {"na", "nan", "none"}:
                return np.nan
        return float(x)
    except Exception:
        return np.nan


def clip_coords(lat: pd.Series, lon: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Basic geographic sanity bounds; preserves NaNs.
    """
    lat_c = lat.where(lat.isna() | ((lat >= -90) & (lat <= 90)))
    lon_c = lon.where(lon.isna() | ((lon >= -180) & (lon <= 180)))
    return lat_c, lon_c


def zscore_flags(s: pd.Series, thr: float = 4.0) -> pd.Series:
    if s.dtype.kind not in "biufc":
        return pd.Series(False, index=s.index)
    mu, sd = s.mean(), s.std(ddof=0)
    if (sd is None) or (sd == 0) or not np.isfinite(sd):
        return pd.Series(False, index=s.index)
    z = (s - mu) / sd
    return z.abs() > thr


def iqr_flags(s: pd.Series, mult: float = 3.0) -> pd.Series:
    if s.dtype.kind not in "biufc":
        return pd.Series(False, index=s.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    return (s < lo) | (s > hi)


# --------------------------------------------------------------------------------------
# Data Loaders
# --------------------------------------------------------------------------------------

def load_c3s_wind_indicators(path: Path) -> pd.DataFrame:
    """
    Load C3S Tier-1 Windstorm Annual Indicators (Excel).
    Must contain a 'year' column; other columns are carried through.
    """
    LOGGER.info("Loading C3S wind indicators: %s", path)
    df = pd.read_excel(path, engine="openpyxl")
    # Standardise column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
    )
    if "year" not in df.columns:
        # Attempt to infer
        cand = [c for c in df.columns if re.fullmatch(r"year|yr|\by\b", c)]
        if cand:
            df = df.rename(columns={cand[0]: "year"})
        else:
            raise ValueError("C3S wind indicators require a 'year' column.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df = df.sort_values("year").reset_index(drop=True)
    df["source"] = "C3S_Tier1_Wind_Annual"
    return df


def load_era5_tracks(path: Path) -> pd.DataFrame:
    """
    Load ERA5 storm tracks.
    Expected columns: storm_id, datetime, lat, lon (others kept).
    """
    LOGGER.info("Loading ERA5 storm tracks: %s", path)
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
    )
    # Normalise key fields
    for col in ["lat", "latitude"]:
        if col in df.columns:
            df["lat"] = pd.to_numeric(df[col], errors="coerce")
            break
    for col in ["lon", "longitude"]:
        if col in df.columns:
            df["lon"] = pd.to_numeric(df[col], errors="coerce")
            break
    if "datetime" not in df.columns:
        # try: date or time columns
        for c in ["time", "timestamp", "date", "datetime_utc"]:
            if c in df.columns:
                df["datetime"] = pd.to_datetime(df[c], errors="coerce", utc=True)
                break
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    df["storm_id"] = df.get("storm_id", pd.Series(dtype="object")).astype("string")
    df["lat"], df["lon"] = clip_coords(df["lat"], df["lon"])
    df["year"] = df["datetime"].dt.year.astype("Int64")
    df["month"] = df["datetime"].dt.month.astype("Int64")
    df["source"] = "ERA5_StormTracks"
    return df


def load_ukc11_csv(path: Path, tag: str) -> pd.DataFrame:
    """
    Load UKC11 indicator tables (risk / losses / tier2).
    """
    LOGGER.info("Loading UKC11 %s: %s", tag, path)
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
    )
    # Canonical id fields
    if "area_code" not in df.columns:
        # try common alternates
        for c in ["area", "code", "lad_code", "region", "region_code"]:
            if c in df.columns:
                df = df.rename(columns={c: "area_code"})
                break
    if "year" not in df.columns:
        for c in ["yr", "date", "timestamp"]:
            if c in df.columns:
                df["year"] = pd.to_datetime(df[c], errors="coerce").dt.year.astype("Int64")
                break
    df["year"] = pd.to_numeric(df.get("year", np.nan), errors="coerce").astype("Int64")
    df["source"] = f"UKC11_{tag}"
    return df


def load_curtailment(path: Path) -> pd.DataFrame:
    """
    Enhanced Curtailment Loader (auto-fallback enabled)
    ---------------------------------------------------
    - Automatically detects alternative filenames if primary file not found
    - Harmonises timestamp and coordinate fields
    - Ensures numeric fields are consistent
    - Returns a clean dataframe ready for analysis and parquet export
    """

    LOGGER.info("Loading curtailment events: %s", path)

    # ------------------------------------------------------------------
    # Step 1: If path missing, auto-detect similar files in directory
    # ------------------------------------------------------------------
    if not path.exists():
        LOGGER.warning("Curtailment file not found at %s — attempting auto-detect...", path)
        root = path.parent
        candidates = list(root.glob("*curtailment*site*.csv")) + list(root.glob("*Curtailment*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"❌ No curtailment file found near {root}. "
                "Expected something like 'curtailment-events-site-specific.csv'.")
        path = candidates[0]
        LOGGER.warning("✅ Auto-detected curtailment file: %s", path.name)

    # ------------------------------------------------------------------
    # Step 2: Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
    )

    # ------------------------------------------------------------------
    # Step 3: Handle timestamps
    # ------------------------------------------------------------------
    ts_col = None
    for cand in ["timestamp", "time", "datetime", "event_time", "start_time"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError("Curtailment file must contain a timestamp/time/datetime column.")

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year.astype("Int64")
    df["week"] = df["timestamp"].dt.isocalendar().week.astype("Int64")
    df["month"] = df["timestamp"].dt.month.astype("Int64")

    # ------------------------------------------------------------------
    # Step 4: Coordinates
    # ------------------------------------------------------------------
    if "lat" in df.columns or "latitude" in df.columns:
        df["lat"] = pd.to_numeric(df.get("lat", df.get("latitude")), errors="coerce")
    if "lon" in df.columns or "longitude" in df.columns:
        df["lon"] = pd.to_numeric(df.get("lon", df.get("longitude")), errors="coerce")

    if "lat" in df.columns and "lon" in df.columns:
        df["lat"], df["lon"] = clip_coords(df["lat"], df["lon"])

    # ------------------------------------------------------------------
    # Step 5: Numeric cleanup
    # ------------------------------------------------------------------
    for col in ["curtailed_mw", "curtailment_mw", "lost_mw", "energy_mwh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Step 6: Final metadata
    # ------------------------------------------------------------------
    df["source"] = "NPG_Curtailment_Auto"
    LOGGER.info("Curtailment dataset loaded: %d records, columns=%d", len(df), df.shape[1])
    return df



def load_embedded_capacity(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading embedded capacity: %s", path)
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
    )
    # Normalise capacity units
    cap_cols = [c for c in df.columns if "capacity" in c]
    if cap_cols:
        # Heuristic: prefer installed_capacity_mw; else convert kw->mw if necessary
        if "installed_capacity_mw" in df.columns:
            df["installed_capacity_mw"] = pd.to_numeric(df["installed_capacity_mw"], errors="coerce")
        elif "installed_capacity_kw" in df.columns:
            df["installed_capacity_mw"] = pd.to_numeric(df["installed_capacity_kw"], errors="coerce") / 1000.0
        else:
            # pick first capacity-like column and assume MW
            c0 = cap_cols[0]
            df["installed_capacity_mw"] = pd.to_numeric(df[c0], errors="coerce")

    # Coordinates if present
    for cand in ["lat", "latitude"]:
        if cand in df.columns:
            df["lat"] = pd.to_numeric(df[cand], errors="coerce")
            break
    for cand in ["lon", "longitude"]:
        if cand in df.columns:
            df["lon"] = pd.to_numeric(df[cand], errors="coerce")
            break
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"], df["lon"] = clip_coords(df["lat"], df["lon"])

    df["source"] = "Embedded_Capacity"
    return df


def load_ehv_feeders(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading EHV feeders: %s", path)
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
    )
    # Normalise coords
    for cand in ["lat", "latitude"]:
        if cand in df.columns:
            df["lat"] = pd.to_numeric(df[cand], errors="coerce")
            break
    for cand in ["lon", "longitude"]:
        if cand in df.columns:
            df["lon"] = pd.to_numeric(df[cand], errors="coerce")
            break
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"], df["lon"] = clip_coords(df["lat"], df["lon"])

    df["source"] = "NPG_EHV_Feeders"
    return df


# --------------------------------------------------------------------------------------
# Harmonisation, Validation, Quality
# --------------------------------------------------------------------------------------

def validate_minimal_schema(df: pd.DataFrame, expected_cols: List[str], label: str) -> List[str]:
    """
    Check presence of key columns; return missing list (warning only).
    """
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        LOGGER.warning("[%s] Missing columns: %s", label, missing)
    return missing


def add_calendar_index(cfg: Dict) -> pd.DataFrame:
    """
    Build a master calendar index (A/Q/M/W) to aid later joins.
    """
    tz = cfg.get("timezone", "Europe/London")
    start = f"{cfg['time_index']['start_year']}-01-01"
    end = f"{cfg['time_index']['end_year']}-12-31"
    cal_frames = []

    # Annual, Quarterly, Monthly, Weekly
    for freq in cfg["time_index"]["freqs"]:
        idx = pd.date_range(start=start, end=end, tz="UTC", freq=freq)
        df = pd.DataFrame({"timestamp": idx})
        df["year"] = df["timestamp"].dt.year
        df["quarter"] = df["timestamp"].dt.quarter
        df["month"] = df["timestamp"].dt.month
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["freq"] = freq
        cal_frames.append(df)

    cal = pd.concat(cal_frames, ignore_index=True)
    cal = cal.drop_duplicates(subset=["timestamp", "freq"]).reset_index(drop=True)
    return cal


def make_data_dictionary(tables: Dict[str, pd.DataFrame]) -> Dict:
    """
    Construct a column-level data dictionary (type, count, missingness, sample).
    """
    meta = {}
    for name, df in tables.items():
        cols = {}
        for c in df.columns:
            s = df[c]
            cols[c] = {
                "dtype": str(s.dtype),
                "n": int(s.shape[0]),
                "n_null": int(s.isna().sum()),
                "pct_null": float(s.isna().mean()) * 100.0,
                "example": s.dropna().iloc[0] if s.notna().any() else None
            }
        meta[name] = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": cols
        }
    return meta


def quality_checks(df: pd.DataFrame, cfg: Dict, label: str) -> Dict:
    """
    Basic quality checks: null rates, numeric outliers (zscore & IQR), coord sanity.
    """
    zthr = cfg["numeric_outlier"]["zscore_threshold"]
    imult = cfg["numeric_outlier"]["iqr_multiplier"]

    report = {"label": label, "columns": {}}
    for c in df.columns:
        s = df[c]
        colr = {
            "dtype": str(s.dtype),
            "n": int(len(s)),
            "n_null": int(s.isna().sum()),
            "pct_null": float(s.isna().mean() * 100),
            "outliers_z": 0,
            "outliers_iqr": 0
        }
        if s.dtype.kind in "biufc":
            zf = zscore_flags(s, zthr)
            iqf = iqr_flags(s, imult)
            colr["outliers_z"] = int(zf.sum())
            colr["outliers_iqr"] = int(iqf.sum())
        report["columns"][c] = colr

    # Extra for geographic columns
    for la, lo in [("lat", "lon"), ("latitude", "longitude")]:
        if la in df.columns and lo in df.columns:
            lat = df[la]
            lon = df[lo]
            bad_lat = ((lat < -90) | (lat > 90)) & lat.notna()
            bad_lon = ((lon < -180) | (lon > 180)) & lon.notna()
            report["geo_check"] = {
                "bad_lat": int(bad_lat.sum()),
                "bad_lon": int(bad_lon.sum())
            }
            break

    return report


# --------------------------------------------------------------------------------------
# Main Orchestration
# --------------------------------------------------------------------------------------

def run_ingestion(config_path: Optional[str] = None) -> Dict:
    """
    Orchestrates Part 1:
      - loads config & resolves paths
      - loads all raw sources
      - validates schemas
      - generates data dictionary & quality reports
      - writes parquet tables and manifests
    Returns a summary dict with key paths.
    """
    cfg = load_config(Path(config_path) if config_path else None)
    root, fallback, out_root = detect_root_paths(cfg)

    # Resolve file paths
    f = cfg["files"]
    paths = {
        "c3s_wind_ind": smart_path(root, f["c3s_wind_ind"], fallback),
        "era5_tracks": smart_path(root, f["era5_tracks"], fallback),
        "ukc11_risk": smart_path(root, f["ukc11_risk"], fallback),
        "ukc11_losses": smart_path(root, f["ukc11_losses"], fallback),
        "ukc11_tier2": smart_path(root, f["ukc11_tier2"], fallback),
        "curtailment": smart_path(root, f["curtailment"], fallback),
        "embedded_capacity": smart_path(root, f["embedded_capacity"], fallback),
        "ehv_feeders": smart_path(root, f["ehv_feeders"], fallback),
    }

    # Load
    tables: Dict[str, pd.DataFrame] = {}
    prov: Dict[str, Dict] = {"generated_at": datetime.utcnow().isoformat()}

    loaders = {
        "c3s_wind_ind": load_c3s_wind_indicators,
        "era5_tracks": load_era5_tracks,
        "ukc11_risk": lambda p: load_ukc11_csv(p, "risk"),
        "ukc11_losses": lambda p: load_ukc11_csv(p, "losses"),
        "ukc11_tier2": lambda p: load_ukc11_csv(p, "tier2"),
        "curtailment": load_curtailment,
        "embedded_capacity": load_embedded_capacity,
        "ehv_feeders": load_ehv_feeders,
    }

    for key, loader in loaders.items():
        p = paths[key]
        if not p.exists():
            LOGGER.warning("Missing file for %s: %s", key, p)
            continue
        try:
            df = loader(p)
            # Minimal schema check (warn only)
            expected = DEFAULT_CONFIG["schema"].get(key, [])
            validate_minimal_schema(df, expected, key)
            tables[key] = df
            prov[key] = {"path": str(p), **file_digest(p)}
        except Exception as e:
            LOGGER.exception("Failed to load %s from %s: %s", key, p, e)

    # Build calendar index
    cal = add_calendar_index(cfg)
    tables["calendar_index"] = cal

    # Quality reports per table
    qreports = {}
    for key, df in tables.items():
        qreports[key] = quality_checks(df, cfg, key)

    # Data dictionary
    ddict = make_data_dictionary(tables)

    # Write outputs
    outs = cfg["outputs"]
    written = {}
    for key, df in tables.items():
        out_name = outs.get(key)
        if not out_name:
            continue
        out_path = Path(cfg["io"]["output_root"]) / out_name
        try:
            df.to_parquet(out_path, index=False)
            written[key] = str(out_path.resolve())
            LOGGER.info("Wrote %s → %s", key, out_path)
        except Exception as e:
            # Fallback to CSV if parquet engine unavailable
            out_csv = out_path.with_suffix(".csv")
            df.to_csv(out_csv, index=False)
            written[key] = str(out_csv.resolve())
            LOGGER.warning("Parquet failed for %s, wrote CSV instead: %s", key, e)

    # Write dictionary, quality report, provenance
    ddict_path = Path(cfg["io"]["output_root"]) / outs["data_dictionary"]
    with open(ddict_path, "w", encoding="utf-8") as fjson:
        json.dump(ddict, fjson, indent=2, default=str)

    qrep_path = Path(cfg["io"]["output_root"]) / outs["quality_report"]
    with open(qrep_path, "w", encoding="utf-8") as fjson:
        json.dump(qreports, fjson, indent=2, default=str)

    prov_path = Path(cfg["io"]["output_root"]) / outs["provenance"]
    with open(prov_path, "w", encoding="utf-8") as fjson:
        json.dump(prov, fjson, indent=2, default=str)

    # Write calendar
    cal_path = Path(cfg["io"]["output_root"]) / outs["combined_calendar"]
    try:
        cal.to_parquet(cal_path, index=False)
    except Exception:
        cal.to_csv(cal_path.with_suffix(".csv"), index=False)

    summary = {
        "written_tables": written,
        "data_dictionary": str(ddict_path.resolve()),
        "quality_report": str(qrep_path.resolve()),
        "provenance_manifest": str(prov_path.resolve()),
        "calendar_index": str(cal_path.resolve()),
        "geopandas_available": _HAS_GPD
    }
    LOGGER.info("Ingestion done. Tables: %s", list(written.keys()))
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CAMH-DT — Part 1: Data Ingestion, Configuration & Provenance")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config; if omitted, DEFAULT_CONFIG is used.")
    args = parser.parse_args()

    summary = run_ingestion(args.config)
    print(json.dumps(summary, indent=2))
