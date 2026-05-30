"""
Extract coast-down and high-speed corner segments from FastF1 lap telemetry.
"""

import numpy as np
import pandas as pd

# DRS state values > 10 indicate DRS open in FastF1 encoding
DRS_OPEN_THRESHOLD = 10


def drs_is_open(drs_value) -> bool:
    try:
        return int(drs_value) > DRS_OPEN_THRESHOLD
    except (TypeError, ValueError):
        return False


def extract_coastdown_segments(
    lap_telemetry: pd.DataFrame,
    min_duration: float = 1.0,
    min_speed_kmh: float = 100.0,
    throttle_threshold: float = 5.0,
) -> list[pd.DataFrame]:
    """
    Return list of telemetry slices where the car is coasting (throttle below
    threshold, brake=0, speed decreasing) for at least min_duration seconds
    above min_speed_kmh.

    throttle_threshold: treat Throttle < this value as off (default 5%).
    FastF1's Throttle channel can sit at 1-4% during engine-braking lift-off
    rather than going to exactly 0.
    """
    tel = lap_telemetry.copy().reset_index(drop=True)
    tel["t"] = tel["Time"].dt.total_seconds()

    # Normalise brake: FastF1 gives bool in newer versions, 0-100 float in older
    brake_col = tel["Brake"]
    if brake_col.dtype == object:
        tel["_braking"] = brake_col.astype(bool)
    elif brake_col.dtype == bool or str(brake_col.dtype) == 'bool':
        tel["_braking"] = brake_col.astype(bool)
    else:
        tel["_braking"] = brake_col > 0

    tel["_coasting"] = (tel["Throttle"] < throttle_threshold) & (~tel["_braking"])

    segments: list[pd.DataFrame] = []
    seg_start: int | None = None

    for i in range(len(tel)):
        coasting = tel.at[i, "_coasting"]
        if coasting and seg_start is None:
            seg_start = i
        elif not coasting and seg_start is not None:
            seg = tel.iloc[seg_start:i].copy()
            _maybe_add(seg, segments, min_duration, min_speed_kmh)
            seg_start = None

    # Handle segment running to end of lap
    if seg_start is not None:
        seg = tel.iloc[seg_start:].copy()
        _maybe_add(seg, segments, min_duration, min_speed_kmh)

    return segments


def _maybe_add(
    seg: pd.DataFrame,
    segments: list,
    min_duration: float,
    min_speed_kmh: float,
) -> None:
    if len(seg) < 3:
        return
    duration = seg["t"].iloc[-1] - seg["t"].iloc[0]
    speed_ok = seg["Speed"].iloc[0] >= min_speed_kmh
    decelerating = seg["Speed"].iloc[0] > seg["Speed"].iloc[-1]
    if duration >= min_duration and speed_ok and decelerating:
        segments.append(seg)


def segment_drs_state(seg: pd.DataFrame) -> bool:
    """Return True if majority of segment has DRS open."""
    if "DRS" not in seg.columns:
        return False
    open_count = seg["DRS"].apply(drs_is_open).sum()
    return open_count > len(seg) / 2


def extract_corner_samples(
    lap_telemetry: pd.DataFrame,
    min_speed_kmh: float = 144.0,   # 40 m/s
    min_lat_g: float = 3.5,          # g
    min_throttle: float = 80.0,
) -> pd.DataFrame:
    """
    Return rows where the car is at high speed, high lateral load, and on throttle —
    the regime where it is near the tyre friction limit in a fast corner.

    Lateral acceleration is computed from GPS-derived position if a direct channel
    is absent (FastF1 doesn't always expose it directly).
    """
    tel = lap_telemetry.copy().reset_index(drop=True)
    tel["t"] = tel["Time"].dt.total_seconds()
    tel["v_ms"] = tel["Speed"] / 3.6

    lat_col = _find_lat_g_column(tel)
    if lat_col is None:
        tel = _compute_lateral_g_from_position(tel)
        lat_col = "lat_g_computed"

    if lat_col not in tel.columns:
        return pd.DataFrame()

    mask = (
        (tel["Speed"] >= min_speed_kmh)
        & (tel[lat_col].abs() >= min_lat_g)
        & (tel["Throttle"] >= min_throttle)
    )
    samples = tel[mask].copy()
    samples["lat_g"] = samples[lat_col].abs()
    return samples


def _find_lat_g_column(tel: pd.DataFrame) -> str | None:
    """Return the name of a lateral-g column if one exists."""
    candidates = [
        "lateral_acceleration", "LateralAcceleration",
        "lateral_g", "LatG", "lat_g",
        "ay", "Ay",
    ]
    for c in candidates:
        if c in tel.columns:
            return c
    return None


def _compute_lateral_g_from_position(tel: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """
    Estimate lateral acceleration from GPS position rows only.

    Filters to Source=='pos' rows first so all dt values come from actual GPS
    fixes at consistent spacing (~10 Hz), avoiding the blow-up caused by mixing
    car-telemetry rows (240 Hz) with GPS rows.
    """
    if "X" not in tel.columns or "Y" not in tel.columns:
        return tel
    if "Source" not in tel.columns:
        return tel

    tel = tel.copy()

    # Work only on actual GPS position rows, dropping rows that are too
    # close together (<0.1 s apart) — those cause quantisation blow-up
    # when differentiating 1-dm-resolution positions.
    gps = tel[tel["Source"] == "pos"].copy().reset_index(drop=True)
    if len(gps) < 5:
        return tel

    dt_all = gps["t"].diff().fillna(0.24)
    gps = gps[dt_all >= 0.1].copy().reset_index(drop=True)
    if len(gps) < 5:
        return tel

    dt = gps["t"].diff().bfill().clip(lower=0.1)
    w = max(3, smooth_window | 1)

    # Convert decimeter coordinates to metres so units match the speed channel
    x_m = gps["X"] * 0.1
    y_m = gps["Y"] * 0.1

    # Velocity from position differences
    vx = (x_m.diff().fillna(0) / dt).rolling(w, center=True, min_periods=1).mean()
    vy = (y_m.diff().fillna(0) / dt).rolling(w, center=True, min_periods=1).mean()

    # Acceleration from velocity differences
    ax = (vx.diff().fillna(0) / dt).rolling(w, center=True, min_periods=1).mean()
    ay = (vy.diff().fillna(0) / dt).rolling(w, center=True, min_periods=1).mean()

    speed = gps["v_ms"].clip(lower=1.0)
    lat_g = (ax * vy - ay * vx) / (speed * 9.81)

    # Interpolate back onto the full telemetry time index
    tel["lat_g_computed"] = np.interp(
        tel["t"].values, gps["t"].values, lat_g.fillna(0).values
    )
    return tel
