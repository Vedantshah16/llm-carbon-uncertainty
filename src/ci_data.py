"""Hourly grid carbon-intensity data (Electricity Maps, via the CarbonX repository).

All series are put on one common hourly UTC axis (2021-01-01 00:00 .. 2024-12-31 23:00).
Missing CI values (a few hours in two zones) are filled by linear interpolation in time.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
T0 = pd.Timestamp("2021-01-01 00:00")
AXIS = pd.date_range(T0, "2024-12-31 23:00", freq="h")
CI_COL = "Carbon intensity gCO₂eq/kWh (direct)"
LCA_COL = "Carbon intensity gCO₂eq/kWh (Life cycle)"


def hour_index(ts) -> int:
    """UTC timestamp (str or Timestamp) -> integer hour on the common axis."""
    return int((pd.Timestamp(ts) - T0) / pd.Timedelta(hours=1))


@lru_cache(maxsize=None)
def load_zone(zone: str, col: str = CI_COL) -> np.ndarray:
    d = pd.read_csv(RAW / f"{zone}.csv")
    s = pd.Series(d[col].to_numpy(float), index=pd.to_datetime(d.iloc[:, 0]))
    s = s[~s.index.duplicated()].reindex(AXIS)
    s = s.interpolate(method="time", limit_direction="both")
    assert np.isfinite(s.to_numpy()).all(), zone
    return s.to_numpy()


class CIBank:
    """Stack of zones with prefix sums for O(1) window means."""

    def __init__(self, zones, col: str = CI_COL):
        self.zones = list(dict.fromkeys(zones))
        self.pos = {z: i for i, z in enumerate(self.zones)}
        self.series = np.vstack([load_zone(z, col) for z in self.zones])
        self.csum = np.concatenate([np.zeros((len(self.zones), 1)),
                                    np.cumsum(self.series, axis=1)], axis=1)

    def window_mean(self, zi, start, hours):
        """Mean CI over [start, start+hours) for zone indices zi (arrays allowed)."""
        zi = np.asarray(zi, int)
        s = np.asarray(np.floor(start), int)
        h = np.maximum(np.asarray(np.rint(hours), int), 1)
        e = s + h
        if (s < 0).any() or (e > self.series.shape[1]).any():
            raise ValueError("window outside data range")
        return (self.csum[zi, e] - self.csum[zi, s]) / h

    def annual_mean(self, zone, year):
        i = self.pos[zone]
        a, b = hour_index(f"{year}-01-01"), hour_index(f"{year + 1}-01-01")
        return self.series[i, a:b].mean()
