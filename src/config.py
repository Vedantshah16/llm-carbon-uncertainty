"""Run disclosures, priors and region mappings.

Every disclosed number below is copied from the cited source; every prior is an
assumption and is labelled as such. Change priors here, not in model code.
"""
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------
# Plausible grid zones for Meta-trained runs (location undisclosed).
# One entry per long-established US Meta campus -> Electricity Maps zone that
# serves it. Equal weight per campus (so PJM, with two campuses, gets 2/11).
# ----------------------------------------------------------------------------
META_CAMPUS_ZONES = {
    "Prineville, OR": "US-NW-PACW",
    "Forest City, NC": "US-CAR-DUK",
    "Altoona, IA": "US-MIDW-MISO",
    "Fort Worth, TX": "US-TEX-ERCO",
    "Los Lunas, NM": "US-SW-PNM",
    "Papillion, NE": "US-CENT-SWPP",
    "Henrico, VA": "US-MIDA-PJM",
    "New Albany, OH": "US-MIDA-PJM",
    "Eagle Mountain, UT": "US-NW-PACE",
    "Huntsville, AL": "US-TEN-TVA",
    "Stanton Springs, GA": "US-SE-SOCO",
}
META_ZONES = list(META_CAMPUS_ZONES.values())

# Zones used for the general hourly-vs-annual analysis (measured hourly data;
# IN-WE and JP-TK excluded because their hourly values are 100% modelled).
GENERAL_ZONES = ["FR", "FI", "SE", "DE", "GB", "PL", "ES", "AU-NSW", "AU-VIC",
                 "US-CAL-CISO", "US-TEX-ERCO", "US-MIDA-PJM", "US-MIDW-MISO",
                 "US-NW-BPAT", "US-SE-SOCO"]


@dataclass
class Run:
    key: str                      # short id
    label: str                    # figure label
    source: str                   # citation key for disclosed numbers
    accel: str                    # accelerator family -> embodied prior
    tdp_kw: float                 # disclosed per-accelerator power [kW]
    gpu_hours: Optional[float]    # disclosed accelerator-hours [GPU h]; None if energy-measured
    zones: list                   # candidate grid zones (len 1 = stated region)
    window: tuple                 # (earliest start, latest end) UTC, ISO strings
    n_gpu: tuple                  # (min, max) accelerators running concurrently
    pue: tuple                    # (min, max) uniform prior on PUE [-]
    fixed_start: Optional[str] = None   # exact start (UTC) if disclosed
    fixed_hours: Optional[float] = None # exact duration [h] if disclosed
    measured_it_kwh: Optional[float] = None  # disclosed measured IT energy [kWh]
    measured_err: tuple = (0.9, 1.1)    # uniform multiplicative error on measured energy
    # --- what the original paper reported (operational unless noted) ---
    reported_op_t: float = 0.0          # reported operational tCO2e
    reported_total_t: Optional[float] = None  # reported total incl. embodied (if any)
    reported_ci: Optional[float] = None       # CI the paper used [gCO2e/kWh]
    reported_pue: Optional[float] = None
    baseline_year: int = 2022           # year of annual-average CI used by baseline
    notes: str = ""


RUNS = [
    Run(key="bloom", label="BLOOM-176B", source="luccioni2023bloom",
        accel="A100", tdp_kw=0.400, gpu_hours=1_082_990,
        zones=["FR"], window=("2022-03-11T19:42", "2022-07-07T00:00"),
        n_gpu=(384, 384), pue=(1.1, 1.4),
        fixed_start="2022-03-11T19:42", fixed_hours=118 * 24 + 5 + 41 / 60,
        reported_op_t=24.69, reported_total_t=50.5, reported_ci=57, reported_pue=1.2,
        baseline_year=2022,
        notes="Start from BLOOM model card (11 Mar 2022 11:42 PST); duration 118 d 5 h 41 m "
              "and 384 GPUs avg from Luccioni et al. Reported total 50.5 t includes idle 14.6 t "
              "and embodied 11.2 t."),
    Run(key="llama65b", label="LLaMA-65B", source="touvron2023llama",
        accel="A100", tdp_kw=0.400, gpu_hours=1_022_362,
        zones=META_ZONES, window=("2022-10-01T00:00", "2023-02-15T00:00"),
        n_gpu=(2048, 2048), pue=(1.05, 1.20),
        reported_op_t=173, reported_ci=385, reported_pue=1.1, baseline_year=2022,
        notes="2048 A100-80GB, ~21 days (paper). Dates and site undisclosed: window is an "
              "assumption ending before the Feb-2023 release."),
    Run(key="llama2_70b", label="Llama 2 70B", source="touvron2023llama2",
        accel="A100", tdp_kw=0.400, gpu_hours=1_720_320,
        zones=META_ZONES, window=("2023-01-01T00:00", "2023-08-01T00:00"),
        n_gpu=(1024, 2048), pue=(1.05, 1.20),
        reported_op_t=291.42, baseline_year=2023,
        notes="Model card: trained between Jan and Jul 2023. GPU count undisclosed (prior)."),
    Run(key="llama31_405b", label="Llama 3.1 405B", source="meta2024llama31card",
        accel="H100", tdp_kw=0.700, gpu_hours=30_840_000,
        zones=META_ZONES, window=("2024-01-01T00:00", "2024-07-15T00:00"),
        n_gpu=(12_288, 16_384), pue=(1.05, 1.20),
        reported_op_t=8930, baseline_year=2024,
        notes="Location-based figure from model card. Up to 16K H100 (Llama 3 paper). "
              "Window between Dec-2023 data cutoff and 23 Jul 2024 release is an assumption."),
    Run(key="olmo7b", label="OLMo-7B (A100)", source="groeneveld2024olmo",
        accel="A100", tdp_kw=0.400, gpu_hours=None,
        zones=["AU-NSW", "AU-VIC"], window=("2023-04-01T00:00", "2024-01-15T00:00"),
        n_gpu=(216, 216), pue=(1.1, 1.6),
        measured_it_kwh=104_000, measured_err=(0.9, 1.1),
        reported_op_t=70, reported_ci=610, reported_pue=1.1, baseline_year=2023,
        notes="Energy measured (104 MWh IT, before PUE 1.1; 104 x 1.1 x 0.610 = 70 t) on 27 nodes x 8 A100-40GB, trained in "
              "Australia (state undisclosed: NSW/VIC prior). Dates undisclosed: 2023 window assumed."),
]

# ----------------------------------------------------------------------------
# Priors (assumptions). All uniform unless stated.
# ----------------------------------------------------------------------------
PRIORS = {
    # mean accelerator draw as a fraction of TDP during training
    "rho_gpu": (0.60, 1.00),
    # host CPU + DRAM + NIC + switch power per accelerator, as a fraction of TDP
    "eta_host": (0.15, 0.80),
    # embodied carbon per accelerator incl. its share of host server [kgCO2e]
    # log-normal fixed by its 5th / 95th percentiles
    "c_emb_q": {"A100": (150.0, 1500.0), "H100": (200.0, 2000.0)},
    # hardware service life [years]
    "life_years": (3.0, 6.0),
    # lifetime-average fraction of time the hardware does allocable work
    "u_life": (0.60, 1.00),
}

MC_SAMPLES = 100_000
SEEDS = [0, 1, 2]
SOBOL_N = 2 ** 14        # base samples; total evals = N * (d + 2)
