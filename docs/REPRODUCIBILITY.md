# Reproducibility sheet — NGC End Term, Topic #57
**Vedant Shah · Application ID 2308624**
How Uncertain Is the Carbon Footprint of LLM Training?

## Tool versions
| Tool | Version |
|---|---|
| Python | 3.14.6 (laptop); 3.13.18 (second machine) |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.2 |
| SALib | 1.6.0 |
| Data | Electricity Maps hourly CI 2021–2024 via CarbonX @ `542536f4` |

## Random seeds
`SEEDS = [0, 1, 2]` (src/config.py), used for Monte Carlo (`numpy.random.default_rng(seed)`),
the Sobol design (`SALib.sample.sobol.sample(..., seed=seed)`), the Sobol bootstrap
(`np.random.seed(seed)` + `analyze(..., seed=seed)`) and hourly-window sampling.
MC convergence check uses seeds 100–102. Two clean end-to-end runs produced bit-identical outputs,
and the laptop run reproduced every Table II median exactly.

## Hardware and wall-clock runtime
| Machine | CPU | RAM | `experiments.py all` | `figures.py` | `report.py` |
|---|---|---|---|---|---|
| Author's laptop (ASUS ROG Strix G16, Windows) | Intel Core i7-14650HX, 16 cores | 16 GB | 49.9 s | 4.8 s | 0.7 s |
| Second machine (Linux) | 2 vCPU Intel Xeon @ 2.30 GHz | 7 GB | 32 s | 2.3 s | < 1 s |

No GPU required. Peak memory < 1 GB.

## Configuration
- Monte Carlo: 100,000 samples per run per seed; 3 seeds; reported values are means across seeds, spread = SD across seeds.
- Sobol: Sobol' design, N = 16,384 base samples, first + total order (131,072–180,224 evaluations per run per seed).
- Zone sweep: 15 zones, 100,000 MC samples + Sobol N = 4,096 per zone per seed.
- Hourly-vs-annual: 2,000 random windows per (zone, duration, seed), 15 zones × 10 durations × 3 seeds.

## Commands that regenerate each figure
```
python src/experiments.py all
python src/figures.py
python src/report.py
```
| Figure | Data it reads | Producer |
|---|---|---|
| Fig. 1 `fig1_bloom_window` | data/raw/FR.csv | `figures.py` |
| Fig. 2 `fig2_distributions` | results/samples_*.npz, mc_summary.csv | `experiments.py mc` → `figures.py` |
| Fig. 3 `fig3_sobol` | results/sobol_summary.csv | `experiments.py sobol` → `figures.py` |
| Fig. 4 `fig4_zone_sweep` | results/zone_sweep.csv | `experiments.py sweep` → `figures.py` |
| Fig. 5 `fig5_hourly_vs_annual` | results/hourly_vs_annual.csv, mc_summary.csv | `experiments.py hourly` + `mc` → `figures.py` |
| Tables II–III, all numbers | results/*.csv, extras.json | `report.py` → results/RESULTS.md |
