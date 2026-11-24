"""All experiments. Usage:  python src/experiments.py {mc,sobol,hourly,sweep,extras,all}

Outputs go to results/. Every stochastic step is seeded (config.SEEDS).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from config import RUNS, SEEDS, MC_SAMPLES, SOBOL_N, GENERAL_ZONES, PRIORS
from ci_data import CIBank, load_zone, hour_index, LCA_COL, CI_COL
from model import evaluate, baseline, active_inputs

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
ALL_ZONES = sorted({z for r in RUNS for z in r.zones} | set(GENERAL_ZONES) | {"FI"})
Q = dict(q05=0.05, q50=0.50, q95=0.95)


def bank():
    return CIBank(ALL_ZONES)


def qstats(x):
    return {k: float(np.quantile(x, q)) for k, q in Q.items()}


# ----------------------------------------------------------------------------
def run_mc():
    b = bank()
    rows = []
    for run in RUNS:
        d = len(active_inputs(run))
        base = baseline(run, b)
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            out = evaluate(run, rng.random((MC_SAMPLES, d)), b)
            err = out["ci_eff"] / out["ci_annual"] - 1.0
            rec = {"seed": seed}
            for var in ["total", "op", "emb", "emb_share", "ci_eff", "hours"]:
                for k, v in qstats(out[var]).items():
                    rec[f"{var}_{k}"] = v
            for k, v in qstats(err).items():
                rec[f"hourly_err_{k}"] = v
            rec["p_total_gt_reported_op"] = float((out["total"] > run.reported_op_t).mean())
            rec["p_op_gt_reported_op"] = float((out["op"] > run.reported_op_t).mean())
            if run.reported_total_t:
                rec["p_total_gt_reported_total"] = float((out["total"] > run.reported_total_t).mean())
            per_seed.append(rec)
            if seed == SEEDS[0]:
                np.savez_compressed(OUT / f"samples_{run.key}.npz",
                                    **{k: v for k, v in out.items() if k != "zone"},
                                    zone=out["zone"].astype("U16"))
        df = pd.DataFrame(per_seed).drop(columns="seed")
        row = {"run": run.key, "label": run.label, "n_inputs": d,
               "baseline_t": base["total"], "baseline_ci": base["ci_annual"],
               "baseline_pue": base["pue_nom"], "reported_op_t": run.reported_op_t,
               "reported_total_t": run.reported_total_t}
        for c in df.columns:
            row[c] = df[c].mean()
            row[c + "_sd"] = df[c].std(ddof=1)
        lo, hi = row["total_q05"], row["total_q95"]
        row["reported_op_in_total_90"] = bool(lo <= run.reported_op_t <= hi)
        row["reported_op_in_op_90"] = bool(row["op_q05"] <= run.reported_op_t <= row["op_q95"])
        row["baseline_in_total_90"] = bool(lo <= base["total"] <= hi)
        row["median_over_baseline"] = row["total_q50"] / base["total"]
        row["median_over_reported_op"] = row["total_q50"] / run.reported_op_t
        rows.append(row)
        print(f"[mc] {run.label:16s} total median {row['total_q50']:9.1f} t "
              f"90% [{lo:9.1f}, {hi:9.1f}]  baseline {base['total']:8.1f}  reported {run.reported_op_t}"
              f"  emb share {row['emb_share_q50']:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "mc_summary.csv", index=False)

    # convergence check on the run with most inputs
    run = max(RUNS, key=lambda r: len(active_inputs(r)))
    conv = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        U = rng.random((MC_SAMPLES, len(active_inputs(run))))
        y = evaluate(run, U, b)["total"]
        for n in [100, 1_000, 10_000, 100_000]:
            conv.append({"run": run.key, "seed": seed, "n": n,
                         "q50": np.quantile(y[:n], .5), "q05": np.quantile(y[:n], .05),
                         "q95": np.quantile(y[:n], .95)})
    pd.DataFrame(conv).to_csv(OUT / "mc_convergence.csv", index=False)


# ----------------------------------------------------------------------------
def run_sobol():
    b = bank()
    rows = []
    for run in RUNS:
        names = active_inputs(run)
        problem = {"num_vars": len(names), "names": names, "bounds": [[0, 1]] * len(names)}
        for seed in SEEDS:
            U = sobol_sample.sample(problem, SOBOL_N, calc_second_order=False, seed=seed)
            y = evaluate(run, U, b)["total"]
            np.random.seed(seed)  # SALib's bootstrap also draws from the global RNG
            S = sobol_analyze.analyze(problem, y, calc_second_order=False, seed=seed,
                                      print_to_console=False)
            for i, nm in enumerate(names):
                rows.append({"run": run.key, "label": run.label, "seed": seed, "input": nm,
                             "S1": S["S1"][i], "S1_conf": S["S1_conf"][i],
                             "ST": S["ST"][i], "ST_conf": S["ST_conf"][i],
                             "n_evals": len(y)})
        top = (pd.DataFrame([r for r in rows if r["run"] == run.key])
               .groupby("input")["ST"].mean().sort_values(ascending=False))
        print(f"[sobol] {run.label:16s} top ST: " +
              ", ".join(f"{k}={v:.2f}" for k, v in top.head(3).items()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sobol_raw.csv", index=False)
    agg = df.groupby(["run", "label", "input"]).agg(
        S1=("S1", "mean"), S1_sd=("S1", "std"), S1_conf=("S1_conf", "mean"),
        ST=("ST", "mean"), ST_sd=("ST", "std"), ST_conf=("ST_conf", "mean"),
        n_evals=("n_evals", "first")).reset_index()
    agg.to_csv(OUT / "sobol_summary.csv", index=False)


# ----------------------------------------------------------------------------
DURATIONS_H = [1, 6, 24, 72, 168, 336, 720, 1440, 2160, 4380]


def run_hourly(n_starts=2_000):
    """Error of annual-average CI vs the true mean CI over a run window, by duration."""
    b = CIBank(GENERAL_ZONES)
    years = [2021, 2022, 2023, 2024]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for z in GENERAL_ZONES:
            zi = b.pos[z]
            ann = {y: b.annual_mean(z, y) for y in years}
            for h in DURATIONS_H:
                y = rng.choice(years, n_starts)
                a = np.array([hour_index(f"{v}-01-01") for v in y])
                e = np.array([hour_index(f"{v + 1}-01-01") for v in y])
                s = a + rng.random(n_starts) * (e - a - h)
                w = b.window_mean(np.full(n_starts, zi), s, np.full(n_starts, h))
                err = w / np.array([ann[v] for v in y]) - 1.0
                rows.append({"seed": seed, "zone": z, "hours": h,
                             "abs_err_q50": np.quantile(np.abs(err), .5),
                             "abs_err_q90": np.quantile(np.abs(err), .9),
                             "err_q05": np.quantile(err, .05), "err_q95": np.quantile(err, .95)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hourly_vs_annual_raw.csv", index=False)
    agg = df.groupby(["zone", "hours"]).agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.drop(columns=["seed_mean", "seed_std"]).reset_index().to_csv(
        OUT / "hourly_vs_annual.csv", index=False)
    print("[hourly] median |err| at 720 h:",
          df[df.hours == 720].groupby("zone")["abs_err_q50"].mean().round(3).to_dict())


# ----------------------------------------------------------------------------
def run_zonesweep():
    """Place one fixed job (Llama 2 70B disclosure: 1.72M A100-h, 2048 GPUs) in each
    zone during 2023 and ask: how large is the embodied share, and which input
    dominates the variance, as a function of the grid's annual-average CI?"""
    from dataclasses import replace
    b = CIBank(GENERAL_ZONES)
    tmpl = next(r for r in RUNS if r.key == "llama2_70b")
    rows = []
    for z in GENERAL_ZONES:
        run = replace(tmpl, key=f"sweep_{z}", label=z, zones=[z], n_gpu=(2048, 2048),
                      window=("2023-01-01T00:00", "2024-01-01T00:00"), pue=(1.05, 1.20),
                      baseline_year=2023)
        names = active_inputs(run)
        problem = {"num_vars": len(names), "names": names, "bounds": [[0, 1]] * len(names)}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            out = evaluate(run, rng.random((MC_SAMPLES, len(names))), b)
            U = sobol_sample.sample(problem, 2 ** 12, calc_second_order=False, seed=seed)
            np.random.seed(seed)
            S = sobol_analyze.analyze(problem, evaluate(run, U, b)["total"],
                                      calc_second_order=False, seed=seed)
            st = dict(zip(names, S["ST"]))
            rows.append({"zone": z, "seed": seed, "ci_annual_2023": b.annual_mean(z, 2023),
                         **{f"emb_share_{k}": v for k, v in qstats(out["emb_share"]).items()},
                         "total_q50": np.quantile(out["total"], .5),
                         "ST_c_emb": st["c_emb"], "ST_start": st["start"],
                         "ST_power": st["rho_gpu"] + st["eta_host"], "ST_pue": st["pue"],
                         "ST_life_u": st["life"] + st["u_life"]})
    df = pd.DataFrame(rows)
    agg = df.drop(columns="seed").groupby("zone").agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("ci_annual_2023_mean")
    agg.to_csv(OUT / "zone_sweep.csv", index=False)
    print("[sweep]", agg[["zone", "ci_annual_2023_mean", "emb_share_q50_mean",
                          "ST_c_emb_mean", "ST_power_mean"]].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
def run_extras():
    """Small deterministic checks quoted in the paper."""
    b = bank()
    bloom = next(r for r in RUNS if r.key == "bloom")
    s = hour_index(bloom.fixed_start)
    ci_window = b.window_mean([b.pos["FR"]], [s], [bloom.fixed_hours])[0]
    g_tdp = bloom.gpu_hours * bloom.tdp_kw
    ex = {
        "bloom_ci_window_g_per_kwh": ci_window,
        "bloom_ci_annual_2022": b.annual_mean("FR", 2022),
        "fr2022_hourly_min_max": [float(b.series[b.pos["FR"], hour_index("2022-01-01"):hour_index("2023-01-01")].min()),
                                  float(b.series[b.pos["FR"], hour_index("2022-01-01"):hour_index("2023-01-01")].max())],
        "bloom_ci_used_in_paper": bloom.reported_ci,
        # validation of the power prior against BLOOM's measured energy (Luccioni et al.)
        "bloom_measured_dynamic_over_tdp": 433_196 / g_tdp,
        "bloom_measured_dynamic_plus_idle_over_tdp": (433_196 + 256_646) / g_tdp,
        "prior_rho_plus_eta_q05_q95": list(np.quantile(
            np.random.default_rng(0).uniform(*PRIORS["rho_gpu"], 10 ** 6) +
            np.random.default_rng(1).uniform(*PRIORS["eta_host"], 10 ** 6), [.05, .95])),
        # OLMo-7B on LUMI reported 0 t (market-based); location-based with FI 2023 grid:
        "olmo_lumi_fi2023_annual_ci": b.annual_mean("FI", 2023),
        "olmo_lumi_location_based_t": 135_000 * 1.1 * b.annual_mean("FI", 2023) * 1e-6,
        "meta_zone_annual_ci": {f"{z}_{y}": b.annual_mean(z, y)
                                for z in sorted(set(RUNS[1].zones)) for y in (2022, 2023, 2024)},
        "interpolated_hours": {z: int(np.isnan(pd.read_csv(
            Path(__file__).resolve().parents[1] / "data" / "raw" / f"{z}.csv")[CI_COL]).sum())
            for z in ALL_ZONES},
    }
    ex["meta_sites_equal_weight_ci_2022"] = float(np.mean([b.annual_mean(z, 2022) for z in RUNS[1].zones]))
    # validation: eq. (5) with BLOOM's own assumptions (150 kg/GPU, 6 yr, 85 %) vs paper's 3.64 t
    ex["check_eq5_bloom_gpu_embodied_t"] = bloom.gpu_hours * 150 / (6 * 8760 * 0.85) / 1000
    # reproducibility of reported figures from their own stated inputs
    ex["check_llama65b_t"] = 1_022_362 * 0.4 * 1.1 * 0.385 / 1000          # paper: 173
    ex["check_olmo_a100_t"] = 104 * 1.1 * 0.610                             # paper: 70
    ex["implied_ci_llama2_70b"] = 291.42e6 / (1_720_320 * 400) * 1000       # g/kWh
    ex["implied_ci_llama31_405b"] = 8930e6 / (30.84e6 * 700) * 1000         # g/kWh
    # direct vs life-cycle CI ratio for each run's zones (scope sensitivity)
    lca = {}
    for r in RUNS:
        num = np.mean([load_zone(z, LCA_COL)[hour_index(f"{r.baseline_year}-01-01"):
                                            hour_index(f"{r.baseline_year + 1}-01-01")].mean()
                       for z in r.zones])
        den = np.mean([b.annual_mean(z, r.baseline_year) for z in r.zones])
        lca[r.key] = num / den
    ex["lca_over_direct_ci_ratio"] = lca
    (OUT / "extras.json").write_text(json.dumps(ex, indent=2, default=float), encoding="utf-8")
    print("[extras]", json.dumps({k: v for k, v in ex.items()
                                  if not isinstance(v, dict)}, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["mc", "sobol", "hourly", "sweep", "extras", "all"])
    a = ap.parse_args()
    t0 = time.time()
    if a.what in ("extras", "all"): run_extras()
    if a.what in ("mc", "all"): run_mc()
    if a.what in ("sobol", "all"): run_sobol()
    if a.what in ("hourly", "all"): run_hourly()
    if a.what in ("sweep", "all"): run_zonesweep()
    print(f"done in {time.time() - t0:.1f} s")
