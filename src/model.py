"""Embodied-plus-operational carbon model for one training run.

Uncertain inputs are drawn as columns of a unit hypercube U (n x d) and mapped to
physical values by inverse CDFs, so the same function serves plain Monte Carlo
(U ~ iid uniform) and Sobol analysis (U from a Sobol/Saltelli design).

Equations (numbering matches the paper's Section III):
  (1) t      = G / N                          run duration [h]
  (2) E_IT   = G * P_TDP * (rho + eta)        IT energy [kWh]   (TDP-disclosed runs)
      E_IT   = E_meas * m                     IT energy [kWh]   (energy-measured runs)
  (3) CI_eff = (1/t) * sum_{h=s}^{s+t-1} CI_z(h)    effective intensity [g/kWh]
  (4) C_op   = E_IT * PUE * CI_eff * 1e-6     operational [tCO2e]
  (5) C_emb  = G * c_emb / (L * 8760 * u) * 1e-3   embodied share [tCO2e]
  (6) C_tot  = C_op + C_emb
Baseline (standard practice): C_base = G * P_TDP * PUE_nom * CI_annual * 1e-6, no embodied.
"""
import numpy as np
from scipy import stats
from config import PRIORS
from ci_data import CIBank, hour_index


def lognormal_from_q(q05, q95):
    mu = 0.5 * (np.log(q05) + np.log(q95))
    sigma = (np.log(q95) - np.log(q05)) / (2 * stats.norm.ppf(0.95))
    return mu, sigma


def active_inputs(run):
    names = []
    if len(run.zones) > 1:
        names.append("zone")
    if run.fixed_start is None:
        names.append("start")
    if run.n_gpu[0] < run.n_gpu[1]:
        names.append("n_gpu")
    if run.measured_it_kwh is not None:
        names.append("meas_err")
    names += ["rho_gpu", "eta_host", "pue", "c_emb", "life", "u_life"]
    return names


def _u(a, lo, hi):
    return lo + a * (hi - lo)


def evaluate(run, U, bank: CIBank):
    """Map unit-cube samples to outputs. Returns dict of arrays (length n)."""
    names = active_inputs(run)
    col = {k: U[:, i] for i, k in enumerate(names)}
    n = U.shape[0]

    # --- region ---
    if "zone" in col:
        k = np.minimum((col["zone"] * len(run.zones)).astype(int), len(run.zones) - 1)
        zone_names = np.array(run.zones)[k]
    else:
        zone_names = np.array([run.zones[0]] * n)
    zi = np.array([bank.pos[z] for z in zone_names])

    # --- power ---
    rho = _u(col["rho_gpu"], *PRIORS["rho_gpu"])
    eta = _u(col["eta_host"], *PRIORS["eta_host"])
    p_acc = run.tdp_kw * (rho + eta)                     # kW per accelerator incl. host

    # --- scale and duration (eq. 1) ---
    n_gpu = _u(col["n_gpu"], *run.n_gpu) if "n_gpu" in col else np.full(n, float(run.n_gpu[0]))
    if run.measured_it_kwh is not None:
        m = _u(col["meas_err"], *run.measured_err)
        e_it = run.measured_it_kwh * m                    # eq. 2b
        hours = e_it / (n_gpu * p_acc)
        gpu_h = n_gpu * hours
    else:
        gpu_h = np.full(n, float(run.gpu_hours))
        hours = np.full(n, run.fixed_hours) if run.fixed_hours else gpu_h / n_gpu
        e_it = gpu_h * p_acc                              # eq. 2a

    # --- timing and effective intensity (eq. 3) ---
    w0, w1 = hour_index(run.window[0]), hour_index(run.window[1])
    if run.fixed_start is not None:
        start = np.full(n, float(hour_index(run.fixed_start)))
    else:
        slack = np.maximum(w1 - hours - w0, 0.0)
        start = w0 + col["start"] * slack
    ci_eff = bank.window_mean(zi, start, hours)

    # --- operational (eq. 4) ---
    pue = _u(col["pue"], *run.pue)
    c_op = e_it * pue * ci_eff * 1e-6

    # --- embodied (eq. 5) ---
    mu, sg = lognormal_from_q(*PRIORS["c_emb_q"][run.accel])
    c_emb_unit = np.exp(mu + sg * stats.norm.ppf(np.clip(col["c_emb"], 1e-12, 1 - 1e-12)))
    life = _u(col["life"], *PRIORS["life_years"])
    u_life = _u(col["u_life"], *PRIORS["u_life"])
    c_emb = gpu_h * c_emb_unit / (life * 8760.0 * u_life) * 1e-3

    total = c_op + c_emb                                  # eq. 6
    ci_ann = np.array([bank.annual_mean(z, run.baseline_year) for z in run.zones])
    ci_ann_s = ci_ann[[run.zones.index(z) for z in zone_names]] if len(run.zones) > 1 \
        else np.full(n, ci_ann[0])
    return dict(total=total, op=c_op, emb=c_emb, emb_share=c_emb / total,
                ci_eff=ci_eff, ci_annual=ci_ann_s, hours=hours, gpu_h=gpu_h,
                e_it=e_it, pue=pue, zone=zone_names)


def baseline(run, bank: CIBank):
    """Standard practice: operational only, nominal PUE, TDP power, annual-average CI."""
    ci_ann = np.mean([bank.annual_mean(z, run.baseline_year) for z in run.zones])
    pue_nom = run.reported_pue or 1.1
    e = run.measured_it_kwh if run.measured_it_kwh is not None else run.gpu_hours * run.tdp_kw
    return dict(ci_annual=ci_ann, pue_nom=pue_nom, e_it=e, total=e * pue_nom * ci_ann * 1e-6)
