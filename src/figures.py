"""Regenerate every figure from results/.  Usage: python src/figures.py
Figures are sized for one IEEE column (3.5 in) and saved as PDF + 300-dpi PNG.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from config import RUNS
from ci_data import CIBank, AXIS, hour_index

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#1f1f1e", "#5c5b55", "#d9d8d2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
W1 = 3.5  # IEEE single-column width [in]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def box_from_samples(ax, y, x, color, width=0.55):
    q05, q25, q50, q75, q95 = np.quantile(x, [.05, .25, .5, .75, .95])
    ax.plot([q05, q95], [y, y], color=color, lw=1.2, solid_capstyle="round", zorder=2)
    ax.add_patch(plt.Rectangle((q25, y - width / 2), q75 - q25, width, facecolor=color,
                               alpha=0.30, edgecolor=color, lw=1.0, zorder=2))
    ax.plot([q50, q50], [y - width / 2, y + width / 2], color=color, lw=2.0, zorder=3)


# ----------------------------------------------------------------------------
def fig_bloom_window():
    """Fig. 1: hourly French CI in 2022, BLOOM's training window and the annual mean."""
    b = CIBank(["FR"])
    a, e = hour_index("2022-01-01"), hour_index("2023-01-01")
    ci = b.series[0, a:e]
    t = AXIS[a:e]
    run = next(r for r in RUNS if r.key == "bloom")
    s = hour_index(run.fixed_start) - a
    h = int(round(run.fixed_hours))
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.axvspan(t[s], t[s + h - 1], color=BLUE, alpha=0.10, lw=0)
    ax.plot(t, ci, color=INK2, lw=0.4, label="Hourly CI (Electricity Maps)")
    ax.axhline(ci.mean(), color=ORANGE, lw=1.5, ls="--",
               label=f"2022 annual mean = {ci.mean():.1f}")
    w = ci[s:s + h].mean()
    ax.plot([t[s], t[s + h - 1]], [w, w], color=BLUE, lw=2.0,
            label=f"BLOOM-window mean = {w:.1f}")
    ax.axhline(run.reported_ci, color=AQUA, lw=1.2, ls=":",
               label=f"Value used by Luccioni et al. = {run.reported_ci}")
    ax.set_ylabel("Carbon intensity (gCO$_2$e/kWh)")
    ax.set_xlabel("Date (2022, UTC)")
    ax.set_ylim(0, ci.max() * 1.05)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), frameon=False, ncol=2,
              handlelength=1.6, columnspacing=0.8, fontsize=6.3)
    ax.text(t[s + h // 2], 3, "BLOOM training window", ha="center", va="bottom",
            color=BLUE, fontsize=6.5)
    save(fig, "fig1_bloom_window")


# ----------------------------------------------------------------------------
def fig_distributions():
    """Fig. 2: (a) total CO2e relative to each paper's reported figure; (b) embodied share."""
    mc = pd.read_csv(RES / "mc_summary.csv").set_index("run")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 4.0), gridspec_kw={"height_ratios": [1.25, 1]})
    labels = []
    for i, run in enumerate(RUNS[::-1]):
        smp = np.load(RES / f"samples_{run.key}.npz")
        rep = run.reported_op_t
        box_from_samples(a1, i, smp["total"] / rep, BLUE)
        a1.plot(mc.loc[run.key, "baseline_t"] / rep, i, marker="D", ms=5, color=ORANGE,
                mec="white", mew=0.8, zorder=4, ls="none")
        if run.reported_total_t:
            a1.plot(run.reported_total_t / rep, i, marker="^", ms=6, color=AQUA,
                    mec="white", mew=0.8, zorder=4, ls="none")
        box_from_samples(a2, i, 100 * smp["emb_share"], BLUE)
        labels.append(run.label)
    a1.axvline(1.0, color=INK, lw=1.0, ls="--", zorder=1)
    a1.set_xscale("log")
    a1.set_xticks([0.5, 0.7, 1, 1.5, 2, 3, 4])
    a1.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    a1.set_xlabel("Total CO$_2$e / operational figure reported by the paper (–)")
    a1.set_title("(a)", loc="left")
    handles = [Patch(facecolor=BLUE, alpha=0.3, edgecolor=BLUE, label="This work: IQR, 5–95 %, median"),
               Line2D([], [], marker="D", color=ORANGE, ls="none", mec="white", label="Baseline (annual CI, op. only)"),
               Line2D([], [], marker="^", color=AQUA, ls="none", mec="white", label="Paper's own total incl. embodied"),
               Line2D([], [], color=INK, ls="--", lw=1, label="Paper's reported figure (=1)")]
    a1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.08), frameon=False,
              ncol=2, fontsize=6.0, columnspacing=0.8, handlelength=1.5)
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("Embodied share of total CO$_2$e (%)")
    a2.set_xlim(0, 70)
    a2.set_title("(b)", loc="left")
    for ax in (a1, a2):
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig2_distributions")


# ----------------------------------------------------------------------------
NICE = {"zone": "Grid region", "start": "Start time", "n_gpu": "No. of GPUs",
        "meas_err": "Energy meas. error", "rho_gpu": "GPU draw / TDP", "eta_host": "Host power",
        "pue": "PUE", "c_emb": "Embodied / GPU", "life": "Lifetime", "u_life": "Life utilisation"}
ORDER = ["zone", "start", "n_gpu", "rho_gpu", "eta_host", "meas_err", "pue", "c_emb", "life", "u_life"]


def fig_sobol():
    """Fig. 3: total-order Sobol indices (runs x inputs)."""
    s = pd.read_csv(RES / "sobol_summary.csv")
    runs = [r.key for r in RUNS]
    M = np.full((len(ORDER), len(runs)), np.nan)
    for j, rk in enumerate(runs):
        for i, inp in enumerate(ORDER):
            v = s[(s.run == rk) & (s.input == inp)]["ST"]
            if len(v):
                M[i, j] = max(v.iloc[0], 0)
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#1c5cab", "#0d366b"])
    cmap.set_bad("#efefec")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "fixed", ha="center", va="center", fontsize=6, color="#9a998f")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.45 else INK)
    ax.set_xticks(range(len(runs)))
    short = {"bloom": "BLOOM\n176B", "llama65b": "LLaMA\n65B", "llama2_70b": "Llama 2\n70B",
             "llama31_405b": "Llama 3.1\n405B", "olmo7b": "OLMo\n7B"}
    ax.set_xticklabels([short[r.key] for r in RUNS], fontsize=6.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([NICE[o] for o in ORDER])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Total-order Sobol index $S_{T}$ (–)")
    cb.outline.set_visible(False)
    save(fig, "fig3_sobol")


# ----------------------------------------------------------------------------
def fig_sweep():
    """Fig. 4: same job placed in 15 grids: embodied share and which input dominates."""
    z = pd.read_csv(RES / "zone_sweep.csv").sort_values("ci_annual_2023_mean")
    x = z["ci_annual_2023_mean"].to_numpy()
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    med = 100 * z["emb_share_q50_mean"]
    lo = med - 100 * z["emb_share_q05_mean"]
    hi = 100 * z["emb_share_q95_mean"] - med
    ax.errorbar(x, med, yerr=[lo, hi], fmt="o", ms=4, color=BLUE, ecolor=BLUE, elinewidth=0.9,
                capsize=0, label="Embodied share (median, 5–95 %)", zorder=3)
    ax.plot(x, 100 * z["ST_c_emb_mean"], "-s", color=ORANGE, ms=3.5, lw=1.2,
            label="$S_T$ of embodied/GPU (×100)")
    ax.plot(x, 100 * z["ST_power_mean"], "--^", color=INK2, ms=3.5, lw=1.0,
            label="$S_T$ of power inputs (×100)")
    ax.set_xscale("log")
    ax.set_xlabel("Grid annual-average carbon intensity, 2023 (gCO$_2$e/kWh, log)")
    ax.set_ylabel("Share or index (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks([3, 10, 30, 100, 300, 700])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    offs = {"SE": (5, -2), "FR": (5, -3), "FI": (5, -3), "GB": (4, 4), "US-CAL-CISO": (4, 6),
            "PL": (-4, 10)}
    for zn, xi, yi in zip(z["zone"], x, med):
        if zn in offs:
            ax.annotate(zn.replace("US-CAL-", ""), (xi, yi), xytext=offs[zn],
                        textcoords="offset points", fontsize=6, color=INK2,
                        ha="right" if zn == "PL" else "left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=2,
              fontsize=6.3, columnspacing=0.8)
    save(fig, "fig4_zone_sweep")


# ----------------------------------------------------------------------------
def fig_hourly():
    """Fig. 5: error of annual-average CI vs run duration across 15 grids."""
    h = pd.read_csv(RES / "hourly_vs_annual.csv")
    piv = h.pivot(index="hours", columns="zone", values="abs_err_q50_mean") * 100
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for zn in piv.columns:
        hl = zn in ("FR", "PL")
        ax.plot(piv.index, piv[zn], color=BLUE if zn == "FR" else (ORANGE if zn == "PL" else GRID),
                lw=1.6 if hl else 0.8, zorder=3 if hl else 1)
        if hl:
            ax.annotate(zn, (piv.index[-1], piv[zn].iloc[-1]), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, va="center",
                        color=BLUE if zn == "FR" else ORANGE)
    medall = piv.median(axis=1)
    ax.plot(piv.index, medall, color=INK, lw=1.6, ls="--", zorder=4, label="Median of 15 grids")
    mc = pd.read_csv(RES / "mc_summary.csv").set_index("run")
    bl = mc.loc["bloom"]
    ax.plot(bl["hours_q50"], abs(bl["hourly_err_q50"]) * 100, marker="*", ms=9, color=BLUE,
            mec="white", ls="none", zorder=5, label="BLOOM (actual window)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 24, 168, 720, 4380])
    ax.set_xticklabels(["1 h", "1 d", "1 wk", "1 mo", "6 mo"])
    ax.set_xlabel("Training-run duration (h, log)")
    ax.set_ylabel("Median |CI$_{window}$/CI$_{annual}$ − 1| (%)")
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(handles=[Line2D([], [], color=GRID, lw=0.8, label="Individual grid"),
                       *ax.get_legend_handles_labels()[0]], loc="lower left", frameon=False)
    save(fig, "fig5_hourly_vs_annual")


if __name__ == "__main__":
    missing = [r.key for r in RUNS if not (RES / f"samples_{r.key}.npz").exists()]
    if missing or not (RES / "sobol_summary.csv").exists():
        raise SystemExit("Experiment outputs not found. Run  python src/experiments.py all  first.")
    fig_bloom_window()
    fig_distributions()
    fig_sobol()
    fig_sweep()
    fig_hourly()
