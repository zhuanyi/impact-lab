"""Validation report: rebuild every chart and summary.csv from a fixed seed.

python -m impact_lab.report [--out reports] [--seed N] [--quick]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from impact_lab.capacity import (
    CapacityParams,
    capacity_band,
    capacity_curve,
    capacity_points,
    make_universe,
)
from impact_lab.experiments import (
    DEFAULT_SEED,
    alpha_bias,
    default_laws,
    fit_from_row,
    misspecification,
    out_of_sample,
    seed_spread,
)
from impact_lab.fit import binned_means
from impact_lab.generate import generate_metaorders
from impact_lab.laws import SqrtLaw, local_elasticity
from impact_lab.optimizer import OptimizerParams, reoptimised_capacity

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator  # noqa: E402

LAW_TITLES = {"sqrt": "Square root (truth = fit form)",
              "crossover_paper": "Crossover, η* = 1e-3 (paper)",
              "crossover_stress": "Crossover, η* = 5e-2 (our stress choice)",
              "log": "Logarithmic (Zarinelli et al. shape)"}
PHI_LINE = np.logspace(-4, 0, 200)
AUM_GRID = np.logspace(8, 11, 61)


def _rows(chart: str, metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({"chart": chart, "metric": list(metrics), "value": list(metrics.values())})


def _bn(x: float) -> str:
    return "none below \\$10tn" if np.isnan(x) else f"\\${x / 1e9:.1f}bn"


def _band(fit, rng, phi, k=500, q=(0.05, 0.95)):
    d = fit.draws(rng, k)
    lines = d[:, :1] * phi[None, :] ** d[:, 1:]
    return np.quantile(lines, q, axis=0)


# --- (a) misspecification ------------------------------------------------------------------

def chart_misspecification(seed: int, out_path, n: int = 200_000) -> pd.DataFrame:
    fits, curves = misspecification(seed, n)
    laws = default_laws()
    rng = np.random.default_rng(seed + 1)
    fig = plt.figure(figsize=(19, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.1], hspace=0.08, wspace=0.28)
    metrics = {}
    for j, (name, law) in enumerate(laws.items()):
        f_row = fits[fits["law"] == name].iloc[0]
        fit = fit_from_row(f_row)
        c = curves[curves["law"] == name].sort_values("phi")
        ins = c[c["in_sample"]]
        ax = fig.add_subplot(gs[0, j])
        strip = fig.add_subplot(gs[1, j], sharex=ax)
        for a in (ax, strip):
            a.axvspan(f_row["phi_min"], f_row["phi_max"], color="0.92", zorder=0)
            a.set_xscale("log")
        ok = ins["z_mean"] > 0
        ax.errorbar(ins.loc[ok, "phi"], ins.loc[ok, "z_mean"], yerr=2 * ins.loc[ok, "z_se"],
                    fmt="o", ms=3.5, color="tab:blue", label="binned mean of z ± 2 SE")
        ax.plot(c["phi"], c["true"], "k-", lw=1.8, label="true law")
        lo, hi = _band(fit, rng, PHI_LINE)
        ax.fill_between(PHI_LINE, lo, hi, color="tab:red", alpha=0.18, lw=0)
        ax.plot(PHI_LINE, fit.predict(PHI_LINE), "-", color="tab:red", lw=1.5,
                label="square-root fit (90% band)")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.0)
        ax.set_title(LAW_TITLES[name], fontsize=11)
        ax.tick_params(labelbottom=False)
        se_Y, se_d = fit.se()
        txt = f"Ŷ = {fit.Y:.3f} ± {se_Y:.3f}\nδ̂ = {fit.delta:.3f} ± {se_d:.3f}"
        if name == "sqrt":
            txt += f"\ntrue Y = {law.Y}, δ = {law.delta}"
        else:
            el = local_elasticity(law, np.array([1e-3, 1e-2, 1e-1]))
            txt += "\ntrue local δ (T = 0.25) at φ=1e-3/1e-2/1e-1:\n" + " / ".join(
                f"{e:.2f}" for e in el)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
        if j == 0:
            ax.set_ylabel("impact / daily σ")
            strip.set_ylabel("fitted / true")
            ax.legend(loc="lower right", fontsize=8.5)
        strip.axhline(1.0, color="k", lw=0.8)
        strip.plot(c["phi"], c["ratio"], "-o", color="tab:red", ms=3)
        strip.set_yscale("log")
        strip.set_ylim(0.5, max(2.0, 1.2 * c["ratio"].max()))
        strip.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
        strip.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        strip.yaxis.set_minor_locator(NullLocator())
        strip.set_xlabel("metaorder size φ = Q / daily volume")
        for q in (1e-3, 1e-1, 0.5):
            metrics[f"{name}_ratio_at_{q:g}"] = float(f_row[f"ratio_at_{q:g}"])
        metrics[f"{name}_delta_hat"] = fit.delta
    log_r = fits.set_index("law").loc["log"]
    fig.suptitle(
        "(a) A square-root fit is right where the data are and wrong where you extrapolate: "
        f"on a log-law truth it over-prices 0.1%-ADV orders by "
        f"{(log_r['ratio_at_0.001'] - 1) * 100:.0f}% and 50%-ADV orders by "
        f"{(log_r['ratio_at_0.5'] - 1) * 100:.0f}%", fontsize=12.5)
    fig.text(0.5, -0.01,
             "Shaded: sampled range of φ (flow orders, 100k). Inside it the true line is the "
             "binned mean of true impact; outside it the law is evaluated at T = 0.25 of the "
             "day's volume, which is why the crossover curves kink at the edge.",
             ha="center", fontsize=9.5)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return _rows("a", metrics)


# --- (b) alpha bias ------------------------------------------------------------------------

def chart_alpha_bias(seed: int, out_path, n: int = 200_000) -> pd.DataFrame:
    ab = alpha_bias(seed, n).set_index("subset")
    df = generate_metaorders(np.random.default_rng(seed), n, SqrtLaw())
    z = (df["ret"] / df["sigma_d"]).to_numpy()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    for lab, m, col in (("alpha orders", ~df["flow"], "tab:orange"),
                        ("flow orders", df["flow"], "tab:blue")):
        b = binned_means(df.loc[m, "phi"], z[m.to_numpy()])
        ok = b["z_mean"] > 0
        ax.errorbar(b.loc[ok, "phi_mid"], b.loc[ok, "z_mean"], yerr=2 * b.loc[ok, "z_se"],
                    fmt="o", ms=3.5, color=col, alpha=0.8, label=f"{lab}: binned z ± 2 SE")
    phi = np.logspace(-3, np.log10(0.2), 100)
    ax.plot(phi, SqrtLaw().impact(phi, phi), "k-", lw=2, label="true law (Y = δ = 0.5)")
    styles = {"all": ("0.45", "--"), "alpha": ("tab:orange", "-"), "flow": ("tab:blue", "-")}
    for s, (col, ls) in styles.items():
        r = ab.loc[s]
        ax.plot(phi, r["Y"] * phi ** r["delta"], ls, color=col, lw=1.4,
                label=f"fit on {s}: Ŷ = {r['Y']:.2f}, δ̂ = {r['delta']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("metaorder size φ = Q / daily volume")
    ax.set_ylabel("impact / daily σ")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Binned returns and square-root fits", fontsize=11)
    x = np.arange(2)
    for i, s in enumerate(("all", "alpha", "flow")):
        r = ab.loc[s]
        vals = [r["ratio_1pct"], r["ratio_10pct"]]
        err = [1.96 * r["se_ratio_1pct"], 1.96 * r["se_ratio_10pct"]]
        bars = bx.bar(x + (i - 1) * 0.26, vals, 0.24, yerr=err, capsize=4,
                      color=styles[s][0], label=f"fit on {s}")
        for rect, v, e in zip(bars, vals, err, strict=True):
            bx.text(rect.get_x() + rect.get_width() / 2, v + e + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    bx.axhline(1.0, color="k", ls="--", lw=1)
    bx.set_xticks(x, ["1% of ADV", "10% of ADV"])
    bx.set_ylabel("fitted / true impact (95% CI, delta method)")
    bx.set_ylim(0, 2.0)
    bx.legend(fontsize=9, loc="upper right")
    bx.set_title("Fitted cost relative to truth", fontsize=11)
    fig.suptitle(
        "(b) Alpha in the orders makes impact look flatter and dearer; flow orders recover it: "
        f"δ̂ {ab.loc['alpha', 'delta']:.2f} on alpha orders vs {ab.loc['flow', 'delta']:.2f} "
        "on flow", fontsize=12.5)
    fig.text(0.5, -0.02,
             f"Mean realised alpha over the execution window: "
             f"{ab.loc['alpha', 'mean_alpha_ret_bps']:.1f} bps on alpha orders, "
             f"{ab.loc['all', 'mean_alpha_ret_bps']:.1f} bps on all orders. Fitting on flow "
             f"orders removes {ab['bias_removed'].iloc[0] * 100:.0f}% of the 1%-ADV bias.",
             ha="center", fontsize=10)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    metrics = {}
    for s in ("all", "alpha", "flow"):
        for col in ("Y", "delta", "se_Y", "se_delta", "ratio_1pct", "ratio_10pct",
                    "mean_alpha_ret_bps", "n"):
            metrics[f"{s}_{col}"] = float(ab.loc[s, col])
    metrics["bias_removed"] = float(ab["bias_removed"].iloc[0])
    # SE checked three ways: sandwich SE, SD across seeds, coverage of the 95% interval
    spread = seed_spread(range(5) if n < 200_000 else range(20), n // 2)
    metrics.update({"seeds_n": float(len(spread)), "seeds_flow_orders_each": float(n // 2),
                    "seeds_delta_sd": float(spread["delta"].std(ddof=1)),
                    "seeds_delta_mean_sandwich_se": float(spread["se_delta"].mean()),
                    "seeds_Y_sd": float(spread["Y"].std(ddof=1)),
                    "seeds_Y_mean_sandwich_se": float(spread["se_Y"].mean()),
                    "seeds_delta_coverage_95": float(spread["covered"].mean())})
    return _rows("b", metrics)


# --- (c) out of sample ---------------------------------------------------------------------

def chart_out_of_sample(seed: int, out_path, n: int = 200_000) -> pd.DataFrame:
    regs, dec = out_of_sample(seed, n)
    regs = regs.set_index(["fit_on", "test_on"])
    series = [(("flow", "flow"), "tab:blue", "-o"), (("flow", "alpha"), "tab:green", "-s"),
              (("alpha", "flow"), "tab:red", "-^"), (("alpha", "alpha"), "tab:orange", ":v")]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    top = 0.0
    for key, col, fmt in series:
        d = dec[(dec["fit_on"] == key[0]) & (dec["test_on"] == key[1])]
        r = regs.loc[key]
        ax.errorbar(d["predicted_bps"], d["realised_bps"], yerr=2 * d["se_bps"], fmt=fmt,
                    color=col, ms=5, capsize=2,
                    label=f"fit on {key[0]}, test on {key[1]}: a = {r['a']:+.1f} bps, "
                          f"b = {r['b']:.2f}, mean resid = {r['mean_resid']:+.1f} bps")
        top = max(top, d["predicted_bps"].max(), d["realised_bps"].max())
    lim = top * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="45° (calibrated)")
    ax.set_xlim(0, lim)
    ax.set_ylim(-5, lim)
    ax.set_xlabel("mean predicted cost per decile (bps)")
    ax.set_ylabel("mean realised arrival-to-end return (bps, ± 2 SE)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("(c) Calibrated on alpha orders, the model passes its own test and fails on\n"
                 f"flow: mean residual {regs.loc[('alpha', 'alpha'), 'mean_resid']:+.1f} bps on "
                 f"alpha orders vs {regs.loc[('alpha', 'flow'), 'mean_resid']:+.1f} bps on flow",
                 fontsize=12)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    metrics = {}
    for (fo, to), r in regs.iterrows():
        for col in ("a", "b", "se_a", "se_b", "mean_resid", "n"):
            metrics[f"fit_{fo}_test_{to}_{col}"] = float(r[col])
    return _rows("c", metrics)


# --- (d) capacity --------------------------------------------------------------------------

def _mark(ax, be, opt, net_at, col):
    for x, m in ((be, "o"), (opt, "D")):
        if np.isfinite(x) and AUM_GRID[0] <= x <= AUM_GRID[-1]:
            ax.plot(x, net_at(x) * 100, m, color=col, ms=7, mec="k", zorder=5)


def chart_capacity(seed: int, out_path, n: int = 200_000, quick: bool = False) -> pd.DataFrame:
    cap = CapacityParams()
    universe = make_universe(np.random.default_rng(seed))
    ab = alpha_bias(seed, n).set_index("subset")
    flow, naive = fit_from_row(ab.loc["flow"]), fit_from_row(ab.loc["all"])
    laws = default_laws()
    mfits, _ = misspecification(seed, n, {k: laws[k] for k in ("log", "crossover_stress")})
    mfits = mfits.set_index("law")
    metrics = {}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    ax = axes[0]
    band = capacity_band(AUM_GRID, universe, flow, cap, np.random.default_rng(seed + 2),
                         n_draws=50 if quick else 500)
    ax.fill_between(AUM_GRID, band["lo"] * 100, band["hi"] * 100, color="tab:red", alpha=0.18,
                    lw=0, label="flow fit, 5-95% parameter band")
    pts = {}
    for lab, law, col, ls in (("true", SqrtLaw(), "k", "-"),
                              ("flow", flow.as_law(), "tab:red", "-"),
                              ("naive", naive.as_law(), "0.5", "--")):
        cur = capacity_curve(AUM_GRID, universe, law, cap)
        be, opt = capacity_points(universe, law, cap)
        pts[lab] = (be, opt)
        ax.plot(AUM_GRID, cur["net_alpha"] * 100, ls, color=col, lw=1.8,
                label={"true": "true square-root law", "flow": "flow-order fit",
                       "naive": "naive all-orders fit"}[lab])
        _mark(ax, be, opt, lambda a, law=law: capacity_curve([a], universe, law, cap)
              ["net_alpha"].iloc[0], col)
        metrics[f"d1_{lab}_break_even"] = be
        metrics[f"d1_{lab}_dollar_optimal"] = opt
    q_lo, q_mid, q_hi = band.attrs["break_even_q"]
    metrics.update({"d1_band_break_even_lo": q_lo, "d1_band_break_even_mid": q_mid,
                    "d1_band_break_even_hi": q_hi})
    med_phi_be = float(capacity_curve([pts["true"][0]], universe, SqrtLaw(), cap)
                       ["median_phi"].iloc[0]) if np.isfinite(pts["true"][0]) else float("nan")
    metrics["d1_true_break_even_median_phi"] = med_phi_be
    ax.text(0.02, 0.03,
            f"break-even: true {_bn(pts['true'][0])}, flow fit {_bn(pts['flow'][0])}, "
            f"naive {_bn(pts['naive'][0])}\nflow-fit band: {_bn(q_lo)} to {_bn(q_hi)}\n"
            f"median trade at true break-even: {med_phi_be * 100:.0f}% of ADV\n"
            "(sampled range ends at 20%)\n○ break-even   ◇ dollar-optimal",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
    ax.set_title("(d1) Alpha bias in the cost model", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    for name, col in (("log", "tab:purple"), ("crossover_stress", "tab:green")):
        r = mfits.loc[name]
        fit = fit_from_row(r)
        true_c = capacity_curve(AUM_GRID, universe, laws[name], cap, phi_max=r["phi_max"])
        fit_c = capacity_curve(AUM_GRID, universe, fit.as_law(), cap)
        short = {"log": "log", "crossover_stress": "crossover stress"}[name]
        ax.plot(AUM_GRID, true_c["net_alpha"] * 100, "-", color=col, lw=1.8,
                label=f"true {short} law")
        ax.plot(AUM_GRID, fit_c["net_alpha"] * 100, "--", color=col, lw=1.8,
                label=f"its square-root fit (δ̂ = {fit.delta:.2f})")
        for lab, law in (("true", laws[name]), ("fit", fit.as_law())):
            be, opt = capacity_points(universe, law, cap)
            metrics[f"d2_{name}_{lab}_break_even"] = be
            metrics[f"d2_{name}_{lab}_dollar_optimal"] = opt
        if name == "log":
            out = ~true_c["phi_in_sample"].to_numpy()
            if out.any():
                x0 = AUM_GRID[np.argmax(out)]
                ax.axvspan(x0, AUM_GRID[-1], facecolor="none", hatch="//", edgecolor="0.7",
                           lw=0, label="median trade beyond sampled φ (extrapolation)")
                metrics["d2_extrapolation_from_aum"] = x0
    ax.set_title("(d2) Misspecified law: capacity moves with the extrapolation", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.98, 0.97,
            f"break-even, log: true {_bn(metrics['d2_log_true_break_even'])}, "
            f"fit {_bn(metrics['d2_log_fit_break_even'])}\n"
            f"crossover stress: true {_bn(metrics['d2_crossover_stress_true_break_even'])}, "
            f"fit {_bn(metrics['d2_crossover_stress_fit_break_even'])}",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))

    ax = axes[2]
    opt_grid = np.logspace(8, 11, 3 if quick else 7)
    p = OptimizerParams(n_periods=15, burn_in=5) if quick else OptimizerParams()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ro = reoptimised_capacity(opt_grid, universe,
                                  {"true": SqrtLaw(), "flow": flow.as_law(),
                                   "naive": naive.as_law()}, SqrtLaw(), cap, p, seed)
    tx = ax.twinx()
    for lab, col, ls in (("true", "k", "-"), ("flow", "tab:red", "-"), ("naive", "0.5", "--")):
        g = ro[ro["model"] == lab]
        ax.plot(g["aum"], g["net_alpha"] * 100, ls, marker="o", color=col, lw=1.8,
                label=f"sized with {lab} cost: net alpha")
        tx.plot(g["aum"], g["turnover"], ls, marker="x", color=col, lw=0.9, alpha=0.6)
        for _, r in g.iterrows():
            metrics[f"d3_{lab}_net_alpha_at_{r['aum']:.0e}"] = r["net_alpha"]
            metrics[f"d3_{lab}_turnover_at_{r['aum']:.0e}"] = r["turnover"]
    tx.set_yscale("log")
    tx.set_ylabel("annual traded value / AUM (×, thin lines)")
    ax.set_title("(d3) Optimizer in the loop, charged the true cost", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")

    for a in axes:
        a.set_xscale("log")
        a.set_xlim(AUM_GRID[0], AUM_GRID[-1])
        a.set_xlabel("AUM (USD)")
        a.axhline(0, color="k", lw=0.6)
    axes[0].set_ylim(-4, 7)
    axes[1].set_ylim(-4, 7)
    axes[0].set_ylabel("net alpha (%/yr)")
    ratio = pts["naive"][0] / pts["true"][0]
    fig.suptitle(
        f"(d) Cost-model error is a capacity error: the naive fit puts break-even at {ratio:.2f}x "
        "the truth, because capacity is set by trades larger than any in the data", fontsize=12.5)
    fig.text(0.5, -0.04,
             "Fixed turnover 4x/yr, 12 rebalances, gross alpha 6%/yr, half-spread 5 bps, 50 names "
             "(illustrative). Measured costs tend to be lower than aggregate estimates and "
             "capacity larger (Frazzini, Israel & Moskowitz 2018).\nBreak-even scales as "
             "(margin / Y)^(1/δ): a 10% error in Y moves it by about 20% at δ = 0.5, and a "
             "flatter δ moves it further for trades beyond the sampled range. (d3) uses an AR(1) "
             "signal with an uncalibrated alpha scale: compare the cost models, not the level.",
             ha="center", fontsize=9.5)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return _rows("d", metrics)


def make_report(out_dir="reports", seed: int = DEFAULT_SEED, quick: bool = False) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 20_000 if quick else 200_000
    paths = [out / "a_misspecification.png", out / "b_alpha_bias.png",
             out / "c_out_of_sample.png", out / "d_capacity.png"]
    parts = [chart_misspecification(seed, paths[0], n), chart_alpha_bias(seed, paths[1], n),
             chart_out_of_sample(seed, paths[2], n), chart_capacity(seed, paths[3], n, quick)]
    summary = out / "summary.csv"
    pd.concat(parts, ignore_index=True).to_csv(summary, index=False, float_format="%.6g")
    return [str(p) for p in [*paths, summary]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for p in make_report(a.out, a.seed, a.quick):
        print(p)


if __name__ == "__main__":
    main()
