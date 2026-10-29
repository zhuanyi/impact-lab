"""Experiments behind the charts and the acceptance tests (SPEC 3.4).

Each experiment draws its orders from `numpy.random.default_rng(seed)`, so the same seed gives
the same orders across experiments and laws (laws consume no random numbers).
"""

import numpy as np
import pandas as pd

from impact_lab.fit import PowerFit, binned_means, calibration_regression, fit_power
from impact_lab.generate import MetaorderParams, generate_metaorders
from impact_lab.laws import CrossoverLaw, ImpactLaw, LogLaw, SqrtLaw, local_elasticity

DEFAULT_SEED = 20261028
DEFAULT_N = 200_000
OOS_PHI = (1e-4, 3e-4, 0.3, 0.5, 1.0)   # out-of-sample sizes, evaluated at T = OOS_T
OOS_T = 0.25
RATIO_PHI = (1e-3, 1e-1, 0.5)            # sizes where fitted / true is reported


def default_laws() -> dict[str, ImpactLaw]:
    return {"sqrt": SqrtLaw(), "crossover_paper": CrossoverLaw(eta_star=1e-3),
            "crossover_stress": CrossoverLaw(eta_star=5e-2), "log": LogLaw()}


def _z(df: pd.DataFrame) -> np.ndarray:
    return (df["ret"] / df["sigma_d"]).to_numpy()


def _fit(df: pd.DataFrame) -> PowerFit:
    return fit_power(df["phi"].to_numpy(), _z(df))


def _fit_cols(f: PowerFit) -> dict:
    se_Y, se_delta = f.se()
    return {"Y": f.Y, "delta": f.delta, "se_Y": se_Y, "se_delta": se_delta,
            "cov_YY": f.cov[0, 0], "cov_Yd": f.cov[0, 1], "cov_dd": f.cov[1, 1], "n": f.n}


def fit_from_row(row: pd.Series) -> PowerFit:
    """Rebuild the PowerFit stored in a fits row (columns Y, delta, cov_*, n)."""
    cov = np.array([[row["cov_YY"], row["cov_Yd"]], [row["cov_Yd"], row["cov_dd"]]])
    return PowerFit(Y=float(row["Y"]), delta=float(row["delta"]), cov=cov, n=int(row["n"]))


def ratio_at(df: pd.DataFrame, f: PowerFit, law: ImpactLaw, phi_q: float) -> float:
    """Fitted / true impact near phi_q: mean fitted over mean true impact / sigma of orders with
    phi in [0.8, 1.25] x phi_q (both on the same orders, so a window cut by the sampled range
    does not bias the ratio); law at T = OOS_T when no order is that size."""
    m = (df["phi"] >= 0.8 * phi_q) & (df["phi"] <= 1.25 * phi_q)
    if m.any():
        true = (df.loc[m, "impact"] / df.loc[m, "sigma_d"]).mean()
        return float(f.predict(df.loc[m, "phi"]).mean() / true)
    true = law.impact(np.array([phi_q]), np.array([phi_q / OOS_T]))[0]
    return float(f.predict(phi_q) / true)


def recovery(seed: int = DEFAULT_SEED, n: int = DEFAULT_N, law: SqrtLaw | None = None
             ) -> pd.DataFrame:
    """Square-root fit on flow orders of a square-root truth: the calibration check."""
    law = law or SqrtLaw()
    df = generate_metaorders(np.random.default_rng(seed), n, law)
    f = _fit(df[df["flow"]])
    row = {"subset": "flow", **_fit_cols(f), "level_1pct": f.level(0.01)[0],
           "true_Y": law.Y, "true_delta": law.delta}
    return pd.DataFrame([row])


def misspecification(seed: int = DEFAULT_SEED, n: int = DEFAULT_N,
                     laws: dict[str, ImpactLaw] | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Square-root fit to each law on flow orders. Returns (fits, curves)."""
    laws = laws or default_laws()
    fits, curves = [], []
    for name, law in laws.items():
        df = generate_metaorders(np.random.default_rng(seed), n, law)
        df = df[df["flow"]].reset_index(drop=True)
        f = _fit(df)
        row = {"law": name, **_fit_cols(f), "phi_min": df["phi"].min(),
               "phi_max": df["phi"].max()}
        for q in RATIO_PHI:
            row[f"ratio_at_{q:g}"] = ratio_at(df, f, law, q)
        fits.append(row)

        bins = binned_means(df["phi"], _z(df))
        true_z = binned_means(df["phi"], (df["impact"] / df["sigma_d"]).to_numpy())["z_mean"]
        oos = np.array(OOS_PHI)
        phi = np.concatenate([bins["phi_mid"].to_numpy(), oos])
        true = np.concatenate([true_z.to_numpy(), law.impact(oos, oos / OOS_T)])
        fitted = f.predict(phi)
        curves.append(pd.DataFrame({
            "law": name,
            "phi": phi,
            "true": true,
            "fitted": fitted,
            "ratio": fitted / true,
            "elasticity_true": local_elasticity(law, phi, T=OOS_T),
            "delta_hat": f.delta,
            "in_sample": np.r_[np.ones(len(bins), bool), np.zeros(oos.size, bool)],
            "z_mean": np.r_[bins["z_mean"].to_numpy(), np.full(oos.size, np.nan)],
            "z_se": np.r_[bins["z_se"].to_numpy(), np.full(oos.size, np.nan)],
        }))
    curves_df = pd.concat(curves, ignore_index=True).sort_values(["law", "phi"], kind="stable")
    return pd.DataFrame(fits), curves_df.reset_index(drop=True)


def _ratio(f: PowerFit, law: ImpactLaw, phi_ref: float) -> tuple[float, float]:
    true = float(law.impact(np.array([phi_ref]), np.array([phi_ref / OOS_T]))[0])
    lvl, se = f.level(phi_ref)
    return lvl / true, se / true


def alpha_bias(seed: int = DEFAULT_SEED, n: int = DEFAULT_N, law: ImpactLaw | None = None
               ) -> pd.DataFrame:
    """Square-root fits on all, alpha-only and flow-only orders of one population."""
    law = law or SqrtLaw()
    df = generate_metaorders(np.random.default_rng(seed), n, law)
    rows = []
    for subset, m in (("all", np.ones(len(df), bool)), ("alpha", ~df["flow"]),
                      ("flow", df["flow"])):
        sub = df[m]
        f = _fit(sub)
        r1, se1 = _ratio(f, law, 0.01)
        r10, se10 = _ratio(f, law, 0.1)
        rows.append({"subset": subset, **_fit_cols(f), "ratio_1pct": r1, "se_ratio_1pct": se1,
                     "ratio_10pct": r10, "se_ratio_10pct": se10,
                     "mean_alpha_ret_bps": float(sub["alpha_ret"].mean() * 1e4)})
    out = pd.DataFrame(rows)
    r = out.set_index("subset")["ratio_1pct"]
    out["bias_removed"] = 1.0 - abs(r["flow"] - 1.0) / abs(r["all"] - 1.0)
    return out


def out_of_sample(seed: int = DEFAULT_SEED, n: int = DEFAULT_N, law: ImpactLaw | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate on a random half, predict the other half. Returns (regressions, deciles)."""
    law = law or SqrtLaw()
    rng = np.random.default_rng(seed)
    df = generate_metaorders(rng, n, law)
    calib = rng.uniform(size=n) < 0.5
    masks = {"flow": df["flow"].to_numpy(), "alpha": ~df["flow"].to_numpy(),
             "all": np.ones(n, bool)}
    regs, decs = [], []
    for fit_on in ("flow", "alpha", "all"):
        f = _fit(df[calib & masks[fit_on]])
        for test_on in ("flow", "alpha"):
            t = df[~calib & masks[test_on]]
            pred = (t["sigma_d"] * f.predict(t["phi"])).to_numpy() * 1e4
            real = t["ret"].to_numpy() * 1e4
            reg = calibration_regression(real, pred)
            regs.append({"fit_on": fit_on, "test_on": test_on, **reg.to_dict()})
            g = pd.DataFrame({"pred": pred, "real": real,
                              "decile": pd.qcut(pred, 10, labels=False)}).groupby("decile")
            decs.append(pd.DataFrame({
                "fit_on": fit_on, "test_on": test_on,
                "decile": np.arange(10, dtype=np.int64),
                "predicted_bps": g["pred"].mean().to_numpy(),
                "realised_bps": g["real"].mean().to_numpy(),
                "se_bps": (g["real"].std(ddof=1) / np.sqrt(g["real"].count())).to_numpy(),
            }))
    return pd.DataFrame(regs), pd.concat(decs, ignore_index=True)


def seed_spread(seeds=range(20), n: int = 100_000, law: SqrtLaw | None = None) -> pd.DataFrame:
    """Flow-only square-root fits across seeds: one row per seed with Y, delta, se_Y,
    se_delta and covered (95% sandwich interval for delta contains the truth)."""
    law = law or SqrtLaw()
    rows = []
    for s in seeds:
        df = generate_metaorders(np.random.default_rng(s), n, law, MetaorderParams(flow_share=1.0))
        f = _fit(df)
        se_Y, se_d = f.se()
        rows.append({"seed": s, "Y": f.Y, "delta": f.delta, "se_Y": se_Y, "se_delta": se_d,
                     "covered": abs(f.delta - law.delta) <= 1.96 * se_d})
    return pd.DataFrame(rows)
