# SPEC: impact-lab

Calibrating a metaorder impact model on synthetic data where the true impact law, the true
alpha and the flow/informed label of every order are known, and carrying the calibration error
into a cost-penalised optimizer and a capacity curve.

**Reader.** A quant researcher or portfolio-construction interviewer who asks "how big can this
strategy get, and how wrong is your cost model?" They know mean-variance and regression; they
may not know the square-root law literature. Every chart answers one question and shows
reported vs true vs bias.

**Feasibility.** Every design choice that depends on speed, API shape or statistical power was
probed first in a throwaway feasibility spike (not committed). Numbers marked "spike:" below
come from it.

## 0. Notation

All returns are signed by the order's side, so positive = price moved against the order
(cost). Units are return units in code; charts and the README use bps (x 1e4).

| Symbol | Code | Meaning |
|---|---|---|
| `phi` | `phi` | `Q / V_D`, metaorder size as a fraction of daily volume |
| `T` | `T` | volume-time length of the execution window, fraction of the day's volume, in (0, 1] |
| `eta` | `eta` | participation over the window, `phi / T` |
| `sigma_d` | `sigma_d` | daily return volatility of the name on that day |
| `I` | `impact` | true metaorder impact, arrival mid to end-of-execution mid, return units |
| `alpha_ret` | `alpha_ret` | true PM alpha drift realised over the execution window |
| `ret` | `ret` | observed signed return arrival -> end = `impact + alpha_ret + noise` |
| `z` | — | `ret / sigma_d`, the regression target |
| `Y`, `delta` | `Y`, `delta` | square-root-law prefactor and exponent, `I / sigma_d = Y phi^delta` |
| `A` | `aum` | assets under management, USD |
| `tau` | `turnover` | annual one-way turnover as a multiple of AUM |

`I` is the peak (end-of-execution) impact, the quantity Tóth et al. (2011), Zarinelli et al.
(2015) and Bucci et al. (2019) fit. The cost per unit traded used in the capacity curve and the
optimizer is taken equal to `I`. The average fill pays less than the peak
(about two thirds for a square-root propagator); the same convention is applied to every law, so
comparisons between laws and fits are unaffected. Stated as a limitation.

## 1. Design decisions

1. **impact-lab generates metaorders itself, vectorised, at metaorder level.** The public
   per-minute path in tca-groundtruth (simulate_day -> apply_alpha -> execute) costs 6.3 ms per
   order (spike: 159 orders/s, 50k orders in 5.2 min). The recovery test needs 100k flow orders
   per fit and the report runs about ten fits. The vectorised generator makes 200k orders in
   0.07 s. The per-minute path is also closed to alternative laws (the child impact formula is
   inline in `execute`).
2. **Reuse, not copy, from tca-groundtruth** (the pinned commit):
   `orders.generate_orders` and `OrderParams` (sizes, flow label, alpha, half-life, decision /
   arrival / end minutes), `market.u_profile` (intraday profile; its cumulative sum gives the
   window's volume fraction `T`, which is also its variance fraction because tca-groundtruth uses one profile
   for both). The window alpha drift is the closed form of `orders.apply_alpha`; a test pins the
   two together (spike: max diff 2e-19). `execution.execute` and `ImpactParams` are used in one
   cross-check test that ties the fast generator to tca-groundtruth's propagator (spike: tca-groundtruth with
   `gamma_perm=0` gives a metaorder law `Y=0.476, delta=0.500`). **No change to tca-groundtruth
   is required.**
3. **Impact laws are impact-lab dataclasses** with one method `impact(phi, eta)` in sigma units.
   Three laws: square root; linear-to-square-root crossover in participation (Bucci et al.
   2019); logarithmic in `phi` (Zarinelli et al. 2015). The generator takes any object with that
   method. No registry, no plugin system.
4. **The square-root recovery is the calibration check, not the result.** The results are what a
   square-root fit does under (i) misspecification and (ii) alpha, and how each error moves the
   capacity curve. The README says this in its first paragraph (otherwise the exercise is circular).
5. **Fit by NLS on levels, not log-log OLS.** Single-order `ret` is negative about 40% of the
   time (noise of ~50-130 bps vs impact of ~5-60 bps), so logs of single orders are undefined.
   Binned log-log OLS is biased at this noise level (spike: delta 0.557 vs NLS 0.494 on the same
   100k flow orders), so bins are display only. Covariance is HC0 sandwich (noise variance scales
   with `T`, so it is heteroskedastic). Unweighted NLS is consistent; WLS with weights `1/T` is
   more efficient but not needed for the tolerances (Not done).
6. **The alpha fix is Gomes & Waelbroeck's: fit on flow-labelled orders.** "Control for the
   signal" assumes the TCA team has the signal; it is one README sentence, not code.
   Reversion-based identification is already in tca-groundtruth chart (c); not repeated.
7. **Out-of-sample loop = calibrate on a random half, predict the other half,** with the
   calibration regression `realised = a + b * predicted` evaluated separately on
   flow and on alpha orders. The spike shows the point worth making: a model fitted on alpha
   orders looks calibrated on alpha orders (b = 0.94, mean residual +0.3 bps) and is only exposed
   on flow orders (mean residual -6.5 bps).
8. **Capacity curve = net alpha vs AUM at fixed turnover**, evaluated
   vectorised for any law, with a parameter-uncertainty band from 500 draws of `(Y, delta)` from
   the fit's sandwich covariance. No optimizer is needed for this.
9. **cvxpy is used for one thing: the cost-penalised mean-variance optimizer**, with impact cost
   `c_i |dw_i|^(1+delta)`. It is DCP and solves in ~18 ms for 50 names with power
   cones (spike). It produces one extra panel: realised net alpha vs AUM, under the true cost,
   when the optimizer sizes trades with the true, flow-fitted or naive cost model. That is "cost
   model error flows into position sizing" made concrete. The band is not computed through the
   optimizer (500 draws x grid x periods ~ 2.5 h; spike).
10. **Only the square-root truth goes through the optimizer.** The log and crossover laws are not
    DCP-representable as cost terms; for them capacity is the fixed-turnover curve only.
11. **One seed, fixed sizes.** `make report` rebuilds every chart from seed 20261028, as in tca-groundtruth.

## 2. Assumptions

- One metaorder per simulated day per name; orders are independent. No cross-impact, no
  overlapping metaorders, no herding (Zarinelli et al. discuss it; not modelled).
- Execution is complete at `end_minute` (no partial fills) and proceeds uniformly in volume
  time. Spread and fees are excluded from `ret` (mid to mid); the capacity curve adds a
  half-spread per unit traded.
- `noise = sigma_d * sqrt(T) * N(0, 1)`: the window's share of daily variance. Index and
  idiosyncratic risk are not separated (no market adjustment needed at metaorder level).
- `sigma_d` is drawn per order, uniform in `[0.01, 0.04]`, and is known to the fitter (in
  practice it is estimated; Not done).
- Alpha is tca-groundtruth's: total `alpha_bps` uniform 5-30 bps for informed orders, exponential half-life
  30-240 min from the decision minute, PM-to-desk delay 0-60 min, independent of size. The
  realised window drift is
  `alpha_ret = alpha_bps/1e4 * (exp(-(t0 - td) ln2 / h) - exp(-(t1 - td) ln2 / h))`
  with `td`, `t0`, `t1` the decision, arrival and end minutes. Size-independent alpha biases
  delta down; size-dependent alpha is Not done.
- Sizes: `phi` log-uniform in `[1e-3, 0.2]` (wider than tca-groundtruth's default `[0.005, 0.10]`), i.e. the
  range where Bucci et al. report the square-root regime (`1e-3 <~ phi <~ 1e-1`) plus a margin
  above it.

## 3. Interface contract

Package `impact_lab` under `src/`. All functions are pure given an explicit
`rng: numpy.random.Generator` argument. DataFrames have a `RangeIndex`. Parameter dataclasses are
frozen and passed as `params: X | None = None` (ruff B008, as in tca-groundtruth). Arrays are float64 unless
stated.

### 3.1 `impact_lab.laws`

```python
class ImpactLaw(Protocol):
    def impact(self, phi: np.ndarray, eta: np.ndarray) -> np.ndarray: ...
    # metaorder impact in units of sigma_d; broadcasting numpy arrays; phi > 0, eta > 0

@dataclass(frozen=True)
class SqrtLaw:
    Y: float = 0.5
    delta: float = 0.5
    # impact = Y * phi**delta   (eta ignored)

@dataclass(frozen=True)
class CrossoverLaw:
    Y: float = 0.5
    eta_star: float = 1e-3
    # impact = Y * sqrt(phi) * sqrt(u / (1 + u)),  u = eta / eta_star
    # eta >> eta_star: Y sqrt(phi) (square root, duration-free);
    # eta << eta_star: Y phi / sqrt(T eta_star) (linear in phi at fixed T).
    # Limits from Bucci et al. (2019) eq. 5; the interpolation is a modelling choice.

@dataclass(frozen=True)
class LogLaw:
    a: float = 0.0665
    b: float = 465.0
    # impact = a * log10(1 + b * phi). b from Zarinelli et al. (2015);
    # a = 0.05 / log10(1 + 4.65) so that impact(0.01) == SqrtLaw().impact(0.01) == 0.05.

def local_elasticity(law: ImpactLaw, phi: np.ndarray, T: float = 0.25,
                     h: float = 1e-4) -> np.ndarray
    # d ln impact / d ln phi at fixed T (eta = phi / T), central difference in ln phi.
    # The "true delta" of a non-power law at each size; == delta for SqrtLaw.
```

Errors: `ValueError` on construction if `Y <= 0`, `delta <= 0`, `eta_star <= 0`, `a <= 0` or
`b <= 0` (`__post_init__`).

### 3.2 `impact_lab.generate`

```python
@dataclass(frozen=True)
class MetaorderParams:
    qty_pct_adv: tuple[float, float] = (0.001, 0.2)   # log-uniform phi
    daily_vol: tuple[float, float] = (0.01, 0.04)     # uniform sigma_d per order
    flow_share: float = 0.5
    u_shape: float = 2.0                              # passed to u_profile
    n_minutes: int = 390
    # every other order attribute uses tca_groundtruth.orders.OrderParams defaults

def order_params(p: MetaorderParams) -> tca_groundtruth.orders.OrderParams
    # OrderParams(qty_pct_adv=p.qty_pct_adv, flow_share=p.flow_share)

def window_fraction(arrival: np.ndarray, end: np.ndarray, n_minutes: int,
                    u_shape: float) -> np.ndarray
    # cum = concat([0], cumsum(u_profile(n_minutes, u_shape) / n_minutes)); return cum[end] - cum[arrival]

def alpha_window_return(alpha_bps, halflife_min, decision, arrival, end) -> np.ndarray
    # closed form in section 2; arrays of equal length

def generate_metaorders(rng, n: int, law: ImpactLaw,
                        params: MetaorderParams | None = None) -> pd.DataFrame
```

Implementation: one call to `generate_orders(rng, n, daily_volume=1e8, params=order_params(p),
n_minutes=p.n_minutes)` and `phi = qty / 1e8` (`generate_orders` rounds `qty` to whole shares;
at `1e8` the rounding error is below 1e-5 relative at `phi = 1e-3`). Then vectorised draws from
the same `rng`, in this order: `sigma_d`, then the standard normals for `noise`.

Output columns:

| column | dtype | meaning |
|---|---|---|
| `order_id` | int64 | 0..n-1 |
| `phi` | float64 | `Q / V_D` |
| `T` | float64 | `window_fraction(arrival, end)`, in (0, 1] |
| `eta` | float64 | `phi / T` |
| `sigma_d` | float64 | daily vol |
| `flow` | bool | from `generate_orders` |
| `alpha_bps` | float64 | 0 for flow |
| `alpha_ret` | float64 | `alpha_window_return(...)`, 0 for flow |
| `impact` | float64 | `sigma_d * law.impact(phi, eta)` (true) |
| `noise` | float64 | `sigma_d * sqrt(T) * N(0,1)` |
| `ret` | float64 | `impact + alpha_ret + noise` |

```python
def p1_peak_impact(rng, n: int, impact: tca_groundtruth.execution.ImpactParams,
                   qty_pct_adv=(0.001, 0.2), algo: str = "TWAP") -> pd.DataFrame
    # Cross-check path through tca-groundtruth's per-minute simulator (slow; n <= 1000).
    # generate_orders -> simulate_day -> apply_alpha -> execute with
    # FillParams(passive_frac=0, max_participation=1). Returns columns
    # phi = filled qty / day volume, z = side * impact_shift[last fill] / mid[last fill] / sigma_day
    # (noise-free true peak impact in sigma units).
```

Errors: `ValueError` if `n < 1`, `flow_share` outside [0, 1], or an inverted range.

### 3.3 `impact_lab.fit`

```python
@dataclass(frozen=True)
class PowerFit:
    Y: float
    delta: float
    cov: np.ndarray          # 2x2, order (Y, delta), HC0 sandwich
    n: int
    def se(self) -> tuple[float, float]
    def predict(self, phi) -> np.ndarray               # Y * phi**delta, sigma units
    def level(self, phi_ref: float = 0.01) -> tuple[float, float]
        # (Y * phi_ref**delta, delta-method SE): the cost of a reference-size order,
        # less collinear than Y alone (Y is the value at phi = 1, outside the data)
    def draws(self, rng, k: int) -> np.ndarray          # (k, 2) multivariate normal, Y clipped > 0
    def as_law(self) -> SqrtLaw

def fit_power(phi: np.ndarray, z: np.ndarray) -> PowerFit
    # scipy.optimize.least_squares on r = Y * exp(delta * ln phi) - z, x0 = (0.5, 0.5),
    # bounds Y in (0, inf), delta in (0, 2). Sandwich: B = (J'J)^-1, M = J' diag(r^2) J, cov = B M B.

def binned_means(phi, z, n_bins: int = 20) -> pd.DataFrame
    # quantile bins in ln phi; columns phi_mid (geometric mean), z_mean, z_se, n. Display only.

def calibration_regression(realised: np.ndarray, predicted: np.ndarray) -> pd.Series
    # statsmodels OLS realised ~ 1 + predicted, HC1; index a, b, se_a, se_b, mean_resid, n.
    # Inputs in bps.
```

How `Y` and `delta` are identified: `sigma_d` is known and divided out; `delta` comes from the
variation of `z` across 2.3 decades of `phi`; `Y` is the level. They are strongly correlated
(Y is the extrapolation to `phi = 1`), so the README and the tolerance on level use
`level(0.01)` alongside `Y`.

Standard errors are reported three ways in the report: sandwich SE, the empirical SD across 20
seeds (spike: delta SD 0.013 at 100k flow orders) and coverage of the 95% interval (test 7).

### 3.4 `impact_lab.experiments`

Each returns tidy DataFrames used by both the charts and the tests. Defaults: `seed=20261028`,
`n=200_000` (so about 100k flow orders at `flow_share=0.5`).

```python
def recovery(seed, n=200_000, law=SqrtLaw()) -> pd.DataFrame
    # one row per subset in ("flow",): Y, delta, se_Y, se_delta, level_1pct, true_Y, true_delta, n

def misspecification(seed, n=200_000, laws: dict[str, ImpactLaw] | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]
    # laws default: {"sqrt": SqrtLaw(), "crossover_paper": CrossoverLaw(eta_star=1e-3),
    #                "crossover_stress": CrossoverLaw(eta_star=5e-2), "log": LogLaw()}
    # Flow orders only. fits: one row per law: Y, delta, se_*, phi_min, phi_max (in-sample range).
    # curves: per law and phi bin (20 quantile bins + 6 out-of-sample points 1e-4, 3e-4, 0.3, 0.5, 1.0):
    #   phi, true (mean of true impact/sigma in the bin; out-of-sample points use T = 0.25),
    #   fitted, ratio = fitted / true, elasticity_true (local_elasticity), delta_hat, in_sample (bool)

def alpha_bias(seed, n=200_000, law=SqrtLaw()) -> pd.DataFrame
    # rows: subset in ("all", "alpha", "flow"): Y, delta, se_*, ratio_1pct, ratio_10pct
    # (fitted level / true level at phi = 0.01, 0.1), mean_alpha_ret_bps, n;
    # plus column bias_removed = 1 - |ratio_1pct(flow) - 1| / |ratio_1pct(all) - 1| (same in every row)

def out_of_sample(seed, n=200_000, law=SqrtLaw()) -> tuple[pd.DataFrame, pd.DataFrame]
    # split: rng.uniform < 0.5 -> calibration half. Fit on calib subsets ("flow", "alpha", "all").
    # regressions: one row per (fit_on, test_on in ("flow", "alpha")): calibration_regression output.
    # deciles: per (fit_on, test_on): predicted-cost decile, mean predicted bps, mean realised bps, se.
```

### 3.5 `impact_lab.capacity`

```python
@dataclass(frozen=True)
class Universe:
    n_names: int = 50
    adv_usd: tuple[float, float] = (2e7, 5e8)     # log-uniform daily dollar volume
    daily_vol: tuple[float, float] = (0.018, 0.04)
    beta: tuple[float, float] = (0.6, 1.4)
    index_vol: float = 0.012                       # daily
    half_spread_bps: float = 5.0

def make_universe(rng, u: Universe | None = None) -> pd.DataFrame
    # columns name (int64), adv_usd, sigma_d, beta
def covariance(universe: pd.DataFrame, index_vol: float) -> np.ndarray
    # annual, one-factor: 252 * (beta beta' index_vol^2 + diag(sigma_d^2 - beta^2 index_vol^2));
    # ValueError if any idiosyncratic variance <= 0 (the spike hit this; Universe defaults avoid it)

@dataclass(frozen=True)
class CapacityParams:
    gross_alpha: float = 0.06       # annual, illustrative
    turnover: float = 4.0           # annual one-way, multiple of AUM
    rebalances: int = 12            # per year
    exec_days: float = 1.0          # each rebalance trade executes over one day: T = 1, eta = phi

def trade_phi(aum: np.ndarray, universe, cap) -> np.ndarray
    # (len(aum), n_names): aum * turnover / (rebalances * n_names) / adv_usd (equal dollar trade per name)

def net_alpha(aum, universe, law: ImpactLaw, cap) -> np.ndarray
    # gross_alpha - turnover * mean_i(half_spread + sigma_i * law.impact(phi_i, phi_i / exec_days))

def capacity_points(universe, law, cap) -> tuple[float, float]
    # (break_even, dollar_optimal): brentq on net_alpha(A) = 0 in ln A over [1e6, 1e13];
    # minimize_scalar of -A * net_alpha(A) in ln A over [1e6, break_even]. nan if net_alpha(1e6) <= 0.

def capacity_curve(aum_grid, universe, law, cap) -> pd.DataFrame
    # columns aum, net_alpha, dollar_alpha, median_phi, phi_in_sample (bool, median_phi <= fit's phi_max)

def capacity_band(aum_grid, universe, fit: PowerFit, cap, rng, n_draws=500,
                  q=(0.05, 0.95)) -> pd.DataFrame
    # columns aum, lo, mid, hi (net_alpha quantiles over draws), plus attrs break_even_q (lo, mid, hi)
```

For the square-root law with one name, constant `sigma`, `ADV`:
`net(A) = alpha_g - tau*eps - b sqrt(A)`, `b = tau sigma Y sqrt(tau / (m N ADV))`, so
`A_0 = ((alpha_g - tau eps)/b)^2` and `A* = (4/9) A_0`; for a general power `delta`,
`A*/A_0 = (1 + delta)^(-1/delta)`. Tests 13-14 check the code against these.

### 3.6 `impact_lab.optimizer`

```python
@dataclass(frozen=True)
class OptimizerParams:
    risk_aversion: float = 2.0
    gross_limit: float | None = 2.0       # sum |w| <= gross_limit; None = no constraint
    signal_persistence: float = 0.9       # AR(1) per rebalance period
    alpha_scale: float = 0.10             # annual alpha per unit signal per unit annual vol
    n_periods: int = 60
    burn_in: int = 10

def optimize(alpha: np.ndarray, cov: np.ndarray, w0: np.ndarray, cost_coef: np.ndarray,
             delta: float, half_spread: float, p: OptimizerParams) -> np.ndarray
    # max  alpha'w - (lam/2) w' cov w - half_spread * ||w - w0||_1 - cost_coef' |w - w0|^(1+delta)
    # s.t. sum(w) == 0, ||w||_1 <= gross_limit. All terms per rebalance period (alpha and cov / rebalances).
    # cvxpy, cp.power(cp.abs(dw), 1 + delta, approx=False), solver CLARABEL.
    # RuntimeError if status not in ("optimal",).

def cost_coef(aum: float, universe, fit_or_law: SqrtLaw) -> np.ndarray
    # Y * sigma_i * (aum / adv_i)**delta : cost of |dw_i| is cost_coef_i * |dw_i|^(1+delta)

def reoptimised_capacity(aum_grid, universe, cost_models: dict[str, SqrtLaw], true_law: SqrtLaw,
                         cap: CapacityParams, p: OptimizerParams, seed: int) -> pd.DataFrame
    # Same signal path (seeded) for every model and AUM. Each period: optimize with the model's
    # cost_coef; charge the realised trade with true_law's cost; record alpha'w - true cost.
    # columns: model, aum, net_alpha (annualised, mean after burn_in), turnover (annual one-way)
```

### 3.7 `impact_lab.report`

```python
def chart_misspecification(seed, out_path, n=200_000) -> pd.DataFrame      # (a)
def chart_alpha_bias(seed, out_path, n=200_000) -> pd.DataFrame            # (b)
def chart_out_of_sample(seed, out_path, n=200_000) -> pd.DataFrame         # (c)
def chart_capacity(seed, out_path, n=200_000, quick=False) -> pd.DataFrame  # (d)
def make_report(out_dir="reports", seed=20261028, quick=False) -> list[str]
    # writes a_misspecification.png, b_alpha_bias.png, c_out_of_sample.png, d_capacity.png,
    # summary.csv (columns chart, metric, value); quick=True uses n=20_000, 3 AUM points,
    # n_periods=15, n_draws=50.
```

`python -m impact_lab.report [--out reports] [--seed N] [--quick]` and `make report` call
`make_report`. Full report target: under 5 minutes on one core.

## 4. Metaorder generator design (why it is valid)

- The generator is the metaorder-level image of tca-groundtruth's propagator. With `delta_child =
  beta_kernel = 0.5` (Gatheral's boundary) a uniform-in-volume-time metaorder has peak impact
  `Y sigma_d sqrt(phi)`, independent of duration (Zarinelli et al. 2015 s3.2 derive this for the
  propagator; Tóth et al. 2011 report it empirically). Test 5 confirms it numerically on tca-groundtruth's
  `execute`.
- tca-groundtruth's default `gamma_perm = 0.3` adds a permanent linear term, so tca-groundtruth's default metaorder law is
  not a pure square root (spike: apparent `delta = 0.557, Y = 0.653`). impact-lab's sqrt
  generator corresponds to `gamma_perm = 0`; the README says so.
- The crossover law depends on `eta`, so it carries a duration effect: at fixed `phi`, a longer
  window means lower participation and, below `eta*`, lower impact. That is the "impact surface"
  dependence, in its simplest form.
- Noise uses the window's variance fraction from the same `u_profile` as tca-groundtruth. No intraday path is
  simulated, so no price, spread or volume path exists at metaorder level.

## 5. Chart definitions

All charts: seed 20261028, flow orders unless stated, bps on axes, in-sample `phi` range shaded.
Each title states the finding, not the method.

**(a) `a_misspecification.png` — "A square-root fit is right where the data are and wrong where
you extrapolate."** Four panels, one per law in `misspecification()` (sqrt, crossover paper,
crossover stress, log). Each panel, log-log: x = `phi`; y = impact / `sigma_d`. Points: binned
means of `z` with +-2 SE bars. Black line: true law (binned mean of true impact; out-of-sample
points at `T = 0.25`). Red line: fitted square root with a 90% band from `PowerFit.draws`. Text:
`Y_hat +- SE`, `delta_hat +- SE`; for sqrt the true `(Y, delta)`; for the others the true local
elasticity at `phi = 1e-3, 1e-2, 1e-1`. A strip under each panel: fitted / true by `phi`
(1.0 line). summary.csv: per law `delta_hat`, `ratio_at_1e-3`, `ratio_at_1e-1`, `ratio_at_0.5`.

**(b) `b_alpha_bias.png` — "Alpha in the orders makes impact look flatter and dearer; flow orders
recover it."** Left, log-log: binned `z` vs `phi` for alpha orders and for flow orders; true law;
fitted lines for "all", "alpha only", "flow only". Right: grouped bars, fitted / true level at
1% and 10% ADV for the three fits, 95% CI from the delta method; dashed line at 1.0. Caption:
mean realised alpha over the window (bps) and `bias_removed`. summary.csv: the `alpha_bias()` rows.

**(c) `c_out_of_sample.png` — "Calibrated on alpha orders, the model passes its own test and fails
on flow."** Predicted-cost decile plot: x = mean predicted cost (bps), y = mean realised (bps)
+-2 SE, 45-degree line. Three series: flow-fitted model on flow test orders; flow-fitted model on
alpha test orders; alpha-fitted model on flow test orders; plus alpha-fitted on alpha (dotted).
Legend shows `a`, `b`, mean residual for each. summary.csv: `regressions` rows.

**(d) `d_capacity.png` — "Cost-model error is a capacity error."** Three panels, x = AUM (USD,
log, 1e8 to 1e11), y = net alpha (%/yr).
- (d1) Alpha bias: true sqrt law (black); flow fit (red) with 5-95% band; naive all-orders fit
  (grey). Markers at break-even and dollar-optimal for each. Text: break-even true / flow /
  naive, and the band's break-even interval.
- (d2) Misspecification: for the log and crossover-stress truths, the true curve (solid) and the
  curve under its square-root fit (dashed). Hatched region where the median trade `phi` is
  outside the fitted range (extrapolation).
- (d3) Optimizer in the loop: `reoptimised_capacity` with cost models {true, flow fit, naive fit}
  under the true sqrt cost; net alpha (left axis) and turnover (right axis). The naive model
  over-penalises trading, so it under-trades and leaves alpha on the table at small AUM.
Cite Frazzini, Israel & Moskowitz (2018) in the caption for "measured costs are lower than
aggregate estimates and capacity larger", and note how far break-even moves with `delta` and `Y`.
summary.csv: break-even and dollar-optimal per curve, band interval.

## 6. Acceptance tests (inputs -> expected outputs)

Seed 20261028 unless stated; "flow fit" = `fit_power` on flow orders of `generate_metaorders(rng,
200_000, law)`. Tolerances come from the spike (values in brackets).

1. **Laws.** `SqrtLaw().impact(0.04, 1.0) == 0.1` (1e-12). `CrossoverLaw(eta_star=1e-3)`: at
   `eta = 1.0` within 0.1% of `SqrtLaw`; at `eta = 1e-6`, within 0.1% of
   `Y sqrt(phi) sqrt(eta/eta_star)`. `LogLaw().impact(0.01, .) == 0.05` within 1e-3 relative.
   `local_elasticity(SqrtLaw(delta=0.37), ...) == 0.37` within 1e-6; `LogLaw` elasticity at
   `phi = 1e-6` > 0.99 and at `phi = 0.5` < 0.25. Constructors raise on non-positive params.
2. **Generator schema.** Columns and dtypes exactly as 3.2; `n` rows; same seed -> identical frame;
   `0 < T <= 1`; `eta == phi / T`; `flow == (alpha_bps == 0)`; `alpha_ret == 0` where flow;
   `ret == impact + alpha_ret + noise` (1e-15); `impact == sigma_d * law.impact(phi, eta)`;
   `phi` within `qty_pct_adv` (rounding 1e-5 relative).
3. **Noise.** `n = 200_000`: `noise / (sigma_d sqrt(T))` mean within 0.01 of 0, SD within 0.01 of 1.
4. **Alpha closed form = tca-groundtruth.** 50 orders from `generate_orders`; for each, the window return from
   `apply_alpha(simulate_day(...), order).alpha_shift` at `end` minus at `arrival`, divided by the
   decision-minute `mid_base`, equals `alpha_window_return` within 1e-12 [2e-19].
5. **Cross-check with tca-groundtruth's propagator** (`p1_peak_impact`, 300 TWAP orders, seed 11):
   `ImpactParams(gamma_perm=0)` -> `|delta_hat - 0.5| <= 0.05` and `|Y_hat / 0.5 - 1| <= 0.10`
   [0.500, 0.476]. `ImpactParams()` (default `gamma_perm=0.3`) -> `delta_hat >= 0.53` [0.557].
6. **Recovery.** Flow fit, sqrt truth `(0.5, 0.5)`, ~100k flow orders: `|delta_hat - 0.5| <= 0.05`,
   `|Y_hat / 0.5 - 1| <= 0.10`, `|level(0.01) / 0.05 - 1| <= 0.05` [spike over 20 seeds at 100k:
   delta SD 0.013, max error 0.030; Y SD 0.022].
7. **Coverage.** Seeds 0..19, 100k flow-only orders each: the 95% sandwich interval for `delta`
   covers 0.5 in at least 16 of 20 (about 3 s total; runs in CI).
8. **Fit sanity.** Noise-free `z = 0.3 phi^0.6` on 1,000 log-spaced `phi` -> `Y, delta` within
   1e-6. `binned_means` returns `n_bins` rows with `n` summing to the input length.
9. **Misspecification** (flow fits, 200k orders): `LogLaw()` -> `delta_hat <= 0.45` [0.372] and
   fitted / true at `phi = 1e-3` >= 1.3 [1.64]; `CrossoverLaw(eta_star=5e-2)` -> `delta_hat >= 0.55`
   [0.594] and ratio at `1e-3` >= 1.3 [1.81]; `CrossoverLaw(eta_star=1e-3)` ->
   `|delta_hat - 0.5| <= 0.08` [0.538]. The `ratio at 1e-3` is `predict(1e-3)` over the mean true
   impact / sigma of orders with `phi` in `[0.8e-3, 1.25e-3]`.
10. **Alpha bias** (`alpha_bias()`): alpha-only `delta_hat <= 0.42` [0.359] and
    `ratio_1pct >= 1.3` [1.56]; all-orders `ratio_1pct >= 1.15` [1.26]; flow
    `0.9 <= ratio_1pct <= 1.1` [0.97]; `bias_removed >= 0.7` [0.88].
11. **Out of sample** (`out_of_sample()`): flow-fit on flow test: `|a| <= 1.5` bps,
    `|b - 1| <= 0.1` [-0.11, 1.01]; flow-fit on alpha test: `mean_resid >= +4` bps [+6.9];
    alpha-fit on flow test: `mean_resid <= -4` bps [-6.5]; alpha-fit on alpha test:
    `|mean_resid| <= 1.5` bps [+0.3].
12. **Capacity shape.** `net_alpha` is strictly decreasing in AUM for every law; at `aum -> 1e6`
    it is within 1 bp of `gross_alpha - turnover * half_spread`.
13. **Capacity closed form, square root.** One-name universe (`sigma = 0.02`, `ADV = 1e8`),
    `SqrtLaw()`: `break_even` within 0.5% of `((alpha_g - tau eps)/b)^2` and `dollar_optimal /
    break_even` within 0.5% of 4/9.
14. **Capacity closed form, general delta.** Same universe, `SqrtLaw(delta=1.0)`:
    `dollar_optimal / break_even` within 0.5% of 1/2; `SqrtLaw(delta=0.3)`: within 0.5% of
    `1.3^(-1/0.3)`.
15. **Capacity bias.** Default universe: naive all-orders fit -> break-even <= 0.8 x true
    [spike, illustrative universe: 0.50]; the flow fit's 5-95% break-even interval contains the true
    break-even; band `lo <= mid <= hi` at every AUM.
16. **Optimizer.** (i) `cost_coef = 0`, `half_spread = 0`, `gross_limit=None`: solution equals the
    closed-form dollar-neutral mean-variance weights `(lam S_p)^-1 (alpha_p - mu 1)` (per-period
    `S_p = cov / rebalances`, `alpha_p = alpha / rebalances`) with `mu` set so weights sum to 0, max abs diff <= 1e-5. (ii) `cost_coef = 1e3`: `max |w - w0| <= 1e-4`.
    (iii) status optimal for `delta` in {0.36, 0.5, 0.6}. (iv) `reoptimised_capacity` (quick sizes):
    turnover non-increasing in AUM for each model (1% slack for solver tolerance).
17. **Report smoke.** `make_report(tmp_path, seed=1, quick=True)`: returns 5 existing paths, four
    PNGs with PNG magic bytes, `summary.csv` readable with columns `chart, metric, value`; runs in
    under 60 s.

## 7. Success criteria (runnable)

```
.venv/bin/ruff check .        # no findings
.venv/bin/pytest -q           # all green, under 2 min
make report                   # writes reports/*.png and reports/summary.csv from seed 20261028
make check                    # ruff + pytest
```

## 8. Repository layout

```
impact-lab/
  pyproject.toml   README.md   SPEC.md   LICENSE (MIT)   Makefile   .gitignore
  .github/workflows/ci.yml
  src/impact_lab/{__init__,laws,generate,fit,experiments,capacity,optimizer,report}.py
  tests/test_{laws,generate,fit,experiments,capacity,optimizer,report}.py
  reports/   # committed outputs from seed 20261028, as in tca-groundtruth
```

Stack: Python >= 3.11, numpy, pandas, scipy, statsmodels, cvxpy (with Clarabel, its default
install), matplotlib, pytest, ruff. No hypothesis (no property-style invariant here that a
deterministic test does not cover). ruff config copied from tca-groundtruth.

## 9. Dependency and CI

- `pyproject.toml` pins `tca-groundtruth @ git+https://github.com/zhuanyi/tca-groundtruth@<SHA>`:
  a full commit SHA, not a branch, so the dependency cannot move under impact-lab. Bump the pin
  in its own change when impact-lab needs a newer tca-groundtruth.
- Local install: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- CI: `.github/workflows/ci.yml` (Python 3.11/3.12, `pip install -e ".[dev]"`, ruff, pytest,
  15-minute timeout), the same steps as tca-groundtruth's workflow. PyPI is not planned.

## 10. README outline

1. **Question.** "How wrong is your impact model, in which direction, and what does that do to
   how big your strategy can get?"
2. **Headline chart:** (d1) capacity with the flow fit's band vs the naive fit and the truth.
3. **Three findings, in bps and dollars** (numbers filled from `summary.csv`):
   - Alpha in the orders over-prices a 1%-ADV order by ~X% and flattens delta to ~0.36; fitting on
     flow orders removes ~Y% of the bias; break-even capacity under the naive fit is ~Z% of the
     truth.
   - A square-root fit to a log law is within ~10% across the sampled sizes but over-prices 0.1%-
     ADV orders by ~60% and 50%-ADV orders by ~30%; capacity moves by W%.
   - Out-of-sample checks on the same population do not catch the alpha bias; monitoring on flow
     orders does (mean residual ~-6 bps).
4. **Method:** generator (laws, alpha, noise), NLS with sandwich SE, flow-order control,
   calibrate/predict split, fixed-turnover capacity with a parameter band, cvxpy optimizer with
   3/2-power cost. "Square-root recovery on square-root data is the calibration check, not the
   result."
5. **Why synthetic:** the true law, the alpha and the flow label are known per order; real data
   has none of the three. Sample size: ~100k uninformed metaorders to pin delta to +-0.05.
6. **Limitations:** peak impact used as cost per unit traded; alpha independent of size; no
   cross-impact or overlapping metaorders; `sigma_d` known; crossover interpolation is not the
   LLOB scaling function; log level is matched, not taken from the paper; capacity uses an
   equal-dollar trade list at fixed turnover; optimizer only for the square-root truth; one
   rebalance-day execution.
7. **References** (section 12).
8. **Data and IP statement:** "All data are simulated. Model forms and parameter values come from
   the published papers cited below; no proprietary data, schemas, parameters or code were used."
9. Relationship to tca-groundtruth (one line, link) and how to run (`make report`).

## 11. tca-groundtruth follow-ups (none blocking; separate small PRs, not made here)

1. **README/SPEC note:** with the default `gamma_perm = 0.3` the metaorder law is not a pure
   square root (apparent `delta ~ 0.56, Y ~ 0.65`); state it next to the "square-root law"
   claim. (Doc only.)
2. **Optional helper** `orders.alpha_window_return(...)` (the closed form impact-lab uses), so the
   formula lives in one place. impact-lab pins the two with test 4 either way.
3. **Tag** `v0.1.0` on main for a readable pin (optional; the SHA pin works without it).

## 12. Not done

- Almgren-Chriss schedule frontier (standard; tca-groundtruth covers execution schedules).
- Reversion-based identification of alpha vs impact (tca-groundtruth chart (c) already does it) and "control
  for the signal" (one README sentence).
- A full impact surface in (participation, duration); only the crossover law's `eta`
  dependence is shown.
- Size-dependent alpha (`E[alpha | Q]` rising with `Q`), which would push delta up instead of down.
- WLS with `1/T` weights, estimated `sigma_d`, spread/fee-inclusive cost targets.
- Multi-period optimizer (Gârleanu-Pedersen); the optimizer panel is a rolling one-period problem.
- Parameter band through the optimizer (too slow; fixed-turnover band only).
- Cross-impact, overlapping metaorders, herding.
- Non-equity asset classes.

## 13. Parameter sources

| Parameter | Value | Source | Status |
|---|---|---|---|
| Square-root form `I = Y sigma (Q/V)^delta`, delta ~ 0.5 | truth Y = 0.5, delta = 0.5 | Tóth et al. (2011); Almgren et al. (2005) | Form standard; Y = 0.5 is a modelling choice ("of order one"); unverified against the papers |
| Power-law fit on US metaorders | `Y = 0.15 +- 0.01`, `delta = 0.47 +- 0.01` | Zarinelli et al. (2015), arXiv:1412.2152v1 s4.1 | Verified (text); quoted in README, not used as truth |
| Log law shape | `b = 465 +- 33` | same, s4.1 | Verified (text) |
| Log law level | `a = 0.0665` | matched to sqrt at 1% ADV | Modelling choice (paper's `a = 0.028` normalisation unconfirmed) |
| Crossover limits, `eta* ~ 1e-3` | linear below, sqrt above | Bucci et al. (2019), arXiv:1811.05230 eq. 5 and text | Verified (text) |
| Crossover interpolation `sqrt(u/(1+u))` | — | — | Modelling choice |
| Crossover stress | `eta* = 5e-2` | — | Modelling choice, labelled as stress |
| Square-root regime in `phi` | `1e-3 .. 1e-1` | Bucci et al. (2019) | Verified (text) |
| Reversion to ~2/3 of peak | — | Zarinelli et al. (2015) abstract; Farmer et al. (2013) | Verified (Zarinelli text); README context only |
| Propagator gives duration-free sqrt at `delta + beta = 1` | — | Zarinelli et al. (2015) s3.2; Gatheral (2010) | Verified (Zarinelli text); Gatheral unverified |
| Alpha 5-30 bps, half-life 30-240 min, delay 0-60 min | tca-groundtruth defaults | tca-groundtruth | Modelling choice |
| Uninformed orders revert, informed do not | — | Gomes & Waelbroeck (2015) | Not re-checked against the paper |
| `sigma_d` 1-4%, ADV $20m-$500m, 50 names, beta 0.6-1.4 | — | — | Modelling choice |
| Capacity inputs: gross alpha 6%, turnover 4, half-spread 5 bps | — | Illustrative worked example | Illustrative |
| `A*/A_0 = (1+delta)^(-1/delta)`, 4/9 for sqrt | — | Section 3.5 derivation | Derived; tests 13-14 check it |
| Capacity larger than aggregate estimates | qualitative | Frazzini, Israel & Moskowitz (2018), AQR WP | Cited qualitatively; no number quoted |

## 14. FTE estimate (days)

| Module | Days |
|---|---|
| `laws` + `generate` (+ tca-groundtruth cross-check path) and tests 1-5 | 1.25 |
| `fit` + tests 6-8 | 0.75 |
| `experiments` (misspecification, alpha bias, OOS) + tests 9-11 | 1.25 |
| `capacity` (curve, closed-form checks, band) + tests 12-15 | 1.0 |
| `optimizer` (cvxpy, re-optimised curve) + test 16 | 1.0 |
| `report`: charts (a)-(d), summary.csv, Makefile, test 17 | 1.5 |
| README, SPEC sync, references | 1.0 |
| Code review, fixes, CI file | 1.0 |
| **Total** | **8.75** (range 7.5-9.5) |

Cutting (d3) saves about 1 day and drops cvxpy entirely.

## 15. References

- Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). "Direct Estimation of Equity Market Impact." *Risk*.
- Bucci, F., Benzaquen, M., Lillo, F. and Bouchaud, J.-P. (2019). "Crossover from Linear to Square-Root Market Impact." *Physical Review Letters* 122, 108302. arXiv:1811.05230.
- Farmer, J. D., Gerig, A., Lillo, F. and Waelbroeck, H. (2013). "How Efficiency Shapes Market Impact." *Quantitative Finance* 13(11).
- Frazzini, A., Israel, R. and Moskowitz, T. J. (2018). "Trading Costs." AQR working paper.
- Gârleanu, N. and Pedersen, L. H. (2013). "Dynamic Trading with Predictable Returns and Transaction Costs." *Journal of Finance* 68(6).
- Gatheral, J. (2010). "No-Dynamic-Arbitrage and Market Impact." *Quantitative Finance* 10(7).
- Gomes, C. and Waelbroeck, H. (2015). "Is Market Impact a Measure of the Information Value of Trades?" *Quantitative Finance* 15(5).
- Tóth, B. et al. (2011). "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets." *Physical Review X* 1, 021006.
- Zarinelli, E., Treccani, M., Farmer, J. D. and Lillo, F. (2015). "Beyond the Square Root: Evidence for Logarithmic Dependence of Market Impact on Size and Participation Rate." *Market Microstructure and Liquidity* 1(2). arXiv:1412.2152.
- Bouchaud, J.-P., Bonart, J., Donier, J. and Gould, M. (2018). *Trades, Quotes and Prices.* CUP.

## 16. Implementation notes: where the code departs from this SPEC

1. **Test 15 and the capacity finding are reversed.** The spike's "naive fit halves break-even"
   came from an illustrative capacity script that multiplied the impact cost by 12 in addition to
   turnover, which put break-even at about 1% of ADV, where the alpha-biased fit over-prices. With
   the formula in 3.5 and the stated defaults (gross alpha 6%, turnover 4, half-spread 5 bps,
   sigma 1.8-4%), the margin per unit traded (about 145 bps) is only reached at trades of about
   60% of ADV, beyond the sampled range (20%). There the flattened naive fit (delta 0.42)
   under-prices, so it overstates break-even (1.35x the truth). Test 15 asserts
   `naive break-even >= 1.2 x true` and that the median trade at the true break-even lies outside
   the fitted range; the flow-band and band-ordering parts are unchanged. The README states this
   as the capacity finding.
2. **`ratio_at_q`** (test 9, chart (a)) averages the fitted and the true impact over the same
   orders in `[0.8, 1.25] x q`. With `predict(q)` in the numerator, the window at `q = 1e-3` is cut
   by the sampled minimum and a correctly specified fit reads 0.91; the per-order average reads
   0.96. A regression test pins it.
3. **Test 12's small-AUM limit** is checked at $1,000: at $1m the square-root impact still costs
   about 4 bps a year, so "within 1 bp at 1e6" does not hold for the square-root law.
4. **Extra arguments and columns**, all additive: the universe frame carries `half_spread`
   (needed by `net_alpha`, which has no other source for it); `capacity_curve(..., phi_max=inf)`;
   `optimize(..., rebalances=12)` (alpha and cov are annual and are divided by it; `half_spread`
   may be a per-name array); `reoptimised_capacity(..., index_vol=0.012)`; fits frames carry
   `cov_YY, cov_Yd, cov_dd` and `fit_from_row` rebuilds a `PowerFit`; misspecification curves
   carry `z_mean, z_se` for the chart; `alpha_bias` carries delta-method `se_ratio_*`;
   `seed_spread` reports the 20-seed SD and coverage that 3.3 asks the report to show.
5. **Turnover in (d3)** is annual `sum |dw|` (traded value / AUM), the same unit as
   `CapacityParams.turnover` in the fixed-turnover curve, not half of it.
6. **Out-of-sample points** in `misspecification` curves: the five listed sizes (1e-4, 3e-4, 0.3,
   0.5, 1.0); 3.4 says six.
7. **`capacity_points`** also returns nan when net alpha is still positive at $10tn (the log
   law never breaks even inside the bracket).
8. **Dependency pin:** impact-lab uses only tca-groundtruth's order generator, intraday profile
   and per-minute propagator, so the pin can move to any commit that keeps those interfaces.
