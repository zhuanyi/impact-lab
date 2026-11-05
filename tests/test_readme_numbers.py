"""Every number in the README traces to reports/summary.csv."""
from pathlib import Path

import pandas as pd

README = Path(__file__).resolve().parents[1] / "README.md"
SUMMARY = Path(__file__).resolve().parents[1] / "reports" / "summary.csv"

# (readme_substring, metric_key, format_spec)
ROWS = [
    ("$15.6bn", "d1_naive_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("$11.5bn", "d1_true_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("$11.6bn", "d1_flow_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("$5.1bn", "d1_true_dollar_optimal", lambda v: f"${v / 1e9:.1f}bn"),
    ("$6.8bn", "d1_naive_dollar_optimal", lambda v: f"${v / 1e9:.1f}bn"),
    ("63%", "d1_true_break_even_median_phi", lambda v: f"{v * 100:.0f}%"),
    ("120%", "d1_true_break_even_phi_mean", lambda v: f"{v * 100:.0f}%"),
    ("365%", "d1_true_break_even_phi_max", lambda v: f"{v * 100:.0f}%"),
    ("40% of the names", "d1_true_break_even_share_phi_gt1",
     lambda v: f"{v * 100:.0f}% of the names"),
    ("60% of the cost", "d1_true_break_even_cost_share_phi_gt1",
     lambda v: f"{v * 100:.0f}% of the cost"),
    ("$21.6bn", "d4_adv_weighted_true_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("$21.9bn", "d4_adv_weighted_flow_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("$28.2bn", "d4_adv_weighted_naive_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("$13.8bn", "d4_eta_max_0.25_crossover_stress_true_break_even", lambda v: f"${v / 1e9:.1f}bn"),
    ("1.81", "log_ratio_point_at_0.001", lambda v: f"{v:.2f}"),
    ("delta 0.36", "alpha_delta", lambda v: f"delta {v:.2f}"),
    ("90%", "bias_removed", lambda v: f"{v * 100:.0f}%"),
]


def test_readme_numbers_trace_to_summary():
    readme_text = README.read_text()
    summary = pd.read_csv(SUMMARY)
    values = dict(zip(summary["metric"], summary["value"], strict=True))
    failures = []
    for substring, key, fmt in ROWS:
        if key not in values:
            failures.append(f"  {substring!r}: key {key!r} not in summary.csv")
            continue
        formatted = fmt(values[key])
        if formatted not in substring:
            failures.append(
                f"  {substring!r}: expected formatted value {formatted!r} from key {key!r}")
        if substring not in readme_text:
            failures.append(f"  {substring!r}: not found in README.md")
    assert not failures, "README number mismatches:\n" + "\n".join(failures)
