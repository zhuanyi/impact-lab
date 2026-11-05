"""Chart labels for m5. Captures the figure that chart_capacity saves."""
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from impact_lab.report import chart_capacity


def test_capacity_chart_labels(tmp_path, monkeypatch):
    figs = []
    orig = Figure.savefig
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: (figs.append(self), orig(self, *a, **k))[1])
    chart_capacity(1, tmp_path / "d.png", n=20_000, quick=True)
    ax1, ax2, ax3 = figs[0].axes[0], figs[0].axes[1], figs[0].axes[2]
    assert ax1.get_ylabel() and ax2.get_ylabel() and ax3.get_ylabel()
    assert "net alpha" in ax3.get_ylabel()
    legend2 = ax2.get_legend()
    assert legend2 is not None and legend2._loc != 3          # not "lower left": it hid the curves
    labels3 = [t.get_text() for t in ax3.get_legend().get_texts()]
    assert any("turnover" in t for t in labels3)
