import time
from pathlib import Path

import pandas as pd

from impact_lab.report import make_report


def test_report_smoke(tmp_path):
    t = time.perf_counter()
    paths = make_report(tmp_path, seed=1, quick=True)
    elapsed = time.perf_counter() - t
    assert len(paths) == 5
    assert all(Path(p).exists() for p in paths)
    pngs = [p for p in paths if p.endswith(".png")]
    assert len(pngs) == 4
    for p in pngs:
        assert Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    s = pd.read_csv(paths[-1])
    assert list(s.columns) == ["chart", "metric", "value"]
    assert set(s["chart"]) == {"a", "b", "c", "d"}
    assert elapsed < 60
