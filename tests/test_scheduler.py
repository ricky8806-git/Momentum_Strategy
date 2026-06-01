"""Tests for generate_report() in scheduler.py."""
import os
import tempfile
import pandas as pd
from scheduler import generate_report


def _eligible_pass_df():
    return pd.DataFrame([
        {"ticker": "AMD",  "score": 0.42, "passes_filter": True,
         "ret_long": 0.48, "ret_short": 0.22, "rank": 1},
        {"ticker": "INTC", "score": 0.38, "passes_filter": True,
         "ret_long": 0.43, "ret_short": 0.18, "rank": 2},
        {"ticker": "MU",   "score": 0.31, "passes_filter": True,
         "ret_long": 0.35, "ret_short": 0.15, "rank": 3},
    ])


def test_generate_report_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "2026-06-02_rebalance_report.md")
        generate_report(
            report_path=path,
            signal_day=pd.Timestamp("2026-05-30"),
            nav=127294.86,
            n_universe=503,
            n_scanned=480,
            eligible_pass=_eligible_pass_df(),
            entries=[{"ticker": "INTC", "shares": 69, "price": 121.45}],
            exits=[{"ticker": "LITE", "shares": 7, "price": 936.70, "reason": "filter_exit"}],
            held_tickers=["AMD", "MU"],
        )
        assert os.path.exists(path)
        text = open(path).read()
        assert "INTC"       in text
        assert "LITE"       in text
        assert "AMD"        in text
        assert "127,294"    in text
        assert "2026-05-30" in text
        assert "503"        in text
        assert "480"        in text


def test_generate_report_no_trades():
    """Report generates cleanly when there are no buys or sells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.md")
        generate_report(
            report_path=path,
            signal_day=pd.Timestamp("2026-05-30"),
            nav=100000.0,
            n_universe=503,
            n_scanned=470,
            eligible_pass=_eligible_pass_df(),
            entries=[],
            exits=[],
            held_tickers=["AMD"],
        )
        text = open(path).read()
        assert "No exits"   in text or "no exits"   in text.lower()
        assert "No entries" in text or "no new"     in text.lower()
