"""Adaptation regression against the frozen baseline.

Expected values are the gsc_adapt.py numbers in
validation/baseline_outputs.txt (jax 0.10.2, CPU, x64) — the regression
oracle. Tolerance is 0.1 percentage point / 0.1 month: tight enough to catch
any model change, loose enough for cross-machine float noise. If these fail
after an intentional methodology change, re-freeze via the full protocol
(PLAN.md §4), never by editing the expectations casually."""

import math

import pytest

from core import adaptation


CASES = [
    # (buffer place, months, transition, kill, observe, dip_pct, rec_months)
    ("neon w/ 6mo stockpile", "Ne_purified", 6.0, "Purify_Ne", 0.5, "fab", -0.7, 0.0),
    ("tohoku 2011",           "Wafers",      2.0, "WaferSupply", 0.25, "fab", -0.6, 0.0),
    ("sumitomo 1993",         "Chips",       1.5, "Package", 0.60, "ship", 57.6, math.inf),
    ("photoresist realized",  "Ne_purified", 2.0, "Purify_Ne", 0.15, "fab", -0.7, 0.0),
    ("photoresist feared",    "Ne_purified", 2.0, "Purify_Ne", 0.90, "fab", 64.0, 13.7),
]


@pytest.mark.parametrize("label,place,months,trans,kill,observe,exp_dip,exp_rec",
                         CASES, ids=[c[0] for c in CASES])
def test_adaptive_replay_matches_baseline(label, place, months, trans, kill,
                                          observe, exp_dip, exp_rec):
    d, r = adaptation.run(place, months, trans, kill, adaptation.ALPHA,
                          observe=observe)
    assert abs(100 * d - exp_dip) < 0.1, f"{label}: dip {100*d:.2f} vs {exp_dip}"
    if math.isinf(exp_rec):
        assert math.isinf(r), f"{label}: expected no recovery inside the horizon"
    else:
        assert abs(r - exp_rec) < 0.1, f"{label}: recovery {r} vs {exp_rec}"


def test_alpha_is_the_frozen_value():
    assert adaptation.ALPHA == 0.06, "frozen parameter moved outside a methodology change"
