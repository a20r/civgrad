"""Invariants for the analysis package (analysis/).

The load-bearing property: the parameterized twin in analysis/engine.py is
the SAME model as the frozen one in core/ at the unperturbed point. If the
twin drifts, every number in SENSITIVITY.md / PRICE_EXPERIMENT.md /
POLICY.md audits the wrong model — so the selfcheck is a hard test, not a
convention. The experiment invariants below pin the price experiment's two
qualitative results (documented in PRICE_EXPERIMENT.md) so silent behavior
changes in the analysis code fail here first.
"""

import jax.numpy as jnp
import pytest

from core.continuous import cap0
from analysis import engine
from analysis.engine import K_SAT0, UTILIZATION0, ALPHA0, HORIZON0


def test_engine_twin_reproduces_frozen_baseline():
    checked = engine.selfcheck()
    # spot-check the values the docs quote
    assert checked["sumitomo_1993"][0] == pytest.approx(57.6, abs=0.05)
    assert checked["sumitomo_1993"][1] == float("inf")


def test_margin_law_dissolves_sumitomo_attractor_and_spares_passes():
    from analysis.price_experiment import run_event
    d, r = run_event("sumitomo_1993", "margin")
    # the attractor dissolves: finite recovery, same transient depth
    assert r != float("inf") and r < 24.0
    assert d == pytest.approx(57.6, abs=0.5)
    # a passing event is untouched by the margin term (its restoration is
    # never margin-gated at these depths)
    d0, r0 = run_event("tohoku_2011", "frozen")
    d1, r1 = run_event("tohoku_2011", "margin")
    assert d1 == pytest.approx(d0, abs=0.1)
    assert r1 == pytest.approx(r0, abs=0.1)


def test_equal_weights_rationing_reproduces_frozen_law():
    """The arm-2 allocation machinery must be a no-op at equal weights —
    otherwise its 'worse collapse' result is an artifact of the plumbing."""
    from analysis.price_experiment import run_event, PKG_CONSUMERS
    w = jnp.full(len(PKG_CONSUMERS), 1.0 / len(PKG_CONSUMERS))
    d0, r0 = run_event("sumitomo_1993", "frozen")
    d1, r1 = run_event("sumitomo_1993", "rationed", weights=w)
    assert d1 == pytest.approx(d0, abs=0.05)
    assert r0 == r1 == float("inf")
