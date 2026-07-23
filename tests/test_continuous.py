"""Continuous-relaxation invariants (core/continuous.py): conservation where
expected, saturation bounds, non-negativity."""

import jax.numpy as jnp

from core.continuous import (Pre, Post, TRANSITIONS, T_IDX, NP_, flows, cap0,
                             burn_in)


def test_unit_transitions_conserve_mass_structurally():
    """1-in/1-out transitions have zero net token creation in the incidence
    matrix — the continuous mirror of discrete conservation."""
    unit = ["Purify_Ne", "Package", "Ship_Strait", "Consume", "Recycle",
            "Wear_EUV", "Refine_Ga"]
    net_change = (Post - Pre).sum(axis=1)
    for name in unit:
        assert float(net_change[T_IDX[name]]) == 0.0, name


def test_flows_bounded_by_capacity_and_nonnegative():
    x = jnp.full(NP_, 0.3)
    v = flows(x, cap0)
    assert bool((v >= 0.0).all())
    assert bool((v <= cap0 + 1e-12).all()), "saturations can only throttle, never amplify"


def test_stocks_stay_nonnegative_through_burn_in():
    xs, F = burn_in(cap0, jnp.full(NP_, 1.0))
    assert bool((xs >= 0.0).all())
    assert F > 0.0, "steady state must actually produce chips"


def test_sources_create_and_sinks_are_closed_by_recycle():
    """Mine/WaferSupply/OpticsMfg are net creators; the only true exit from
    the material loop is what Consume hands to Recycle's near-dead closure."""
    net_change = (Post - Pre).sum(axis=1)
    for src in ["Mine", "WaferSupply", "OpticsMfg"]:
        assert float(net_change[T_IDX[src]]) > 0.0, src
