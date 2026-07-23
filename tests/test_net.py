"""Discrete-net invariants (core/net.py). These tests pin the structural
facts that the essay and scorecard rely on; they are not a modeling test
suite."""

from core.net import build_semiconductor_slice


def fresh():
    return build_semiconductor_slice()


def test_spofs_for_goods_are_the_main_chain():
    net = fresh()
    assert net.spofs("Goods") == ["Fab", "Package", "Ship_TW_Strait"]


def test_doomed_state_is_livelock_not_deadlock():
    """Wear out the last tool before any chips are banked: chips become
    forever unreachable while the net still has enabled transitions —
    busy futility, the doomed-state demo from the baseline."""
    net = fresh()
    net.places["EUV_tools"].tokens = 1
    net.places["Chips_fabbed"].tokens = 0
    net.places["Chips_packaged"].tokens = 0
    net.transitions["Wear_EUV"].fire()
    assert net.places["EUV_tools"].tokens == 0
    assert not net.can_reach_tokens_in("Chips_fabbed")
    assert net.enabled_transitions(), "raw materials keep flowing: livelock, not deadlock"


def test_unit_transitions_conserve_tokens():
    """Every 1-in/1-out transition conserves total token count when fired."""
    net = fresh()
    unit = ["Purify_Ne", "Package", "Ship_TW_Strait", "Consume", "Recycle", "Wear_EUV"]
    for name in unit:
        t = net.transitions[name]
        for p in t.inputs:
            p.tokens = max(p.tokens, 1)
        for p in t.inhibitors:
            p.tokens = 0
        before = sum(p.tokens for p in net.places.values())
        assert t.enabled(), name
        t.fire()
        after = sum(p.tokens for p in net.places.values())
        assert after == before, f"{name} should conserve tokens"


def test_export_ban_inhibits_refining():
    net = fresh()
    net.places["Ga_byproduct"].tokens = 1
    assert net.transitions["Refine_Ga"].enabled()
    net.places["ExportBan_CN_US"].tokens = 1
    assert not net.transitions["Refine_Ga"].enabled()
