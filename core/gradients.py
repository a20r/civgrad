"""
core/gradients.py — capacity/stockpile gradients; expected objectives.

  d(throughput)/d(capacity_i)     -> where to invest in CAPACITY
  d(throughput)/d(initial stocks) -> where to invest in STOCKPILES

The ranked gradient vector is the project thesis made literal:
marginal resilience per marginal unit of investment.

expected_throughput averages total fab throughput over a crude prior of
disruption scenarios, so investment rankings stop being hostage to one
hand-picked catastrophe. Tail objectives (CVaR over the scenario prior)
are future work — see METHODOLOGY.md §3.

`python3 -m core.gradients` reproduces the gsc_grad.py baseline demo
(see validation/baseline_outputs.txt).
"""

import jax
import jax.numpy as jnp

from core.continuous import (PLACES, TRANSITIONS, T_IDX, cap0, x0,
                             simulate, simulate_recovery)

throughput = lambda cap, x0: simulate(cap, x0, kill_frac=0.9)[0]

# crude prior over disruptions (prob, transition hit, severity, rebuild tau)
SCENARIOS = [
    (0.30, T_IDX["Purify_Ne"],  0.5, 9.0),    # regional war hits inert gas
    (0.25, T_IDX["Ship_Strait"],0.8, 12.0),   # strait blockade
    (0.20, T_IDX["Fab"],        0.9, 60.0),   # Taiwan fab loss (slow rebuild)
    (0.15, T_IDX["Refine_Ga"],  0.9, 30.0),   # export ban hardens
    (0.10, T_IDX["OpticsMfg"],  0.9, 120.0),  # Zeiss knocked out (near-unrebuildable)
]


def expected_throughput(cap, x0, scenarios=None, t0=6.0):
    """loss = sum_s p_s * throughput(scenario_s) over a disruption distribution."""
    scenarios = SCENARIOS if scenarios is None else scenarios
    tot = 0.0
    for prob, tidx, kill, tau in scenarios:
        t, _ = simulate_recovery(cap, x0, tidx, kill, tau, t0=t0)
        tot = tot + prob * t
    return tot


if __name__ == "__main__":
    base, _ = simulate(cap0, x0, 0.0)
    hit,  _ = simulate(cap0, x0, 0.9)
    print(f"chip throughput, no disruption:      {base:8.2f}")
    print(f"chip throughput, 90% EUV destroyed:  {hit:8.2f}   "
          f"({100*(1-hit/base):.0f}% civilizational haircut)\n")

    g_cap = jax.grad(throughput, argnums=0)(cap0, x0)
    g_stk = jax.grad(throughput, argnums=1)(cap0, x0)

    print("d(throughput)/d(capacity) — where to invest in CAPACITY:")
    for i in jnp.argsort(-g_cap):
        print(f"   {float(g_cap[i]):8.3f}  {TRANSITIONS[int(i)][0]}")

    print("\nd(throughput)/d(initial stock) — where to invest in STOCKPILES:")
    for i in jnp.argsort(-g_stk):
        print(f"   {float(g_stk[i]):8.3f}  {PLACES[int(i)]}")
