# analysis/ — off-to-the-side research analyses (sensitivity audit, price
# experiment, policy synthesis). Nothing in core/ or validation/ imports from
# here; the frozen model and its gates are unaffected by anything in this
# package (AGENTS.md rule 2). Every engine function used by these analyses is
# asserted against its frozen counterpart at the unperturbed point
# (analysis/engine.py::selfcheck, tests/test_analysis.py).
