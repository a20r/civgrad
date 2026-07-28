version: v0.4 — audited numbers + the policy section

# The civilization gradient

*The living essay of the [civgrad](https://github.com/a20r/civgrad) project.
Versioned like the methodology: see `essay/VERSION` and `essay/CHANGELOG.md`.
Every number in it traces to the repo's frozen baseline or its committed
audits. The canonical, fullest version of this essay now lives on
[the blog](https://a20r.github.io/blog/posts/the-civilization-gradient/) —
it carries two extra sections (the formal machinery, and a reflection on
the method) and the full audit and policy tables inline, which this
repo-site copy summarizes in prose (its renderer has no table support);
POLICY.md, SENSITIVITY.md, and PRICE_EXPERIMENT.md in the repo root are
the regenerable sources of every number.*

## 1. An island with no off-island

I was in the middle of the Anasazi chapter of Jared Diamond's *Collapse* when
the thought arrived, uninvited, about a different chapter: Easter Island. To a
Rapa Nui islander, the island effectively *was* the world — a closed system
ringed by an ocean that might as well have been vacuum. A few thousand people
to that island is, at relative scale, a few billion people to this planet. The
ratio changes; the geometry doesn't.

Before leaning on the metaphor, I should state it honestly, because the famous
version of the story is contested. Terry Hunt and Carl Lipo have argued for
years that the ecocide narrative is mostly wrong: Polynesian rats, not
profligate islanders, drove the deforestation by eating palm seeds, and the
demographic catastrophe came *after* European contact, from disease and slave
raiding — not before it, from self-inflicted resource collapse. Diamond's
flagship case is shakier than his chapter admits, and this essay keeps both
readings on the table.

Here's the thing: the metaphor survives either reading. What matters isn't
*why* the trees disappeared. It's that when they did, there was nowhere to
import canoes from. A closed system has to solve its problems with what's
inside it, at the speed its internal machinery allows. Earth has no
off-island.

So this isn't an essay about running out of things. It's about the question
the island geometry actually raises: in a closed, hyperconnected economy,
*where precisely is it brittle* — and if you had one marginal dollar of
resilience to spend, where should it go?

## 2. Brittleness is not depletion

The collapse conversation fixates on depletion — peak this, running out of
that. But for most industrial inputs, aggregate abundance is fine. The scary
structure is different: **concentration times rebuild time**. The supply
graph is absurdly non-redundant at a small number of nodes, and some of those
nodes would take a decade to rebuild.

A short tour of the neighborhood this project models. Semiconductor lasers
breathe neon, which is a byproduct of Soviet-era steel-mill air separation,
and until February 2022 roughly half of the world's chip-grade purification
ran through two Ukrainian companies — one of them in Mariupol. Gallium is not
rare at all; it's sitting in bauxite tailings on every continent. It's just
that ~98% of *refining* happens in China, because everyone else stopped
bothering — which is how a December 2024 export ban could turn a geological
non-issue into a supply crisis. The chokepoint is a license regime, not the
Earth's crust. Every advanced chip passes through extreme-ultraviolet
lithography machines that exactly one company, ASML, knows how to build, a
few dozen a year — and underneath ASML sits Carl Zeiss SMT, whose mirrors are
polished to sub-nanometer figure by three decades of institutional knowledge
that exists in one place. A monopoly *under* the monopoly. In 1993, about 60%
of the world's epoxy molding compound for chip packaging came from a single
Sumitomo plant, right up until it exploded. And roughly 90% of leading-edge
logic fabrication sits on one island, which also hosts the shipping strait,
which means one bad month in one place hits fabrication *and* transport as a
single correlated event.

Notice the two different species of chokepoint. Gallium refining is a
**capacity monopoly**: annoying, but rebuildable in two or three years of
determined investment. Zeiss and TSMC are **knowledge monopolies**: decade-
plus rebuild times, because what's concentrated isn't machines but learning.
Brittleness, throughout this project, means concentration × rebuild-time ×
co-location. Depletion barely makes the list.

## 3. Why Petri nets

Supply chains usually get modeled as graphs — nodes, edges, flows. But the
things that actually break are *stocks* and *concurrency*: buffers draining,
processes waiting on each other, matter being conserved whether you like it
or not. Petri nets give you all three natively. Places hold tokens (stocks);
transitions fire when their inputs are present (processes); conservation is
an invariant you can check rather than an assumption you hope holds. Two arc
types do surprising amounts of work: a **read arc** lets a transition require
a token without consuming it — which is exactly what capital equipment is, a
catalyst — and an **inhibitor arc** lets a token *block* a transition, which
is exactly what a policy is. In this model, China's gallium export ban is
literally one token sitting on one place. Sanctions relief is removing it.

The centerpiece of the discrete model (`core/net.py`) is a failure mode that
flow networks cannot even state. From the initial state, a single legal
firing — one EUV tool wearing out before any chips have been banked — reaches
a state from which no chip can *ever* be made again, while raw materials keep
flowing forever. Mining continues. Refining continues. Everything moves;
nothing can be accomplished. The formal name is livelock; the honest name is
**busy futility**. And it rhymes with the island: Rapa Nui society kept
functioning for generations after the last tree fell. What it lost was the
ability to build ocean-going canoes. Collapse doesn't have to look like
silence. It can look like a perfectly normal Tuesday on which a certain kind
of thing has quietly become impossible.

## 4. The map and the fog

Nobody can model the global supply chain, and this project doesn't pretend
to. It borrows the fog of war from strategy games and makes it a modeling
primitive. The formalism is hierarchical Petri nets: an **oracle** is a
substitution transition — a black box that declares its interface (which
stocks flow in and out) and a one-line contract, and nothing else
(`map/_oracles/`). The demo below draws oracles as literal fog, because
that's what they are: territory the map admits it hasn't surveyed.

The epistemic point got demonstrated twice in this repo's own short history,
in opposite directions. Fog hid *extra fragility*: the first cut treated
"equipment manufacturing" as one box, and expanding it exposed the Zeiss
optics monopoly — a deeper single point of failure than ASML itself, with a
longer rebuild time. And fog manufactured *fake resilience*: unexpanded
source transitions had no upstream constraints, so they behaved as infinite
faucets, and the deadlock detector triumphantly reported that collapse was
impossible. Same cause, opposite errors. Fog is not conservative and it is
not optimistic; **it is wrong in whichever direction its hidden structure
points**. The only honest response is to draw it, label it, and remember
that every conclusion is conditional on where the fog currently sits.

## 5. The gradient of collapse

Then comes the move the project is named for. Relax the discrete net into a
continuous one — real-valued stocks, transitions as flow rates with
saturation kinetics — and the whole system becomes a piecewise-smooth ODE
that runs under JAX. Which means it's differentiable. Which means "where
should civilization's marginal dollar go" stops being a panel discussion and
becomes a vector you can compute: the derivative of disrupted throughput with
respect to every capacity and every stockpile in the net
(`core/gradients.py`). Investing along that vector is gradient ascent on
resilience. The magnitudes are only as good as the calibration — the frozen
numbers live in `validation/baseline_outputs.txt`, and "trust the signs and
rankings, distrust the magnitudes" is now measured rather than argued: a
500-draw audit that lets every confidence-C parameter be wrong by a factor
of two keeps the two big negative signs in 85–92% of draws (100% under pure
disruption-prior fog) and keeps the top of the ranking inside the
production/equipment complex in 91–99% — while *which* node leads is
fog-conditional and fine-grained rank order dissolves (`SENSITIVITY.md`
holds the tables) — but two results deserve to be told straight.

First, the **negative shipping gradient**: −2.68 in the single-scenario run,
−1.63 averaged over the disruption prior. After a shock destroys fabrication
tools, adding *more* shipping capacity makes total output *worse* — because
exports drain the packaged chips that the equipment-rebuilding loop needs.
The model derived wartime rationing from pure topology. Nobody asked it to.
When the means of production are damaged, the correct policy is to hold the
product back from consumers and feed it to the machines that make machines —
and that fell out of a derivative.

Second, the **boneyard result**: worn-out tools are among the most valuable
stockpiles in the entire net — 0.85 against 0.70 for *working* spare tools
in the single-scenario run. The audit took the stronger version of this
claim away: average over the disruption prior and the two flip (0.43 vs
0.51), and under parameter fog worn-on-top survives in only about a fifth
of draws. What survives every regime is that a refurbishable boneyard is
*comparable in value* to pristine inventory, because the refurbishment path
converts junk back into capacity at a fraction of the cost of building new
— still a strong claim about an industry that scraps old fab equipment
routinely, and quietly shreds resilience capital doing it. The US Air Force
figured this out decades ago in the Arizona desert.

<!-- demo -->

## 6. Making recovery emergent

The first version of this model had a dirty secret: recovery time was an
input. The 2022 neon shock replay recovered in about the historical window
because I *told it* alternative supply ramps in nine months. Validation
where you feed in the answer isn't validation. So v1 replaces the imposed
curve with an adaptation law: every transition's capacity grows at a rate
α when it's scarce, and recovery time becomes an *output*.

What counts as "scarce" took five instructive failures to get honest, and
each one was forced by data rather than taste. **One**: the infinite-faucet
bias from the discrete era — unconstrained sources faked away deadlock —
which became the fog doctrine of §4. **Two**: the wrong observable —
Sumitomo 1993 was invisible through fab throughput because packaging sits
*downstream* of the fab; the historical pain was in deliveries, and the
model has to be read at the place history was measured. **Three**: buffers
mask the alarm — a scarcity signal based on stock levels meant a fat
stockpile suppressed the warning until the buffer was gone, and the dip
arrived late anyway. Real markets are forward-looking: neon prices went up
9× within weeks in 2022 while stockpiles were still full, because traders
compute *runway* — months of cover at the current net drain. The fixed
signal fires when runway falls below a planning horizon. **Four**:
equilibrium hides shortage — runway alone goes silent once the system
stabilizes *into* scarcity, because nothing is draining anymore; the model
settled into permanent depression with the alarm off. The fix is a second
signal term anchored to unfilled pre-crisis demand. **Five**: adaptation
cannibalized the scarce good — at full urgency, tool-building outbid
deliveries for the very chips that were scarce and drove shipments to
zero. That's a real phenomenon (fabs need chips to build fabs; 2021 in
miniature) but its magnitude was an artifact of having no prices, so the
law now distinguishes *restoring* destroyed capacity (driven by flow
deficit) from *expanding* beyond baseline (which requires sustained runway
evidence).

α was fitted once, on the neon training event — it lands at 0.06, roughly
"6% of baseline capacity re-buildable per month at full alarm," a plausible
industrial number — and then frozen. Everything after that is holdout.

## 7. The scorecard

The protocol: structure, constants, and the calibration rule are frozen;
per-event inputs are documented historical facts with citations; predictions
are registered before running. `SCORECARD.md` is the source of truth and is
regenerated by CI, misses included. The results: **neon 2022** passes under
both protocols — with the documented ~6-month stockpiles (an industry lesson
from the 2014 Crimea price spike), no fab stoppage, matching history; the
same run *without* stockpiles takes a ~17% hit, which is the model pricing
what that lesson was worth. **Tōhoku 2011** passes: two months of wafer
inventory dwarfs the integrated deficit — 1.0 month-equivalent under the
imposed ramp, 0.6 as measured under the adaptation law (POLICY.md
tabulates all six replays). That inequality
— buffer-months versus the integral of the deficit while capacity ramps — is
the single mechanism that decides nearly every event in the suite.
**Photoresist 2019** passes as the non-event it was; and the counterfactual
run of what everyone *feared* (a real 90% cut) yields a 64% dip — the model
quantifying what the panic implied and the licensing regime prevented.

And then **Sumitomo 1993**, which the model misses twice, in opposite
directions, and which is the most valuable row on the board. Under v0 it's
invisible (−1.7% — the wrong-observable failure). Under v1, measured at the
right observable, it predicts a 57.6% delivery collapse that *never
recovers* within the 72-month horizon — against a historical record of
"price spike, brief pain, no catastrophe." The never-recovers part is the
interesting part, and tracing the trajectory shows the trap precisely: the
outage piles up unpackaged chips upstream, and that bloated buffer *masks
the destroyed capacity* — packaging flow returns to its pre-crisis
reference while 8% of its capacity is still missing, so the restoration
alarm falls silent; downstream, deliveries settle at nine-tenths of
pre-shock with nothing draining, so the runway alarm is silent too. A
self-consistent depressed equilibrium with every alarm quantitatively off —
the wrong-observable failure again, this time *inside the control law*. The
stocks are misallocated — one buffer bloated, the loop it feeds starving —
and reallocation between competing uses is a price mechanism this model
does not yet have. That is hysteresis: the shock knocks the system into a
worse basin that is locally stable. Uninvited, Diamond's actual thesis
showed up in the mathematics — **collapse as attractor, not event**. For
1993 the prediction is simply wrong; the real economy reallocated its way
out in months. But a formalism that can *express* collapse-as-attractor is
exactly what a project with this name requires. It just needs the escape
mechanism reality has, and that gap is a failing test on the scorecard —
not a footnote. (Two results now sit on the other side of that test. The
sensitivity sweep says the miss is structural: 95% of 500 fog draws miss it
too, so no citation will fix it. And the price experiment says the
diagnosis was right: teach the model to see a **processing margin** — a fat
input beside a starved output, the formal shadow of the 1993 epoxy price
spike — and let that margin restore destroyed capacity toward baseline, and
the attractor dissolves: recovery in about ten months, the passing replays
untouched. The same experiment's cautionary arm: rationing the contested
stock by the gradient itself drives deliveries to zero, because a gradient
computed on one catastrophe is the wrong price list for another.
`PRICE_EXPERIMENT.md` has both.)

Which yields the caveat that belongs in bold on everything this model
outputs. It has no price-mediated allocation, no substitution, no
design-around response, and therefore it **systematically overstates how
deep shocks bite and how long they last**. Trust where it says fragility
concentrates — the rankings, the SPOFs, the signs of the gradients.
Distrust how bad it says things get.

<!-- scorecard -->

## 8. The marginal dollar, cashed out

The question in §1 was where one marginal dollar of resilience should go.
Having audited what the model can and cannot license, here is its answer —
each conclusion wearing its measured robustness, all magnitudes under the
standing caveat, full tables in the repo's `POLICY.md`.

**Buffers first — and they have a sizing rule.** A shock bites only when
buffer-months of cover fall short of the integrated capacity deficit while
supply rebuilds; that one inequality decides all six replay rows. The
industry's post-2014 neon stockpile (~6 months) sat just above the measured
deficit (5.7 month-equivalents) — the 2014 lesson bought almost exactly the
right amount of insurance. So: at single-source chokepoints, hold — or at
least require disclosure of — months-of-cover sized against plausible
outages. It is the cheapest resilience anywhere in the model.

**The marginal capacity dollar goes upstream — never to logistics.** In
91–99% of fog draws the top-gradient capacity is fab, equipment-making, or
a fab-input source; the equipment loop is the modal leader; shipping led in
one draw out of a thousand. Which single node wins, the audit refuses to
say — but industrial-policy dollars pointed at shipping, packaging, or
consumption-side capacity are pointed the wrong way in nearly every world
consistent with the map.

**Post-shock, allocation beats capacity — and static priority lists are
dangerous.** The negative shipping gradient (sign-stable in 85–89% of
parameter-fog draws, 100% under prior fog) is the model deriving wartime
rationing from topology. But the price experiment's cautionary arm shows
that a *frozen* priority list applied in the wrong crisis drives deliveries
to zero. Crisis authority should reallocate on current information, per
event — or get out of the way of the prices that re-solve that problem
daily.

**Stop shredding the boneyard.** Comparable value to pristine spares, at
scrap prices, in every fog regime — and in the tail analysis it is the
refurbishment loop that carries the system through the Zeiss-knockout
scenario. A decommissioned-tool registry and refurbishment capacity are
cheap resilience buys.

**Tail-risk planners buy fab redundancy.** Weight the objective toward the
worst 30% of the prior and the ranking concentrates on the Taiwan-fab and
gallium scenarios; no plausible buffer covers a 60-month ramp, so the tail
answer is geographic redundancy for the decade-rebuild nodes. (The optics
monopoly's true cost lives beyond the model's 72-month horizon — pricing it
honestly is named future work, not a comfort.)

All of it conditional on a confidence-C toy slice — and the audit's own
conclusion is that citations, not more cleverness, are what would sharpen
the rankings. That is exactly what the map's provenance gates are for.

## 9. What this is and isn't

This is not a forecast, not a claim that collapse is coming, and not a
policy tool you should point at the real world without reading its
scorecard first — §8's conclusions come with their conditions attached for
exactly that reason. Every parameter in the map carries provenance
metadata; every one is still confidence-C, though the load-bearing ones now
carry verified citations awaiting a reviewer's upgrade. It is a lens — and
the honest surprise of the project is how
much structure a toy lens resolves. Brittleness is concentration times
rebuild time, not depletion. Buffer-months versus ramp-time integrals
decide who feels a shock. Boneyards are resilience capital, and adding
capacity in the wrong place can carry a negative sign. Fog is wrong in
unknown directions, so declare it. And an honest scorecard beats an
impressive demo: the red Sumitomo row is the most valuable pixel on this
site, because it's the model telling you exactly where not to trust it.

This started as: a guy reads the Anasazi chapter and wonders, for a closed
island with no off-island, which way is up. It was built in one long
back-and-forth with an AI (the receipts are in the footer), then hardened
by the discipline any model deserves — frozen parameters, registered
predictions, public misses. The gradient points somewhere. Mostly, I built
this to learn how to read it. Play with the demo above; every number it
shows traces back to the repo; and if you happen to know one of the fogged
territories — resin chemistry, phosphate logistics, optics — the fog is
labeled, and the gate is open.

---

*Design history: the model and its five instructive failures were worked out
in one conversation (July 2026). Full transcript export pending —
`docs/TRANSCRIPT.md` holds the placeholder and the [interim link](https://claude.ai/share/04279425-3700-447c-82c8-9d3861471785).
When the export lands, this footer links it directly.*
