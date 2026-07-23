version: v0.1 — scaffold; stubs only, prose pending

# The civilization gradient

*[This is the living essay. Every section below is a 2–4 sentence STUB naming
what the section will argue, with its sources in the repo — the prose is the
author's to write. Versioned like the methodology: see `essay/VERSION` and
`essay/CHANGELOG.md`.]*

## 1. An island with no off-island

*[stub]* Rereading Diamond's *Collapse* and the Easter Island chapter that
launched a thousand metaphors — and stating the metaphor honestly: Hunt and
Lipo's challenge (rats, disease, contact — not self-inflicted ecocide) has
real force, and the essay keeps both readings on the table. What survives
either reading is the geometry: an island economy has no off-island to import
resilience from, and at planetary scale there is no off-island. That is the
sense in which the metaphor is load-bearing here, and the only sense claimed.

## 2. Brittleness is not depletion

*[stub]* The collapse conversation fixates on running out; the scarier
structure is concentration times rebuild time. The tour: semiconductor-grade
neon purification halved by one war (2022), gallium refining concentrated in
one country's export-license regime, EUV optics in one Zeiss facility's three
decades of know-how, advanced packaging resin — once — mostly in one plant
that exploded in 1993. Sources: `map/semiconductors/net.yaml` (rebuild_years,
regions), `validation/events/*.yaml` (the documented shocks).

## 3. Why Petri nets

*[stub]* Stocks, concurrency, conservation — and failure modes richer than "a
node went down." The centerpiece is the livelock in `core/net.py`: wear out
the last EUV tool before any chips are banked and raw materials keep flowing
forever while no chip can ever be made again — busy futility, a failure that
flow-network models cannot even state. 2–3 sentences on read arcs (equipment
enables without being consumed) and inhibitor arcs (policy as a token).

## 4. The map and the fog

*[stub]* Unmapped territory is declared, not ignored: oracles carry an
interface and a contract string and nothing else (`map/_oracles/`). The
epistemic point, demonstrated twice in this repo's own history: fog hid extra
fragility (the Zeiss monopoly sits a level *deeper* than ASML) and fog
manufactured fake resilience (source transitions with no upstream constraint
acted as infinite faucets). Fog is wrong in unknown directions — that is why
the demo draws it literally.

## 5. The gradient of collapse

*[stub]* Make throughput differentiable and "where should the marginal dollar
go" becomes a vector you can read (`core/gradients.py`;
`validation/baseline_outputs.txt` for the frozen numbers). Two results to
tell straight: the *negative* shipping gradient (−2.68 single-scenario,
−1.63 expected: adding shipping capacity drains buffers faster than
downstream can absorb — rationing emerges from the math, uninvited), and the
boneyard result (the top stockpile gradient is worn-out tools, 0.85, beating
working tools at 0.70: a refurbishable boneyard is worth more than pristine
spares).

<!-- demo -->

## 6. Making recovery emergent

*[stub]* Imposed recovery curves assume the answer; the adaptation law earns
it — capacity responds to runway scarcity and unfilled demand, ALPHA fitted
once on neon 2022 and frozen (`core/adaptation.py`, METHODOLOGY.md §4). Told
straight, the five instructive failures from the design history: the
infinite-faucet oracle bias, the wrong observable, the buffer-masked alarm,
adaptation cannibalizing scarce intermediates, and the hysteresis trap. Each
one changed the model; the transcript (footer) is the receipts.

## 7. The scorecard

*[stub]* Three hits, one double miss — and the miss is the product: Sumitomo
1993 misses in *opposite directions* under the two protocols (invisible
through the fab observable in v0; a 57.6%-dip-that-never-recovers through
delivery in v1, against a historical record of "price spike, brief pain, no
catastrophe"). The pessimism bias stated plainly: no price-mediated
allocation, so the model overstates pain — trust where fragility
concentrates, not how bad it says things get. And the uninvited guest:
collapse-as-attractor (hysteresis) appearing in the math without being asked
for.

<!-- scorecard -->

## 8. What this is and isn't

*[stub]* A lens for perspective, not a fix; a solo exploratory project, not a
movement (README). What the lens is good for: seeing structure — chokepoints,
doomed states, gradients, the shape of the fog. What it is not: a forecast,
a policy tool, or a claim that collapse is coming. Play with the demo above;
every number it shows traces back to the repo.

---

*Design history: the model and its five instructive failures were worked out
in one conversation (July 2026). Full transcript export pending —
`docs/TRANSCRIPT.md` holds the placeholder and the
[interim link](https://claude.ai/share/04279425-3700-447c-82c8-9d3861471785).
When the export lands, this footer links it directly.*
