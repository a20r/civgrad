# Essay changelog

The essay is versioned like METHODOLOGY.md: bump `essay/VERSION`, add a line
here, and keep the sections' claims consistent with the scorecard at that
version.

- **v0.1** — scaffold: eight section stubs (2–4 sentences each, sources named,
  no prose), demo embedded between §5 and §6, scorecard mirror in §7,
  design-history footer with the interim transcript link.
- **v0.2** — first full draft: all eight sections written to the v0.1 stub
  contracts; numbers sourced from validation/baseline_outputs.txt and
  SCORECARD.md (shipping gradient −2.68/−1.63, boneyard 0.85 vs 0.70,
  Sumitomo v1 57.6%/never, α=0.06); Hunt–Lipo caveat, busy-futility
  livelock, five-failures narrative, and pessimism-bias caveat included.
- **v0.2.x (blog-side, unversioned at the time)** — the essay's canonical
  home moved to the blog (a20r/blog PRs #5–#9): §2 citations added, Anasazi
  framing dropped, a formal machinery section (§3) and a method reflection
  (§9) added there, and sensitivity/price-experiment numbers quoted from an
  analysis whose repo artifacts had not yet landed. Recorded here
  retroactively; this repo copy stayed at the 8-section structure.
- **v0.4** — the audit and the conclusion, in full (blog copy): the essay
  absorbs the evidence it previously linked out to — the 500-draw audit
  summary, the buffer-rule table, both robust marginal-value tables, the
  mean-vs-tail comparison, and the price experiment's two result tables
  now sit inline in §8/§9 of the blog copy, styled on the site theme.
  This repo-site copy stays prose-only (its client-side renderer has no
  table support); the generated POLICY.md / SENSITIVITY.md /
  PRICE_EXPERIMENT.md remain the regenerable sources of every number.
  Includes the sign-label fix (strict directional fractions; exactly-zero
  gradients support neither sign).
- **v0.3** — audited numbers + the policy section, both copies: the §5
  sensitivity parenthetical and boneyard passage now quote the committed,
  reproducible audit (SENSITIVITY.md: big negative signs 85–92% under
  parameter fog, 100% under prior fog; partition 91–99%; worn-on-top a
  minority); the §7 Sumitomo passage corrected to the *measured* trap
  anatomy (input-masking in the restoration signal, not tool-fleet wear;
  KNOWN_ISSUES #10) and now cites the structural-miss result (95% of 500
  draws) and the price experiment (margin restoration → ~10 mo recovery;
  gradient-as-price → total collapse); new §8 "The marginal dollar, cashed
  out" condensing POLICY.md; old §8 renumbered §9 with its scope line
  updated. Header notes the blog as the canonical fullest copy.
