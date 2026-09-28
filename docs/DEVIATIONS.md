# Deviations from the preregistration

Deviations from `prereg/PREREGISTRATION.json` (committed in c8e32a0) are listed here with the reason for each.

## 1. H4 proposer changed before any confirmatory run

**What.** In the code committed with the preregistration, the self-improvement proposer made
evolution-strategies steps (step 0.004 per parameter) and Gaussian mutations (sd 0.01). It now
makes *practice* steps (2 Adam steps of backpropagation on 8 fresh wounded tissues, lr 2e-4) and
Gaussian mutations with sd 0.001. Physics edits are unchanged.

**Why.** On the 120-iteration pilot rule (the only rule used for piloting) every ES step and every
0.01 mutation raised the loss by 0.02 to 0.15, far beyond evaluation noise. Both loops would reject
every weight edit, and the naive/gated comparison would reduce to physics edits alone. Smaller
edits land where evaluation noise matters, which is what H4 is about.

**When.** Decided on the pilot rule while rule A was still training, before any H1 to H4 run
on rule A or rule B. The hypotheses, measures, sample sizes, seeds and decision rules are unchanged.

## 2. Training recipe gained a persistence phase; both rules retrained

**What.** The preregistration described rules trained for 2000 iterations on the growth → wound →
regeneration life only. Both rules are now trained for 3000 iterations, where each iteration
averages two half-batches: the same life as before, and a *persistence* half in which tissues drawn
from a pool of old tissues are wounded with probability 1/2 and must regrow or hold the body for
another 48 steps (sample-pool training, as in Mordvintsev et al. 2020).

**Why.** The first rule A (2000 iterations, life only) grew and regenerated almost perfectly within
the 112 steps it was trained on, then drifted and overgrew the grid: anatomical loss 0.0013 at step
111, 0.0033 at step 160, 0.041 at step 250, 0.124 at step 313 (8 tissues). H3b's second
regeneration runs to step 160, partly outside the trained range, and the browser lab runs tissues
for thousands of steps. A body that does not hold its shape cannot test anatomical memory.

**When and what was seen.** Found by opening the browser lab on the first rule A. A confirmatory
`morpheus run all` on that rule had been started. It was stopped within minutes, and its log and
partial output were deleted **without being read**. No H1–H4 number from the first rule A was seen.
The first rule A is not analysed or published. Hypotheses, measures, sample sizes, suite seeds and
decision rules are unchanged.
