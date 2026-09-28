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
