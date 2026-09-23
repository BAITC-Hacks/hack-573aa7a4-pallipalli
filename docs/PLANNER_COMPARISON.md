# Whole-plan search versus greedy campaign selection

Evaluated on 2026-09-23. Reference: `greedy_agent.py`, byte-identical to `agent.py` in commit `6e4c9ba9ec9de63510c391fe742fe669e07763fc`. Candidate: `agent.py` with `planner.py`. No organizer data, environment, or scoring code was changed.

## What changed

The history model, candidate generation, channel estimates, uncertainty threshold, and pilot policy are unchanged. The comparison confirms identical pilot requests and returned feedback for both implementations at every seed.

The old selector repeatedly took the single highest estimated incremental-profit campaign. The new selector explores complete ordered plans, stopping at any size up to ten, and compares their total estimated net gain. It tracks remaining money/contacts, prevents two final campaigns from sharing a tariff/ARPU cell, and models the public scorer's ID-ordered truncation when a campaign cannot reach its full audience.

A deterministic beam of up to 256 states per depth bounds runtime. Equivalent states are merged, and the greedy solution is kept as an incumbent. The returned estimated objective is therefore at least as good as the planner's greedy incumbent; this does not guarantee a higher true score. None of these 91 benchmark runs required beam-width pruning. The search still considers only piloted candidates with positive conservative incremental gain, not every possible campaign in the case.

## Matched-seed results

We used the original 61 seeds (42, 0–29, 100–129) plus 30 new seeds (200–229) chosen before running this experiment. The algorithm was not tuned after inspecting the comparison. Both versions use the same official local scoring function, including pilot costs and effects.

| Metric, all 91 seeds | Previous greedy | Plan search |
| --- | ---: | ---: |
| Mean net gain | 3,137,647 | **3,184,027** |
| Median net gain | 3,238,907 | **3,240,036** |
| Worst net gain | 2,260,431 | 2,260,431 |
| Best net gain | 3,421,170 | 3,421,170 |
| Profitable runs | 91/91 | 91/91 |
| Valid runs | 91/91 | 91/91 |
| Seed-42 net gain | 3,240,024 | 3,240,024 |
| Seed-42 final campaign count | 4 | 4 |

Average improvement: **46,380 (+1.48%)**. Pairwise outcomes: **17 wins, 72 ties, 2 losses**, with a 0.000001 tolerance for floating-point ties.

On the original 61 seeds, mean gain increased from 3,107,343 to 3,166,212. On the 30 new seeds, mean gain increased from 3,199,265 to 3,220,250; all new-seed runs remained profitable.

The two regressions were seed 2 (-7,686 versus greedy) and seed 101 (-146,347). The largest improvement was seed 121 (+719,685). These are effects of changing the selected plan under noisy estimates, despite identical pilot feedback. We retain and disclose the regressions rather than tuning rules to individual seeds.

## Number of campaigns

| Final campaigns | Greedy runs | Plan-search runs |
| --- | ---: | ---: |
| 3 | 19 | 8 |
| 4 | 55 | 59 |
| 5 | 15 | 21 |
| 6 | 1 | 2 |
| 7 | 1 | 1 |

The planner does not pad output to ten. Seed 14 selected seven campaigns. A separate constructed test verifies that seven profitable campaigns are selected while three unprofitable additions are rejected.

## Checks and limits

- All **15 tests passed**: the six existing behavior tests plus nine planner tests.
- Planner tests cover a case where two campaigns beat the individually strongest campaign, exactly seven campaigns, the ten-campaign cap, overlapping cells, paid truncation followed by free contacts, pilot-overlap discounting, a retained greedy incumbent, and rejection of nonpositive candidates.
- Thirty deterministic small synthetic instances match an independent exhaustive search under the same objective.
- All 91 candidate runs stayed within the official limits: maximum spend 99,990 / 100,000; maximum contacts 11,162 / 15,000; maximum pilots 15 / 20. Campaign sizes were checked against the 5,000-contact cap.
- Maximum observed `Agent.act` runtime was approximately 0.096 seconds on this machine. This is local timing, not a deployment guarantee.
- Generating the seed-42 submission twice produced identical CSV content.
- Runtime: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5. Source/data/submission hashes are in the JSON artifact.

These scores are expected ARPU gains on the local synthetic population, not measured real business outcomes or hidden judging scores. Seeds vary pilot samples/noise, not the population. Historical priors are particularly helpful in this mock environment because it is derived from the same supplied history. Estimated uncertainty is heuristic; pilot identities remain private; the mandatory one-campaign fallback may lose ARPU. Beam search is approximate when pruning is necessary.

## Reproduce

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest -v test_agent test_planner
python compare_planners.py
```

`compare_planners.py` regenerates `submission.csv`, `docs/planner-runs.csv`, and `docs/planner-results.json`. Ship **both `agent.py` and `planner.py`** with the submission and history data.

[Per-run CSV](planner-runs.csv) · [Detailed scores and hashes](planner-results.json) · [Previous comparison with organizer starter](BENCHMARK.md)
