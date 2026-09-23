# Feature branch review and controlled experiment

## Outcome

The combined agent averages **4,871,270** net gain on 50 reserved seeds (500–549), versus **4,038,323** for our previous best: **+832,946 / +20.63%**, winning every paired run. The source branch averages **4,842,687**; the combined version adds **28,582 / +0.59%**, with **35 wins, 2 ties and 13 losses**. The largest regression versus the source branch is **265,775**.

The combined version was selected on development results before these reserved runs. All policies remained frozen during final evaluation. The source branch was reproduced at commit [`3cf067f7ce038a0a56145bda0a59e261e9388392`](https://github.com/BAITC-Hacks/hack-573aa7a4-pallipalli/tree/3cf067f7ce038a0a56145bda0a59e261e9388392); its agent blob is `11cb77e22174f88faae42bd5510de684cd99e7c6`. Only its console logging was disabled. Its scorer, environment and all supplied datasets match our official package byte-for-byte.

## What the earlier solution missed

1. **Value of an experiment.** The previous heuristic mostly favored high estimated campaign value and repeated promising offers. The branch estimates how a pilot could change the choice between competing tariffs, evaluates prospective sizes, and subtracts money and contact opportunity costs. Our previous unsuccessful adaptive-size experiment used a different rule; its loss did not rule out this approach.
2. **Evidence-dependent prior uncertainty.** We used the same prior standard deviation for every candidate and shrank sparse means toward zero. The branch shrinks sparse segments toward the same tariff pair, combines sampling uncertainty with a distribution-shift allowance, and uses price-based fallback estimates for unseen transitions. The new prior module also handles malformed/missing history and the exact ARPU=1000 boundary.
3. **Compatible campaign merging.** The branch can combine several origin tariffs sharing target, channel and ARPU segment. The new planner generates such alternatives while retaining individual response estimates and ID-ordered prefixes. Merges require the full union audience to fit 5,000 and cannot overlap other selected groups.
4. **Optimistic shortlisting.** The branch retains four destinations using mean plus uncertainty, versus our previous three selected by mean. We tested this separately because it changes where limited pilots are spent.

## Method and results

Development used seed 42 plus 0–9 (99 recorded runs: nine policies including a refactor control). The reserved comparison used 500–549 (450 recorded runs). A control refactor exactly matched the old policy’s pilots, returned plans and scores across both sets. The input files and policy hashes are recorded in the JSON manifests.

| Policy | Mean net | Median | Worst | Profitable / valid |
|---|---:|---:|---:|---:|
| Previous best (frozen) | 4,038,323 | 4,127,111 | 3,100,216 | 50/50 · 50/50 |
| feat/agent-solution, pinned | 4,842,687 | 4,855,355 | 4,230,649 | 50/50 · 50/50 |
| Add hierarchical priors only | 4,402,369 | 4,491,944 | 3,803,310 | 50/50 · 50/50 |
| Add information-value pilots only | 2,343,780 | 2,287,420 | 1,727,160 | 50/50 · 50/50 |
| Add compatible grouping only | 4,038,313 | 4,127,111 | 3,100,216 | 50/50 · 50/50 |
| Add optimistic four-target shortlist only | 2,735,554 | 2,803,582 | 2,371,900 | 50/50 · 50/50 |
| Priors + information-value pilots + shortlist | 4,773,782 | 4,787,209 | 4,246,878 | 50/50 · 50/50 |
| Combined additions + grouped whole-plan search | 4,871,270 | 4,831,469 | 4,309,757 | 50/50 · 50/50 |

The combinations matter. New priors alone add 364,046 on average, but information-value exploration with the old uniform uncertainty loses 1,694,543; optimistic shortlisting alone loses 1,302,769. Grouping alone is essentially unchanged (49 ties and one 512-unit regression). These weak standalone results were retained.

Within the stronger prior + exploration + shortlist policy, adding grouping increases mean net by **97,488**, with **41 wins, 9 ties and no losses**. These pairs have identical pilot traces, isolating the final-planner addition. The other changed pilot policies do not receive identical feedback merely because their seeds match.

The paired bootstrap 95% interval for mean improvement is **756,306 to 914,354** versus our frozen agent and **7,881 to 47,760** versus the source branch. These intervals summarize pilot randomness on the fixed mock population; they do not establish hidden-judge performance.

## Implementation choices

- `prior_model.py`: pair-level shrinkage, sample-dependent uncertainty, price fallback, finite-value checks and correct segmentation boundaries.
- `kg_policy.py`: approximate information value within each customer group, public SMS/push measurement parameters, legal sizes and reserves, stop diagnostics, and a mandatory first feasible pilot. The inherited opportunity cost of 30 per contact remains a heuristic, not an organizer rule. This is not a full constrained-plan value-of-information calculation.
- `grouped_planner.py`: full-audience merge limits, exact sorted prefixes, member-cell conflict masks and the ungrouped plan retained as an incumbent for estimated value. Search remains approximate: pairs plus deterministic packs, at most 128 merge options, beam width 256.
- `hybrid_agent.py`: integrates the additions with our expected-profit objective, one-standard-deviation eligibility check, expected pilot-overlap discount, and conservative cap on channel multipliers above1. Only piloted offers are deployed.
- Small eligible groups and zero-budget operation remain supported. A mandatory free fallback can still reduce ARPU.
- The source branch’s internal partial take counts are not serialized in campaign filters. Our planner estimates the actual ID-ordered resource-truncated audiences, including merged filters. The source branch itself passed all official runs in this experiment.
- No LLM API or external service is used for scored decisions. The optional analyst remains separate.

## Submission and checks

At seed 42, the combined agent reports **PASS**, net **4,730,893**, gross **4,830,873**, spend **99,980**, **9,308** total contacts, **20 pilots** and **6 final campaigns**. The source branch scores **4,772,379** at this seed, so the combined solution is lower by **41,486** here despite its higher reserved-set average.

Across the reserved 50, the combined agent is valid/profitable in **50/50**; maximum spend 100,000, contacts 15,000 and pilots 20; final campaigns 6–10; maximum Agent.act time **0.334s** on this machine. 44 runs use at least one merged campaign. Observed pilot sizes range 115–200 (often limited by audience size); this does not demonstrate a benefit from very small pilots alone.

**72 tests passed:** 8 prior tests, 12 information-value tests, 10 grouping tests, 9 original planner tests, 6 integration tests, 6 existing agent contract tests and 21 optional-analyst regression checks using offline/fake clients. No live API calls were made. Two regenerated seed 42 submissions match the saved CSV byte-for-byte.

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python run_experiment.py --split development
python run_experiment.py --split validation
python -m unittest -v test_prior_model test_kg_policy test_grouped_planner test_planner test_hybrid_behaviour
python local_eval.py
python make_submission.py
```

The reserved seeds are now observed; use a new set for future policy optimization. `agent.py` is a small entry point importing the numerical modules. Submit all seven runtime Python files, the CSV, requirements and provided migration history together.

## Evidence and limits

[Per-seed scores](../results/validation/runs.csv) · [Summary, source hashes and regressions](../results/validation/summary.json) · [Selection made before validation](../development-selection.json) · [Experiment manifest](../experiment.json). Complete pilot traces are in the local results folder.

All data and scores are synthetic. The local mock derives effects from the supplied history, which benefits historical priors; a new random seed does not change that population. History records migrations, not all contacted customers, so migration frequency does not identify conversion probability. Cross-channel transfer, uncertainty scale, overlap probabilities and the contact opportunity cost remain modeling assumptions. Neither a confidence threshold nor keeping an estimated-value incumbent guarantees profitable hidden outcomes.
