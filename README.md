# PalliPalli — Beeline tariff campaign agent

Offline agent for Track 04. Uses pandas and NumPy; no API key, hosted model, GPU, or network connection is required at evaluation time.

## Run and submit

```bash
pip install -r requirements.txt
python make_submission.py
python local_eval.py
```

Submit `agent.py`, **`planner.py`**, `submission.csv`, `requirements.txt`, and the provided `data/change_tariff.csv`. `agent.py` imports the planner module, so both Python files must be included. The organizer's submission generator uses seed 42. Keep the organizer environment, evaluator, and data files unchanged.

## Optional OpenAI analyst

```bash
pip install -r requirements-llm.txt
python analyst.py report                 # offline report, no API cost
python analyst.py report --openai        # uses OPENAI_API_KEY
```

The analyst explains the plan using grounded public evidence and proposes candidate hypotheses. It runs separately from the scored agent, with deterministic fallback reports on API errors. Suggestions can be frozen for future evaluation; they are not activated automatically.

[Integration guide](docs/OPENAI_INTEGRATION.md) · [Live validation](docs/OPENAI_VALIDATION.md) · [Project context](docs/PROJECT_CONTEXT.md)

## How decisions are made

1. Rank destination tariffs using clipped historical relative ARPU changes, migration frequency, and shrinkage for small samples. Migration frequency is a ranking proxy, not an identified causal conversion probability.
2. Compare channels using public costs and effectiveness multipliers. Explore different groups, then repeat promising pilots when resources allow. Reserve 75% of initial money for deployment and at most 27% of contacts (capped at 4,000) for exploration.
3. Update a weak historical prior with pilot feedback and the documented pilot noise. Use a conservative lower estimate of campaign profit. The uncertainty estimate is heuristic, not a calibrated guarantee under population shift or adaptive selection.
4. **Compare whole ordered campaign plans of different sizes, up to ten.** Beam search retains up to 256 states per depth, merges equivalent states, and keeps the previous greedy plan as an incumbent. It never pads a plan to ten campaigns. Order matters when money or contacts can only fund part of a campaign.
5. Select disjoint current-tariff/ARPU groups. Discount possible overlap with pilots without accessing their private subscriber identities. Track money, contact, and 5,000-contact campaign caps in every proposed plan.

If no campaign clears the profitability threshold, the case still requires one. The agent uses a small audience and a free channel to limit exposure. This fallback may lose ARPU; free contact does not imply a safe tariff change.

## Latest comparison

Compared with the frozen greedy version from commit `6e4c9ba`, on 91 matched seeds:

- Mean net gain: **3,184,027**, versus **3,137,647** (+1.48%).
- 17 wins, 72 ties, 2 losses; both versions profitable on all 91 seeds.
- Seed 42: **3,240,024**, unchanged, with four final campaigns.
- Selected campaign counts ranged from three to seven.
- All 15 behavior/planner tests passed. Submission CSV is reproducible.

The same pilots and feedback were confirmed for both versions on every seed, isolating the change in plan selection. A better estimated plan can still have a worse true simulated score because the pilot estimates are noisy. Beam search is approximate; keeping the greedy incumbent protects the estimated objective, not the hidden score.

[Latest comparison and limitations](docs/PLANNER_COMPARISON.md) · [Per-run CSV](docs/planner-runs.csv) · [Detailed JSON and hashes](docs/planner-results.json)

## Reproduce checks

```bash
python compare_planners.py
python -m unittest -v test_agent test_planner
```

`greedy_agent.py` is the frozen reference used only for comparison. `compare_planners.py` regenerates current `submission.csv` and planner result artifacts. The six original agent tests and nine planner tests cover variable campaign count, budget/contact truncation, disjoint groups, deterministic output, and agreement with exhaustive search on small instances.

`python benchmark.py` compares the current agent with the organizer starter. The [earlier starter comparison](docs/BENCHMARK.md) records the previous greedy revision's results. All local scores use synthetic data; changing seeds changes pilot randomness, not the underlying audience. Hidden judging uses different effects.

See the [solution tactics and experiment order](docs/SOLUTION_TACTICS.md) and [initial project brief](docs/INITIAL_PROJECT_BRIEF.md).
