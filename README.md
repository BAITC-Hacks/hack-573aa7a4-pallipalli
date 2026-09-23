# PalliPalli — Beeline tariff campaign agent

Offline agent for Track 04. Uses pandas and NumPy; no API key, hosted model, GPU, or network connection is required at evaluation time.

## Run

```bash
pip install -r requirements.txt
python make_submission.py
```

This creates `submission.csv` with the organizer's fixed seed of 42. Submit it together with `agent.py`, `requirements.txt`, and the provided `data/change_tariff.csv`. Keep the organizer's environment and evaluator files unchanged.

To reproduce the comparison and behavior checks:

```bash
python local_eval.py
python local_eval.py --runs 10
python benchmark.py
python -m unittest -v test_agent
```

## Approach

1. Group the public audience by current tariff and ARPU segment.
2. Rank destinations using historical relative ARPU changes, clipped to limit outliers and shrunk toward zero for small samples. Historical migration frequency is only a ranking proxy: this dataset does not identify causal conversion rates.
3. Compare available channels using their public costs and multipliers. Explore different groups, then repeat promising pilots to reduce uncertainty. Reserve 75% of the initial monetary budget for deployment and at most 27% of contacts (capped at 4,000) for exploration.
4. Update a weak historical prior with observed pilot ratios and the documented pilot noise. Use a conservative lower estimate when selecting final campaigns. This is a heuristic uncertainty estimate, not a calibrated guarantee under population shift or adaptive selection.
5. Greedily choose up to ten campaigns by estimated incremental net value, respecting money, contact, and 5,000-contact campaign caps. Final campaigns use disjoint tariff/ARPU cells. The planner conservatively discounts possible pilot overlap without accessing private subscriber identities.

If no campaign clears the profitability threshold, the case still requires at least one. The agent chooses a small audience and a free channel to limit exposure. Such a fallback can lose ARPU; free contact does not imply a safe tariff change.

## Status

The agent was evaluated against the unchanged organizer starter on 61 matched seeds: 42, 0–29, and 100–129. It was profitable and beat the starter in all 61 runs. At seed 42, net gain was **+3,240,024**, versus **-1,035,279** for the starter. All six behavior tests passed, and generating the seed-42 submission twice produced identical CSV content with four campaigns.

See [the comparison report](docs/BENCHMARK.md), [per-run results](docs/benchmark-runs.csv), and [runtime, checksums, and detailed seed-42 scores](docs/benchmark-results.json). These checks vary pilot randomness on one local population; they do not validate performance on the hidden population.

The hidden judging audience differs from the historical data and local simulator. Greedy planning is not a globally optimal allocation. Pilot costs and outcomes count toward the final score, and uncertainty remains after pilots.

See the [initial project brief](docs/INITIAL_PROJECT_BRIEF.md).
