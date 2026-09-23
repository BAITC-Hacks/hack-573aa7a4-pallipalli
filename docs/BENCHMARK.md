# Campaign agent comparison

Evaluated on 2026-09-23 with the official participant package. Baseline: unchanged `agent_template.py`, equivalent to `agent.py` on repository main at `ae3a526aee5239e268735f7333c275d3cbd7df29`. Candidate: the new history-informed pilot agent. No organizer environment, simulator, scorer, or data files were modified.

## Results

Both implementations used the same fixed 61 seeds: 42, 0–29, and 100–129. The candidate implementation was not tuned between these runs.

| Metric | Starter | New agent |
| --- | ---: | ---: |
| Seed-42 net ARPU gain | -1,035,279 | +3,240,024 |
| Mean net, 61 seeds | -548,216 | +3,107,343 |
| Median net | -386,999 | +3,238,907 |
| Worst net | -1,908,609 | +2,260,431 |
| Best net | -72,156 | +3,421,170 |
| Profitable runs | 0 / 61 | 61 / 61 |
| Valid final campaign count and schema | 57 / 61 | 61 / 61 |

The candidate beats the starter on all 61 paired seeds. At seed 42 the improvement is **4,275,304 net ARPU units** (computed before rounding).

## Seed-42 resource use

| Metric | Starter | New agent |
| --- | ---: | ---: |
| Pilot campaigns | 6 | 14 |
| Final campaigns | 2 | 4 |
| Total communication spend | 22,280 | 92,048 |
| Contacts, including pilots | 5,570 | 5,359 |
| Customers with negative scored effects | 66.3% | 0.0% |

Across all 61 candidate runs, maximum spend was 99,990 / 100,000; maximum contacts were 11,162 / 15,000; maximum pilots were 15 / 20. Runtime and exact per-run metrics are recorded in the linked artifacts. All candidate runs satisfied the five-minute runtime threshold.

The evaluator includes pilots in its printed campaign total; the ten-campaign limit applies to the final plan. Its printed environment balance after execution reflects pilot deductions only; this report uses final scoring totals instead.

## Checks performed

- Both agents scored with the same official `evaluate_agent` implementation, including pilot effects, duplicate-customer scoring, channel costs, and contact caps.
- All new-agent runs completed without dropped campaigns or evaluator warnings; all returned 1–10 valid campaigns and used pilots.
- Pilot requests stayed within 10–200, at most 20 calls, and nonnegative remaining resources.
- Total scored spend, total contacts, and per-campaign contacts were checked against official limits.
- Submission generated twice at seed 42 with byte-identical CSV content.
- Six behavior tests passed: positive versus negative pilot feedback changes decisions; zero budget uses free channels; missing history still uses pilots; weak evidence triggers the documented fallback; seed-42 output is deterministic; repeated pilot evidence narrows posterior uncertainty.
- The starter returned no final campaigns in four seeds; its scored pilot losses are retained in the comparison, and these runs are marked invalid rather than silently omitted.

## Reproduce

From the repository root, using the organizer data files:

```bash
pip install -r requirements.txt
python benchmark.py
python -m unittest -v test_agent
```

Runtime used: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5. `benchmark.py` records SHA-256 hashes for the agent, baseline, simulator, relevant input data, and generated submission. It regenerates `submission.csv`, `docs/benchmark-runs.csv`, and `docs/benchmark-results.json`.

## Interpretation and limits

These are local simulated expected ARPU effects on synthetic data. Changing seeds changes pilot samples and noise, not the underlying population. The mock model is itself built from the provided migration history, so a history-based agent has an advantage here that may shrink or disappear in hidden judging. The controlled negative-feedback test checks behavioral adaptation; it does not prove profitable performance under population shift.

The uncertainty threshold is heuristic, the allocation is greedy, and pilot/final overlap cannot be precisely removed because pilot identities are private. The mandatory fallback may still lose ARPU. No API credits or GPU were used.

[Per-run CSV](benchmark-runs.csv) · [Detailed JSON and file hashes](benchmark-results.json)
