# Campaign analyst report

Local mock seed: **42**.

## Authoritative run summary

Numbers below are rendered by Python from public evidence. Estimated deployment
spending is separate from observed pilot spending; no scorer result is reported.

| Metric | Value |
|---|---:|
| Audience | 23,441 |
| Pilots completed | 14 |
| Final campaigns | 4 |
| Initial budget | 100,000.00 |
| Pilot cost (observed) | 24,992.00 |
| Budget after pilots (observed) | 75,008.00 |
| Deployment cost (estimated) | 67,056.00 |
| Budget after deployment (estimated) | 7,952.00 |
| Pilot contacts (observed) | 2,311 |
| Contacts after pilots (observed) | 12,689 |
| Deployment contacts (estimated) | 3,048 |
| Conservative final-plan net (estimated, excludes pilots) | 906,469.05 |
| Actual scored net gain | Not calculated by this analyst |

## Selected campaigns

| Campaign | Current tariff | ARPU segment | Target | Channel | Estimated contacts | Estimated cost |
|---|---|---|---|---|---:|---:|
| plan_search_1 | tariff_4 | MID | tariff_8 | digital_ads | 1,227 | 26,994.00 |
| plan_search_2 | tariff_13 | MID | tariff_8 | digital_ads | 1,022 | 22,484.00 |
| plan_search_3 | tariff_12 | MID | tariff_8 | digital_ads | 342 | 7,524.00 |
| plan_search_4 | tariff_8 | LOW | tariff_9 | digital_ads | 457 | 10,054.00 |

## Evidence limitations

- Local mock evidence is not the hidden judging score.
- Pilot effects are noisy observations, not true model effects.
- Historical migration shares are ranking proxies, not causal conversion probabilities.
- Catalog ranking uses population-average ARPU; it is not a scored campaign result.
- Deployment cost and contacts are estimates from public filters and remaining limits.
- Planner estimated_net is a conservative incremental final-plan estimate; it excludes pilot net effects.
- No actual scored net gain, true effects, or subscriber records are included.
- Catalog hypotheses are suggestions only and do not change or execute the submission.

## Optional LLM commentary

Generated interpretation; the authoritative numerical summary above takes precedence.
Hypotheses are proposals only and have not changed or executed the agent's plan.

Further pilot evidence could help evaluate alternatives to the current plan.

### Observations

- candidate_001 is included in the selected plan.
- candidate_001 has historical migration evidence from a different population.
- candidate_001 has matching pilot observations, which remain noisy.
- candidate_001 has positive observed lift in its matching pilots; this is not a guarantee.
- candidate_002 is included in the selected plan.
- candidate_002 has positive observed lift in its matching pilots; this is not a guarantee.

### Suggested hypotheses (not executed)

- **candidate_003**: candidate_003 has historical migration evidence from a different population.
  Evidence to collect: Repeat a matching pilot to reduce uncertainty in observed ARPU lift.
- **candidate_004**: candidate_004 has historical migration evidence from a different population.
  Evidence to collect: Run a first matching pilot and record observed ARPU lift, contact count, and cost.
- **candidate_006**: candidate_006 has historical migration evidence from a different population.
  Evidence to collect: Repeat a matching pilot to reduce uncertainty in observed ARPU lift.
- **candidate_020**: candidate_020 has historical migration evidence from a different population.
  Evidence to collect: Repeat a matching pilot to reduce uncertainty in observed ARPU lift.
- **candidate_015**: candidate_015 has historical migration evidence from a different population.
  Evidence to collect: Run a first matching pilot and record observed ARPU lift, contact count, and cost.

### Interpretation limitations

- Local mock evidence is not the hidden judging score.
- Pilot effects are noisy observations, not true model effects.
- Historical migration shares are ranking proxies, not causal conversion probabilities.
- Catalog ranking uses population-average ARPU; it is not a scored campaign result.
- Deployment cost and contacts are estimates from public filters and remaining limits.
- Planner estimated_net is a conservative incremental final-plan estimate; it excludes pilot net effects.

## Analyst mode

Status: openai. Reason: completed.

This report does not modify the campaign plan or submission CSV.
