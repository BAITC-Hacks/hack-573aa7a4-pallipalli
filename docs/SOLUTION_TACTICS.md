# Solution tactics

Proposed on 2026-09-23. These are experiment plans, not measured improvements.

## Objective and current position

Maximize unique-subscriber ARPU gain minus all communication costs, including pilots. Respect the 100,000 money budget, 15,000 contacts, 20 pilots, 10–200 requested contacts per pilot, 5,000 contacts per final campaign, and 1–10 final campaigns.

The current numerical agent uses historical priors, noisy pilot updates, and ordered plan search. Its mean local gain is 3,184,027 across 91 seeds. At seed 42 it chooses four final campaigns after 14 pilots and spends 92,048. Neither 20 pilots, ten campaigns, nor spending the whole budget is a target: an additional action must improve expected net value after accounting for scarce contacts and uncertainty.

The 91 seeds vary pilot sampling/noise on one population. They establish local noise robustness, not robustness to the different effects used in judging. The optional OpenAI analyst has passed a live API check but has not improved campaign scores.

## Priority 1 — make pilot selection feasible and adaptive

Verified limitations in `agent.py`:

- Candidate generation excludes a channel when a full 200-person pilot would exceed the 25% exploration allowance, even if a smaller legal pilot is affordable.
- Repeat exploration stops when its top choice cannot be piloted, without considering cheaper alternatives.
- Pilot size usually starts at 200, exploration covers at most 12 distinct candidates, and repeat priority does not explicitly estimate the value of changing the final plan.

First experiment: enumerate feasible candidate/size pairs (for example 10, 30, 60, 100, and 200, capped by audience and remaining resources). Skip infeasible choices rather than stopping the entire search. Preserve a deployment reserve initially so we can isolate this change.

Next experiment: compare a new candidate, a repeat pilot, and stopping. Prioritize uncertainty that could change the best deployment plan. A large pilot is justified when a valuable campaign is close to the selection boundary; a clearly poor candidate should stop receiving contacts. Include pilot communication cost, its own ARPU contribution, possible duplication with deployment, and the opportunity cost of using contacts.

Only after those comparisons should we replace the fixed 75% deployment reserve with a plan-aware allocation. Extra free push pilots still consume contacts and may produce negative ARPU effects.

## Priority 2 — consider better campaign alternatives

The current generator keeps two destinations and one preferred channel per destination for each current-tariff/ARPU group. This can remove useful alternatives before a pilot is run.

- Retain a small, diverse shortlist across destinations and channels; compare cheap exploration with potentially stronger paid deployment.
- Explore the allowed data and call segments where there is enough audience and evidence. Start with disjoint subdivisions to preserve the planner's overlap assumptions.
- Keep campaign definitions expressible using the organizer's supported filters. Individual model scores alone are not valid campaign filters.
- Use actual public channel feedback. Treat sharing effect estimates across channels as a hypothesis to validate; the public multipliers alone do not establish that every hidden effect transfers exactly.

The existing beam search was not pruned in any of the 91 benchmark runs. Expanding beam width is therefore a lower priority than improving the candidates and their evidence.

## Priority 3 — learn a better historical prior

Build a small tabular regression experiment for relative ARPU change after migration. Start with origin/destination tariffs, pre-migration ARPU, tariff price/package differences, and support counts. Add traffic and ARPU trends only when records can be joined strictly before the migration timestamp.

Compare a regularized model and a tree model with the current grouped statistical prior. Use chronological validation and keep subscriber identities separated across train/validation data where possible. Never use post-migration revenue as an input or join future monthly features into historical rows.

Historical records contain migrations, not all contacted and non-converting subscribers. The model predicts outcomes among recorded migrations; it cannot identify campaign conversion probability or causal contact uplift from these records alone. Treat its output as a weak prior, recalibrate with public pilots, and reduce its weight when pilot evidence disagrees.

A model with lower historical prediction error is not automatically a better campaign agent. Adoption depends on downstream net gain and downside behavior. A small CPU model is the initial implementation; GPU use is an optional experiment, not a dependency for this design.

## Priority 4 — use OpenAI for bounded hypothesis generation

The working integration selects grounded facts and proposes experiments. Its five frozen suggestions are inactive. Evaluate them as a candidate proposal policy alongside ordinary numerical selection.

OpenAI can propose a supported segment/target/channel combination or explain why two alternatives need another pilot. Python validates all identifiers, filters, limits, and evidence; the numerical model estimates value, and the planner chooses the final campaigns.

Initially keep proposals frozen and versioned so their effect can be measured reproducibly. Only consider live LLM decisions after showing useful incremental value and resolving submission reproducibility, timeout, and fallback behavior. A larger model or GPU-hosted model is not yet justified by campaign results.

## Evaluation and promotion

1. Freeze the current numerical agent and planner as the reference before changing them.
2. Compare one change at a time: feasible pilots, adaptive sizes, broader candidates, learned prior, and LLM proposals. Then compare the useful combinations.
3. Use paired development seeds and reserve untouched seeds for the final comparison. Once inspected, a holdout becomes development data for future iterations.
4. Separately build participant-owned stress fixtures with weaker/reversed priors, sparse support, null or negative lift, and different channel responses. Do not modify organizer files or derive policies from private simulator effects. Stress scores are diagnostic, not official scores.
5. Report mean and median net gain, lower-tail/worst outcomes, wins/ties/losses, paired uncertainty, money and contacts used, pilot/campaign counts, validity, runtime, and deterministic output. Log why exploration stopped and why budget was left unused.
6. Adopt improvements only with all hard constraints satisfied and a credible gain after considering regressions and downside. Preserve unsuccessful experiment results; do not tune to named seeds.

## Implementation order

**Next change:** feasible variable-size pilot scheduling, with explicit stop reasons. Compare it with the current agent before expanding the candidate pool or adding ML. Commit every accepted change to `galammadin`.

Then expand candidate coverage, evaluate a learned prior, and measure whether OpenAI proposals add value. Keep a working submission available at every step.

## Evidence

- [Participant requirements](../PARTICIPANT_GUIDE.md)
- [Current agent](../agent.py) and [planner](../planner.py)
- [Existing matched-seed comparison and limitations](PLANNER_COMPARISON.md)
- [OpenAI integration validation](OPENAI_VALIDATION.md)
