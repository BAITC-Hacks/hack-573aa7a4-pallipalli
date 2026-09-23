# PalliPalli - Beeline Tariff Campaign Agent

Initial project brief | HackAlem AI | 23 September 2026

## Status

Requirements review and proposed implementation plan only. No agent, evaluation results, generated submission, or deployed application is claimed in this document.

Team choice: Track 04 - Telecommunications, Beeline Tariff Marketing Campaigns Case. At review time the platform still displayed Energy as selected; the captain should confirm the official track selection before submission.

## Problem and user

A marketing analyst needs a plan for next month's tariff migration campaigns. Poor targeting can reduce revenue through downsell and waste communication spend. Build an agent that explores uncertain campaign effects with pilots, learns from their results, and proposes a constrained campaign portfolio.

All supplied data is synthetic. It does not represent real Beeline subscribers, tariffs, or business performance.

## Required interface and deliverables

- Implement agent.py with class Agent and method act(self, env) returning a list of campaign dictionaries.
- Return 1-10 valid campaigns with existing target_tariff and channel values.
- Use env.run_pilot and base decisions on observed results; a fixed campaign list is insufficient.
- Respect budget, contact, campaign, and pilot limits.
- Generate submission.csv using the supplied make_submission.py; do not construct it manually.
- Declare additional dependencies in requirements.txt.
- Provide a complete README with architecture, setup, dependencies, environment variables, launch commands, validation scenario, limitations, and third-party attribution.

Allowed environment interface: customer_profile, tariffs, channels, remaining_budget, remaining_contacts, pilots_left, run_pilot, and pilot_history. Use only participant-provided data and this public interface. Never inspect hidden effects, closures, garbage collector objects, or organizer-only files.

Campaign fields: campaign_name; target_tariff; channel; optional filter_arpu_segment, filter_data_segment, filter_call_segment, and filter_current_tariff. Current-tariff filters can contain semicolon-separated tariff codes. Omitted filters mean no filtering on that field.

## Objective and hard constraints

Optimize incremental revenue across unique subscribers minus communication costs. Lift is proportional to each subscriber's predicted ARPU. Pilot activity contributes to the score and consumes the same resources as final campaigns. Repeated contact does not multiply revenue: a subscriber receives the effect of their best campaign, while redundant contacts still incur costs.

| Constraint | Limit |
| --- | --- |
| Target population | 23,441 synthetic subscribers |
| Target tariffs | 21 |
| Final campaigns | 1-10 |
| Subscribers credited per campaign | At most 5,000 |
| Total contacts, including pilots | At most 15,000 |
| Total communication budget, including pilots | 100,000 fictional units |
| Pilot calls | At most 20; at least one required |
| Requested sample per pilot | 10-200 subscribers |
| Runtime in written specification | At most 10 minutes, including model calls |

The specification reports baseline total predicted ARPU of 150,641,084. This is not an achieved result from our agent.

| Channel | Cost per contact | Published effect multiplier |
| --- | ---: | ---: |
| push | 0 | 0.50 |
| sms | 4 | 0.65 |
| digital_ads | 22 | 0.85 |
| call | 160 | 1.20 |

Free push still consumes contacts. Select channels using expected net benefit and opportunity cost, rather than using a single channel everywhere.

## Reviewed participant materials

The ZIP contains agent_template.py, environment.py, mock_environment.py, scoring_core.py, local_eval.py, make_submission.py, PARTICIPANT_GUIDE.md, PARTICIPANT_GUIDE.pdf, customer_profile.csv, tariff_dictionary.csv, feature_dictionary.csv, and the data/ folder.

| Dataset | Purpose |
| --- | --- |
| customer_profile.csv | Target population, current tariff, behavioral segments, predicted_arpu |
| data/change_tariff.csv | Historical sample of 14,824 tariff transitions; useful as prior evidence |
| data/traffic.csv | Monthly voice, SMS, data, and device features |
| data/arpu_monthly.csv | Historical monthly revenue |
| data/dict_tariff.csv | Tariff prices and packages |
| tariff_dictionary.csv, feature_dictionary.csv | Data definitions |

History describes a different audience from the hidden evaluation population. Historical estimates must be updated using pilot observations. Do not optimize against hardcoded mock effects or treat local scores as predictions of hidden-test performance.

Documented profile categories: arpu_segment = LOW/MID/HIGH; data_segment = NON_USER/LITE/HEAVY; call_segment = LOW/MEDIUM/HIGH. Confirm exact boundary handling from supplied data before implementing segmentation.

## Proposed approach (not yet implemented)

1. Validate the supplied schemas and construct candidate segments from current tariff and behavioral categories. Compute audience sizes and predicted revenue exposure.
2. Estimate historical transition priors with shrinkage for sparse groups; combine them with tariff characteristics to rank plausible hypotheses.
3. Run adaptive pilots. Start with a diverse shortlist, update expected lift and uncertainty from actual returned sample sizes, and spend additional pilots where information can improve the final allocation.
4. Estimate campaign net benefit using pilot evidence, ARPU exposure, channel cost, and uncertainty. Treat negative observations cautiously rather than blindly selecting the largest noisy result.
5. Build a portfolio of at most ten campaigns using incremental benefit after overlap, remaining contacts, remaining budget, and the per-campaign cap. Reserve enough resources for final execution while exploring.
6. Return valid campaign dictionaries and log the evidence behind decisions. Keep ordering and tie-breaking deterministic for reproducibility.

Proposed baseline: CPU-based Python with pandas/numpy and a statistical decision policy. GPU hosting is not a case requirement. An LLM may later help propose hypotheses or explain choices, but numerical constraints remain enforced in code and a complete fallback must work without it.

## LLM access and reproducibility

The specification says organizers will supply OPENAI_API_KEY for judging. Read it from the environment; never commit credentials. This does not establish support for custom NVIDIA endpoints or a particular model, so confirm those before introducing such a dependency.

All remote calls need bounded timeouts, error handling, and a deterministic fallback. Optional LLM output must not make the required submission irreproducible.

The supplied make_submission.py uses SUBMISSION_SEED = 42 and states that judges regenerate the CSV on the same mock seed. Any internal randomness must be controlled accordingly. Hidden evaluation runs are expected to yield different decisions because the environment and pilot observations differ.

## Validation plan (not yet executed)

After implementing agent.py and placing the official package alongside it:

    python local_eval.py
    python local_eval.py --runs 10
    python make_submission.py

Acceptance: no runtime errors; no discarded campaigns; pilots performed and used; valid resource usage; 1-10 campaigns; reproducible submission.csv; stable multi-seed behavior; runtime below the conservative target. Check key-absent and API-failure paths if LLM support is added. Preserve the official evaluator and submission generator.

## Case evaluation rubric

The Google specification includes this rubric; the reviewed ZIP guide and Word attachment omit the appended rubric.

| Criterion | Points |
| --- | ---: |
| Task fit and working solution | 25 |
| Technical implementation | 25 |
| README and reproducibility | 25 |
| Value and applicability | 15 |
| Development potential and originality | 10 |
| Total | 100 |

The objective is net revenue improvement, while this rubric also assesses engineering and documentation. Record both measured local outcomes and limitations honestly.

## Schedule and unresolved differences

- Official event regulations specify 13:00-18:00 Astana time on 23 September 2026. The case heading says nine hours. Planning assumption: finish and push before 18:00 unless organizers explicitly clarify otherwise.
- The written case permits ten minutes and optional LLM calls. agent_template.py's header says offline and five minutes. Conservative engineering target: a complete offline-capable baseline under five minutes; seek clarification before relying on network access.
- The case marks the README optional, but general rules require reproducible launch instructions and assign no further consideration to an unlaunchable project. Include the README.
- Confirm the official Track 04 selection and case submission through the platform; this documentation commit is not the final platform submission.
- General rules require demonstrable progress after every competition hour, with history in the organizer-provided repository. Commit meaningful progress throughout development, not only the final code.
- The version fixed at 18:00 is the evaluated version, including Demo Day. Do not rely on post-deadline fixes.

## Planned milestones

1. Requirements and initial design: this document.
2. Import the official package, inspect schemas, and implement a valid pilot-driven baseline.
3. Add uncertainty-aware exploration and constrained campaign allocation.
4. Compare multi-seed results, review overlap and budget behavior, and simplify unreliable components.
5. Complete clean setup verification, README, generated submission.csv, final push, and platform submission before the deadline.

## Sources and attribution

- [Track page](https://edu.astanahub.com/hackathons/df4743f5-c492-415c-b45a-1f13adb78e06?tab=tracks)
- [Official Google specification](https://docs.google.com/document/d/1bt_tgnIXnsnGMMjeaqbOY165MXmYQOTllRCDwcKySKI/edit)
- [Official participant ZIP](https://drive.google.com/file/d/1cQUKtE_cm9TVXgzpcFwQYYUpmuFaJHHT/view)
- [Word specification attachment](https://drive.google.com/file/d/18zfkukVXGxjjeTaRxpCcTRxskdsNbAT-/view)
- [General regulations](https://edu.astanahub.com/hackathons/df4743f5-c492-415c-b45a-1f13adb78e06?tab=regulations)

Reviewed: Google specification, all four pages of the Word preview, ZIP inventory, PARTICIPANT_GUIDE.md, agent_template.py, make_submission.py, and the participant environment contract. No supplied scripts have been executed for this initial documentation milestone. Organizer code and synthetic data remain attributed to the supplied participant package. This analysis and proposed plan were prepared with Codex assistance for PalliPalli.
