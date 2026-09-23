# PalliPalli — project context from the hackathon conversation

## Purpose and standing instructions

Team PalliPalli is building a tariff marketing campaign agent for HackAlem AI, Track 04 / Telecommunications, Beeline case. Work with the user through implementation, evaluation, and delivery.

- GitHub repository: https://github.com/BAITC-Hacks/hack-573aa7a4-pallipalli
- **Always commit to `galammadin`; create it if absent.** This was explicitly requested by the user.
- Latest prior remote implementation: `c3038836dfe1c0fd7b1bd67d118f678a25dbcfef` (whole-plan search). This context accompanies the subsequent optional OpenAI integration commit.
- This local folder was assembled from the official participant ZIP and GitHub connector reads. It does **not** currently have `.git` metadata. GitHub commits have been made through the connected GitHub tools. A regular CLI fetch requires GitHub authentication that was not available in this runtime.

## Case requirements

Implement `agent.py`, class `Agent`, `act(self, env) -> list[dict]`. Return 1–10 final campaigns, use at least one pilot, and adapt to pilot feedback. Limits: 20 pilots, requested size 10–200, total money 100000, total contacts 15000 including pilots, 5000 contacts per campaign. Contact effects are proportional to predicted ARPU; duplicate customers receive their best campaign effect once but all contacts cost resources.

Only public environment data and pilot calls are allowed. Never inspect private effects/closures. Supplied migration history describes a different audience from hidden judging. Local scores are not hidden judge scores.

Submit `agent.py`, `planner.py`, deterministic `submission.csv`, `requirements.txt`, and provided history data. `make_submission.py` uses seed 42 and judges regenerate the CSV. Keep API calls out of scored decisions unless reproducibility is solved. API fallback and a total runtime under 10 minutes are required; our local agent is far faster.

## Sources

- Participant guide: `PARTICIPANT_GUIDE.md`.
- Case specification: https://docs.google.com/document/d/1bt_tgnIXnsnGMMjeaqbOY165MXmYQOTllRCDwcKySKI/edit
- Word specification: https://drive.google.com/file/d/18zfkukVXGxjjeTaRxpCcTRxskdsNbAT-/view
- Official base ZIP: https://drive.google.com/file/d/1cQUKtE_cm9TVXgzpcFwQYYUpmuFaJHHT/view
- Regulations: https://edu.astanahub.com/hackathons/df4743f5-c492-415c-b45a-1f13adb78e06?tab=regulations

## Implemented numerical agent

`agent.py` ranks destinations from clipped historical relative ARPU changes, migration frequency proxies, and sample-size shrinkage. It compares channel costs, explores several tariff/ARPU groups, and updates a weak prior with noisy pilot results. It reserves 75% of initial money for deployment. Its pilot selection can stop when a preferred pilot is unaffordable within the exploration allowance; considering cheaper alternatives is a possible future improvement.

`planner.py` searches complete ordered plans of variable size with a beam width 256, retains the greedy incumbent, respects money/contacts/campaign caps, prevents duplicate final groups, and conservatively discounts possible overlap with pilots. Ten is a maximum, not a target. The previous greedy version is frozen in `greedy_agent.py`.

### Results

- Original organizer starter: seed 42 net **-1035279**; 0/61 profitable runs.
- First statistical/greedy agent: seed 42 **+3240024**; 61/61 profitable.
- Whole-plan search versus greedy,91 matched seeds: mean **3184027** versus **3137647** (+1.48%); 17 wins, 72 ties, 2 losses; both 91/91 profitable.
- At seed 42,14 pilots and4 final campaigns; pilot cost 24992, final cost 67056, total 92048/100000.
- Evaluator's printed remaining 75008 refers to the balance after pilots, not after final campaigns.
- Plans selected 3–7 campaigns; seed 14 selected 7. See `docs/PLANNER_COMPARISON.md` and raw results.

## Optional OpenAI integration

The user approved adding explanations and proposed exploration hypotheses while protecting the scored agent's reproducibility.

- `analyst.py`: CLI for offline/OpenAI reports and freezing proposals.
- `analyst_evidence.py`: deterministic public aggregates, pilot capture, exact candidate matching, budget tables.
- `openai_analyst.py`: optional Responses API client, strict schema, grounded statement enums, local validation and error fallback.
- `requirements-llm.txt`: optional SDK dependency. Core submission does not need it.
- `policies/openai-proposal-v1.json`: five live-generated hypotheses, explicitly inactive and requiring evaluation.
- `docs/OPENAI_REPORT_EXAMPLE.md`: successful grounded live report.
- `docs/OPENAI_INTEGRATION.md` and `docs/OPENAI_VALIDATION.md`: commands, limitations, and validation.

Model: `gpt-4.1-mini-2025-04-14`, configurable. One call per enabled CLI invocation, 20-second SDK timeout, zero automatic retries, 1800 output-token cap, store=False. No per-subscriber records or IDs are sent. The model selects supported evidence and proposed experiments; Python supplies numerical facts and validates candidate references. An initial free-text prototype mixed references and unavailable churn/conversion metrics; the final enum-constrained version rejects such outputs.

Credentials were used only transiently for live checks and are not in this project. Configure a current API key via `OPENAI_API_KEY` for future calls. A credential was pasted in the prior conversation and should be rotated. Do not copy conversation credentials into files or new task prompts.

## Reproduce

```bash
pip install -r requirements.txt
python make_submission.py
python compare_planners.py
python -m unittest -v test_agent test_planner test_openai_analyst test_analyst_evidence
pip install -r requirements-llm.txt
python analyst.py report
# Configure OPENAI_API_KEY in the runtime for:
python analyst.py report --openai
```

## Next useful work

1. Follow `docs/SOLUTION_TACTICS.md`: first improve feasible variable-size pilot allocation, with explicit stop reasons and comparison against the current agent.
2. Broaden candidate coverage before spending effort on wider beam search; no beam pruning occurred in the existing benchmark.
3. Compare a tabular ML model for historical ARPU-change prediction, calibrated with pilots, against the current statistical prior. Migration records alone do not identify causal conversion rates.
4. Evaluate the frozen OpenAI hypotheses before activating any exploration policy. API integration itself has no measured profit improvement.
5. Preserve deterministic submissions and disclose any losses when comparing strategies. Separate noise robustness from population-shift stress scenarios.

## Codex project setup

The user is adding this folder as a local Codex project named **PalliPalli**. Current conversation title: **Analyze hackathon brief**. Current task ID: `01a0cd30-0956-78b0-be93-aec6f042170d`.

The available app tools cannot create projects or reassign an existing task to a project. Computer automation of the Codex app was explicitly blocked. A user action in the app is required. In a new project task, ask Codex to read this file and continue; it contains the technical context without credentials. Creating a project does not by itself move the original conversation.
