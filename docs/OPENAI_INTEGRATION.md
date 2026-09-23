# Optional OpenAI analyst

The scored campaign agent runs offline. The optional analyst creates a report and proposes candidates for future investigation through the OpenAI Responses API. It does not modify `agent.py`, `planner.py`, or `submission.csv`.

## Install and run

```bash
pip install -r requirements-llm.txt
python analyst.py report
```

The command above is offline and makes no API request. It creates `artifacts/analyst/report.md` and `analysis.json`. For the API version, configure `OPENAI_API_KEY` in the shell or deployment environment, then run:

```bash
python analyst.py report --openai
```

The default model is the pinned snapshot `gpt-4.1-mini-2025-04-14`. Override it using `OPENAI_MODEL` or `--model` with a model accessible to your API project. `.env.example` documents the variable names; the program does not automatically load `.env` files. The optional dependencies are separate from the offline submission requirements.

The API uses Structured Outputs through `responses.create`, `store=False`, a maximum of 1,800 output tokens, a 20-second SDK request timeout, and zero automatic retries. Each invocation makes at most one API request. Missing keys, missing SDK, API failures, incomplete/refused output, and invalid suggestions produce a deterministic local report with a visible reason.

## What OpenAI receives and returns

Python builds aggregate public evidence and a catalog of at most 40 valid candidate campaigns. No subscriber IDs, individual records, private environment state, or true scored outcomes are sent.

For each candidate, Python matches pilot evidence and selected-plan membership using the complete campaign identity: all segment filters, current tariff, target tariff, and channel. A pilot on digital ads cannot be cited as a pilot on calls; a different destination cannot inherit another destination's result.

The model selects relevant factual observations and up to five candidate hypotheses. Factual statements are supplied as schema enums computed from the evidence. Python validates that each selected rationale belongs to the corresponding candidate and that the requested next experiment fits its pilot coverage. The available measurements are observed ARPU lift, contact count, and cost. Conversion and churn are not exposed by the case environment and cannot be requested as if they were.

This constrained design followed a live check in which a free-text draft mixed candidate references and suggested unavailable metrics. The final design uses OpenAI for prioritization, while Python supplies factual wording and numerical tables.

The report separates observed pilot spending, estimated final deployment spending, and conservative estimated net gain. It does not label an estimate as a true local or hidden judging score.

## Freeze hypotheses for evaluation

After a successful API run:

```bash
python analyst.py freeze artifacts/analyst/analysis.json --output policies/proposal-v1.json
```

The command validates the evidence hash, resolves candidate IDs to explicit campaign filters, records model/prompt provenance, and adds a content hash. It refuses to overwrite an existing frozen file.

**Freezing is not activation.** The file is marked `proposal_requires_evaluation` and `active_in_agent: false`. The scored agent does not read it. Before adopting a proposal, implement the applicable general exploration rule, compare it with the current agent using matched seeds, and reproduce `submission.csv`. A saved proposal is not a cached answer for hidden judging.

Live model responses are not guaranteed identical across runs. Keeping the API outside `Agent.act()` preserves campaign and CSV reproducibility even when the key is present.

## Validation

```bash
python -m unittest -v test_agent test_planner test_openai_analyst test_analyst_evidence
```

Checks cover the existing numerical agent, deterministic aggregate evidence, no subscriber IDs, valid catalog entries, unchanged agent output, optional API enablement, timeout/refusal/invalid-output fallbacks, bounded requests, candidate/rationale matching, allowed metrics, stable evidence hashes, and frozen proposal validation. Unit tests use a fake client and incur no API charges.

A live API example and validation status are recorded separately in `docs/OPENAI_VALIDATION.md`. API success demonstrates integration; it does not establish an improvement in campaign profit. Adoption of any proposed exploration rule remains future work.

## Official API references

- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Responses API migration and request shape](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [GPT-4.1 Mini snapshot and supported features](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
