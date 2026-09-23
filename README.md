# PalliPalli — Beeline campaign agent

The current agent combines evidence-dependent historical priors, information-value pilot selection, and compatible campaign grouping with whole-plan resource allocation.

## Run and submit

```bash
pip install -r requirements.txt
python local_eval.py
python make_submission.py
```

Submit **all seven runtime Python files**: `agent.py`, `hybrid_agent.py`, `frozen_agent.py`, `planner.py`, `prior_model.py`, `kg_policy.py`, `grouped_planner.py`; plus `submission.csv`, `requirements.txt` and the supplied `data/change_tariff.csv`. `agent.py` imports the implementation modules. Keep organizer environment, evaluator and data files unchanged.

No API key, LLM service, GPU or network connection is required for scored decisions.

## Latest experiment

On 50 new matched seeds 500–549:

| Policy | Mean net gain | Valid/profitable |
|---|---:|---:|
| Previous local best |4,038,323|50/50|
| `feat/agent-solution` at `3cf067f` |4,842,687|50/50|
| Combined implementation |**4,871,270**|**50/50**|

Gain versus previous best: **20.63%**, 50 paired wins. Versus source branch: **0.59%**, 35 wins, 2 ties, 13 losses. The combined version was selected on development data before the new 50-seed evaluation.

At seed 42: net 4,730,893, 20 pilots, 6 final campaigns, spend 99,980, total contacts 9,308. The source branch is 41,486 better at this single seed. These are synthetic mock results, not hidden judging scores.

[Full review, ablations and regressions](docs/BRANCH_EXPERIMENT.md) · [All scores](results/validation/runs.csv)

## Reproduce the experiment

```bash
python run_experiment.py --split development
python run_experiment.py --split validation
python -m unittest -v test_prior_model test_kg_policy test_grouped_planner test_planner test_hybrid_behaviour
```

The comparison holds the old agent and source branch fixed, adds each technique separately, and compares combinations. Changing only exploration or only optimistic shortlisting hurt; the measured gain belongs to the working combination. Original data and policy hashes are checked on each run. Future work should use new unobserved seeds.

## Optional analyst

The existing offline/OpenAI analyst can explain public evidence separately from the scored agent. It does not control submitted decisions. Its21 offline/fake-client regression tests pass with the new agent. See the existing integration documentation when using that optional component.

## Source and limitations

The new ideas were adapted from [feat/agent-solution](https://github.com/BAITC-Hacks/hack-573aa7a4-pallipalli/tree/3cf067f7ce038a0a56145bda0a59e261e9388392). The exact reference implementation is preserved in `branch_agent.py`; `frozen_agent.py` preserves our preceding best.

The mock population shares information with migration history. Seed variation measures pilot noise, not population shift. The prior uncertainty, channel transfer, pilot overlap and information-value opportunity cost are approximations. Free fallback contacts can still reduce revenue; beam search protects estimated value rather than true hidden profit.
