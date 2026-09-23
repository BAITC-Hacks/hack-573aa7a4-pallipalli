"""Isolate plan selection: frozen greedy agent versus plan search, same pilots."""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from agent import Agent
from greedy_agent import Agent as GreedyAgent
from benchmark import run, summarize
from make_submission import build_submission


def main():
    # Original comparison seeds, plus 30 new seeds fixed before this experiment.
    original = [42] + list(range(30)) + list(range(100, 130))
    fresh = list(range(200, 230))
    rows = {"greedy": [], "plan_search": []}
    for seed in original + fresh:
        old = run(GreedyAgent, seed)
        new = run(Agent, seed)
        assert old["pilot_trace"] == new["pilot_trace"], f"Pilot policy changed at seed {seed}"
        assert not new["validation_errors"] and not new["evaluator_messages"], new
        d = new["plan_diagnostics"]
        assert d["estimated_net"] + 1e-6 >= d["greedy_estimated_net"]
        rows["greedy"].append(old)
        rows["plan_search"].append(new)
    submission = build_submission(Agent()).to_csv(index=False)
    assert submission == build_submission(Agent()).to_csv(index=False)
    Path("submission.csv").write_text(submission)
    deltas = np.array([b["net_arpu_gain"] - a["net_arpu_gain"]
                       for a, b in zip(rows["greedy"], rows["plan_search"])])
    comparison = {
        "wins": int((deltas > 1e-6).sum()), "ties": int((np.abs(deltas) <= 1e-6).sum()),
        "losses": int((deltas < -1e-6).sum()), "mean_delta": float(deltas.mean()),
        "min_delta": float(deltas.min()), "max_delta": float(deltas.max()),
    }
    metrics = ["seed", "net_arpu_gain", "gross_arpu_lift", "total_cost", "total_contacts",
               "unique_customers_targeted", "n_pilots", "final_campaigns", "risk_score_pct",
               "act_seconds", "validation_errors", "evaluator_messages"]
    flat = [dict(agent=label, **{k: r[k] for k in metrics},
                 estimated_net=r["plan_diagnostics"].get("estimated_net"),
                 greedy_estimated_net=r["plan_diagnostics"].get("greedy_estimated_net"),
                 states_pruned=r["plan_diagnostics"].get("states_pruned"))
            for label, scores in rows.items() for r in scores]
    Path("docs").mkdir(exist_ok=True)
    pd.DataFrame(flat).to_csv("docs/planner-runs.csv", index=False)
    paths = ["agent.py", "planner.py", "greedy_agent.py", "compare_planners.py", "benchmark.py",
             "environment.py", "mock_environment.py", "local_eval.py", "scoring_core.py",
             "customer_profile.csv", "data/change_tariff.csv", "data/dict_tariff.csv", "submission.csv"]
    result = {
        "reference_commit": "6e4c9ba9ec9de63510c391fe742fe669e07763fc",
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "original_seeds": original, "fresh_seeds": fresh,
        "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        "summary": {k: summarize(v) for k, v in rows.items()},
        "original_seed_summary": {k: summarize(v[:len(original)]) for k, v in rows.items()},
        "fresh_seed_summary": {k: summarize(v[len(original):]) for k, v in rows.items()},
        "comparison": comparison, "identical_pilots_all_seeds": True,
        "submission_reproducible": True,
        "campaign_count_distribution": {label: pd.Series([r["final_campaigns"] for r in scores]).value_counts().sort_index().to_dict()
                                        for label, scores in rows.items()},
        "seed42": {label: {k: v for k, v in scores[0].items() if k != "pilot_trace"}
                   for label, scores in rows.items()},
    }
    Path("docs/planner-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("sha256", "seed42", "original_seeds", "fresh_seeds")}, indent=2))
    for label, scores in rows.items():
        print(label, "seed42", {k: scores[0][k] for k in ["net_arpu_gain", "final_campaigns", "total_cost"]})


if __name__ == "__main__":
    main()
