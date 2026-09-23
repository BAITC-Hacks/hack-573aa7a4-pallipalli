"""Paired local comparison; no hidden-judge score or data is used.

Run from the repository root: python benchmark.py
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from agent import Agent
from agent_template import Agent as BaselineAgent
from local_eval import evaluate_agent
from make_submission import build_submission
from scoring_core import validate_strategy


class CheckedAgent:
    def __init__(self, implementation):
        self.implementation = implementation
        self.errors = []
        self.campaigns = None
        self.pilot_calls = 0
        self.elapsed = 0.0

    def act(self, env):
        original = env.run_pilot
        def checked_pilot(*args, **kwargs):
            requested = kwargs.get("n_customers", 100)
            assert 10 <= requested <= 200, "Pilot size outside allowed range"
            result = original(*args, **kwargs)
            self.pilot_calls += 1
            assert self.pilot_calls <= 20
            assert env.remaining_budget >= 0 and env.remaining_contacts >= 0
            return result
        env.run_pilot = checked_pilot
        start = time.perf_counter()
        try:
            self.campaigns = self.implementation.act(env)
            validate_strategy(pd.DataFrame(self.campaigns), env.tariffs)
            if not 1 <= len(self.campaigns) <= 10:
                self.errors.append("Final campaign count outside 1–10")
            if self.pilot_calls == 0:
                self.errors.append("No pilots")
            return self.campaigns
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.elapsed = time.perf_counter() - start
            if self.elapsed > 300:
                self.errors.append("Agent runtime exceeded five minutes")


def run(implementation, seed):
    checked = CheckedAgent(implementation())
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        score = evaluate_agent(checked, seed=seed, verbose=False)
    assert score is not None, log.getvalue()
    assert score["total_cost"] <= 100000
    assert score["total_contacts"] <= 15000
    assert all(c["n_contacts"] <= 5000 for c in score["campaigns_detail"])
    return dict(score, seed=seed, final_campaigns=len(checked.campaigns or []),
                act_seconds=checked.elapsed, validation_errors=checked.errors,
                evaluator_messages=log.getvalue())


def summarize(rows):
    net = np.array([r["net_arpu_gain"] for r in rows])
    return {"runs": len(rows), "profitable": int((net > 0).sum()),
            "mean_net": float(net.mean()), "median_net": float(np.median(net)),
            "min_net": float(net.min()), "max_net": float(net.max()),
            "valid_runs": sum(not r["validation_errors"] for r in rows),
            "max_cost": max(r["total_cost"] for r in rows),
            "max_contacts": max(r["total_contacts"] for r in rows),
            "max_pilots": max(r["n_pilots"] for r in rows),
            "max_act_seconds": max(r["act_seconds"] for r in rows)}


def main():
    # The same predeclared seed set is used for both implementations. This
    # explores pilot randomness, not independent population distributions.
    seeds = [42] + list(range(30)) + list(range(100, 130))
    rows = {"baseline": [], "new": []}
    for seed in seeds:
        for label, implementation in [("baseline", BaselineAgent), ("new", Agent)]:
            rows[label].append(run(implementation, seed))
    assert all(not r["validation_errors"] for r in rows["new"]), "New agent failed validation"
    assert all(not r["evaluator_messages"] for r in rows["new"]), "New agent emitted evaluator warnings"
    submission = build_submission(Agent()).to_csv(index=False)
    repeat = build_submission(Agent()).to_csv(index=False)
    assert submission == repeat, "Seed-42 submission is not reproducible"
    Path("submission.csv").write_text(submission)
    files = ["agent.py", "agent_template.py", "environment.py", "local_eval.py",
             "scoring_core.py", "mock_environment.py", "customer_profile.csv",
             "data/change_tariff.csv", "data/dict_tariff.csv", "submission.csv"]
    result = {
        "seed_set": seeds,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        "summary": {k: summarize(v) for k, v in rows.items()},
        "paired_wins": sum(b["net_arpu_gain"] > a["net_arpu_gain"]
                           for a, b in zip(rows["baseline"], rows["new"])),
        "submission_reproducible": True,
        "runs": rows,
    }
    Path("docs").mkdir(exist_ok=True)
    metrics = ["seed", "net_arpu_gain", "gross_arpu_lift", "total_cost", "total_contacts",
               "unique_customers_targeted", "n_pilots", "final_campaigns", "risk_score_pct",
               "act_seconds", "validation_errors", "evaluator_messages"]
    flat = [dict(agent=label, **{k: r[k] for k in metrics})
            for label, scores in rows.items() for r in scores]
    pd.DataFrame(flat).to_csv("docs/benchmark-runs.csv", index=False)
    compact = {k: v for k, v in result.items() if k != "runs"}
    compact["seed42"] = {label: scores[0] for label, scores in rows.items()}
    Path("docs/benchmark-results.json").write_text(json.dumps(compact, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "sha256", "seed_set")}, indent=2))
    print("Seed 42:")
    for label, scores in rows.items():
        r = scores[0]
        print(label, {k:r[k] for k in ["net_arpu_gain", "gross_arpu_lift", "total_cost", "total_contacts", "n_pilots", "final_campaigns", "risk_score_pct"]})


if __name__ == "__main__":
    main()
