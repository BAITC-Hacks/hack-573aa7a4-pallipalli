"""Reproducible comparisons using the unchanged participant evaluator.

Seeds 300–349 were reserved before evaluating the new adaptive-size policy.
No API calls are made. Each implementation receives only the public environment.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from local_eval import evaluate_agent
from make_submission import build_submission
from scoring_core import validate_strategy

ROOT = Path(__file__).resolve().parent
STAGES = {
    "starter": ("agent_template", "Agent", "Organizer plain Python starter"),
    "history_only": ("stage_baselines", "HistoryOnlyAgent", "Historical shortlist + fixed SMS pilots"),
    "history_uncertainty": ("greedy_agent", "Agent", "History, uncertainty, adaptive repeats, channel choice, greedy allocation"),
    "whole_plan": ("plan_search_agent", "Agent", "Previous stage + whole-plan search"),
    "cheap_probes": ("revenue_conservative_agent", "Agent", "Broader shortlist, cheap normalized probes, channel alternatives, expected overlap"),
    "expected_value": ("current_agent", "Agent", "Previous stage + expected gain with uncertainty eligibility gate"),
    "adaptive_sizes": ("adaptive_agent", "Agent", "Previous stage + adaptive pilot sample sizes"),
    "current_greedy": ("stage_baselines", "CurrentGreedyAgent", "Control: current estimates/pilots, greedy final allocation"),
}


class CheckedAgent:
    """Instrument public pilot calls and validate raw output before sanitizing."""

    def __init__(self, implementation):
        self.implementation = implementation
        self.errors = []
        self.plan = []
        self.trace = []
        self.seconds = 0.0

    def act(self, env):
        original = env.run_pilot

        def checked_pilot(*args, **kwargs):
            requested = kwargs.get("n_customers", 100)
            if not 10 <= requested <= 200 or int(requested) != requested:
                raise ValueError("Pilot request outside integer range 10–200")
            result = original(*args, **kwargs)
            self.trace.append({"request": dict(kwargs), "result": dict(result)})
            if len(self.trace) > 20 or min(env.remaining_budget, env.remaining_contacts, env.pilots_left) < 0:
                raise ValueError("Pilot resource limit exceeded")
            return result

        env.run_pilot = checked_pilot
        start = time.perf_counter()
        try:
            self.plan = self.implementation.act(env)
            validate_strategy(pd.DataFrame(self.plan), env.tariffs)
            if not 1 <= len(self.plan) <= 10:
                self.errors.append("Final campaign count outside 1–10")
            if not self.trace:
                self.errors.append("No pilots")
            return self.plan
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            env.run_pilot = original
            self.seconds = time.perf_counter() - start
            if self.seconds > 600:
                self.errors.append("Runtime exceeded participant guide's 10-minute limit")


def implementation(stage):
    module, name, _ = STAGES[stage]
    return getattr(importlib.import_module(module), name)


def run(stage, seed):
    checked = CheckedAgent(implementation(stage)())
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        score = evaluate_agent(checked, seed=seed, verbose=False)
    if score is None:
        raise RuntimeError(f"{stage}, seed {seed}: evaluator returned no score: {output.getvalue()}")
    if score["total_cost"] > 100000 or score["total_contacts"] > 15000:
        checked.errors.append("Scored money/contact limit exceeded")
    if any(c["n_contacts"] > 5000 for c in score["campaigns_detail"]):
        checked.errors.append("Scored campaign exceeds 5000 contacts")
    if output.getvalue().strip():
        checked.errors.append("Evaluator emitted messages; inspect evaluator_messages")
    return dict(score, stage=stage, seed=seed, valid=not checked.errors,
                validation_errors=checked.errors, evaluator_messages=output.getvalue(),
                act_seconds=checked.seconds, final_campaigns=len(checked.plan or []),
                plan=checked.plan, pilot_trace=checked.trace,
                plan_diagnostics=getattr(checked.implementation, "plan_diagnostics", {}))


def summary(rows):
    values = np.array([r["net_arpu_gain"] for r in rows])
    return {"runs": len(rows), "mean_net": float(values.mean()),
            "median_net": float(np.median(values)), "p10_net": float(np.quantile(values, .1)),
            "worst_net": float(values.min()), "best_net": float(values.max()),
            "profitable": int((values > 0).sum()), "valid": sum(r["valid"] for r in rows),
            "max_cost": max(r["total_cost"] for r in rows),
            "max_contacts": max(r["total_contacts"] for r in rows),
            "max_pilots": max(r["n_pilots"] for r in rows),
            "mean_pilots": float(np.mean([r["n_pilots"] for r in rows])),
            "max_act_seconds": max(r["act_seconds"] for r in rows)}


def paired(first, second):
    assert [r["seed"] for r in first] == [r["seed"] for r in second]
    difference = np.array([b["net_arpu_gain"] - a["net_arpu_gain"] for a, b in zip(first, second)])
    rng = np.random.default_rng(573)
    bootstrap = rng.choice(difference, (5000, len(difference)), replace=True).mean(axis=1)
    return {"mean_delta": float(difference.mean()), "median_delta": float(np.median(difference)),
            "worst_delta": float(difference.min()), "best_delta": float(difference.max()),
            "wins": int((difference > 1e-6).sum()), "ties": int((np.abs(difference) <= 1e-6).sum()),
            "losses": int((difference < -1e-6).sum()),
            "mean_delta_bootstrap_95pct": [float(v) for v in np.quantile(bootstrap, [.025, .975])],
            "identical_pilot_traces": all(a["pilot_trace"] == b["pilot_trace"] for a, b in zip(first, second)),
            "regressions": [{"seed": a["seed"], "delta": float(d)} for a, d in zip(first, difference) if d < -1e-6]}


def verify_official_files():
    snapshot = json.loads((ROOT / "snapshot.json").read_text())
    official = ["environment.py", "mock_environment.py", "scoring_core.py", "local_eval.py",
                "make_submission.py", "agent_template.py", "customer_profile.csv",
                "feature_dictionary.csv", "tariff_dictionary.csv", "data/change_tariff.csv",
                "data/traffic.csv", "data/arpu_monthly.csv", "data/dict_tariff.csv"]
    for name in official:
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == snapshot["sha256"][name], f"Organizer file changed: {name}"
    return official


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "validation"], default="validation")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    args = parser.parse_args()
    official = verify_official_files()
    seeds = [42] + list(range(10)) if args.split == "development" else list(range(300, 350))
    output = ROOT / "results" / args.split
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for stage in args.stages:
        rows[stage] = []
        for seed in seeds:
            rows[stage].append(run(stage, seed))
        (output / f"{stage}.json").write_text(json.dumps(rows[stage], indent=2))
        print(stage, json.dumps(summary(rows[stage])), flush=True)
    comparisons = {}
    for first, second in [("starter", "history_only"), ("history_only", "history_uncertainty"),
                          ("history_uncertainty", "whole_plan"), ("whole_plan", "cheap_probes"),
                          ("cheap_probes", "expected_value"), ("expected_value", "adaptive_sizes"),
                          ("current_greedy", "expected_value"), ("starter", "expected_value")]:
        if first in rows and second in rows:
            comparisons[f"{second}_minus_{first}"] = paired(rows[first], rows[second])
    source_paths = list(ROOT.glob("*.py")) + [ROOT / f for f in official if not f.endswith(".py")]
    report = {"split": args.split, "seeds": seeds, "python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__,
              "official_files_unchanged": True,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "stages": {s: STAGES[s][2] for s in rows},
              "summary": {s: summary(r) for s, r in rows.items()}, "comparisons": comparisons,
              "limits": "Seeds vary pilot randomness on the fixed mock population; intervals are not hidden-judge performance bounds."}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    flat = [{k: r[k] for k in ["stage", "seed", "net_arpu_gain", "gross_arpu_lift", "total_cost",
                              "total_contacts", "n_pilots", "final_campaigns", "valid", "act_seconds"]}
            for values in rows.values() for r in values]
    pd.DataFrame(flat).to_csv(output / "runs.csv", index=False)
    verify_official_files()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
