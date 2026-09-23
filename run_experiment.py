"""Freeze and compare feature-branch ideas with the official local evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import compare_stages as evaluator

ROOT = Path(__file__).resolve().parent
STAGES = {
    "frozen": ("frozen_agent", "Agent", "Our previous best, unchanged"),
    "branch": ("branch_reference", "Agent", "Feature branch at 3cf067f, only logging disabled"),
    "control": ("hybrid_agent", "RefactoredControl", "Refactor equivalence control"),
    "prior": ("hybrid_agent", "PriorOnly", "Only hierarchical mean and evidence-dependent prior uncertainty"),
    "kg": ("hybrid_agent", "KnowledgeGradientOnly", "Only knowledge-gradient pilot candidate/sample-size choice"),
    "grouping": ("hybrid_agent", "GroupingOnly", "Only compatible campaign merging"),
    "broad": ("hybrid_agent", "BroadOnly", "Only optimistic four-target shortlist"),
    "prior_kg": ("hybrid_agent", "PriorKnowledgeGradient", "Prior + KG + broader shortlist"),
    "combined": ("hybrid_agent", "Agent", "Prior + KG + shortlist + grouping with whole-plan search"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    args = parser.parse_args()
    evaluator.STAGES = STAGES
    evaluator.verify_official_files()
    manifest = json.loads((ROOT / "experiment.json").read_text())
    for name, digest in manifest["frozen_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    seeds = manifest[f"{args.split}_seeds"]
    destination = ROOT / "results" / args.split
    destination.mkdir(parents=True, exist_ok=True)
    hashes_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    if args.split == "validation":
        selection = json.loads((ROOT / "development-selection.json").read_text())
        for name, digest in selection["source_sha256"].items():
            assert hashes_before[name] == digest, f"Policy changed since development selection: {name}"
    else:
        selection = None
    rows = {}
    for stage in args.stages:
        rows[stage] = [evaluator.run(stage, seed) for seed in seeds]
        (destination / f"{stage}.json").write_text(json.dumps(rows[stage], indent=2))
        print(stage, json.dumps(evaluator.summary(rows[stage])), flush=True)
    if "frozen" in rows and "control" in rows:
        assert all(a["pilot_trace"] == b["pilot_trace"] and a["plan"] == b["plan"]
                   and a["net_arpu_gain"] == b["net_arpu_gain"]
                   for a, b in zip(rows["frozen"], rows["control"])), "Refactor changed frozen policy"
    comparisons = {}
    for reference in ["frozen", "branch", "prior_kg"]:
        if reference not in rows:
            continue
        for candidate in rows:
            if candidate != reference:
                comparisons[f"{candidate}_minus_{reference}"] = evaluator.paired(rows[reference], rows[candidate])
    hashes_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    for name, digest in hashes_before.items():
        assert hashes_after[name] == digest, f"Source changed during evaluation: {name}"
    report = {"split": args.split, "seeds": seeds, "source_commit": manifest["source_commit"],
              "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
              "source_sha256": hashes_before, "selection_before_validation": selection,
              "summary": {label: evaluator.summary(values) for label, values in rows.items()},
              "comparisons": comparisons, "official_files_unchanged": True,
              "limits": "Seeds vary sampling/noise on one synthetic mock population, not hidden-judge populations."}
    (destination / "summary.json").write_text(json.dumps(report, indent=2))
    columns = ["stage", "seed", "net_arpu_gain", "gross_arpu_lift", "total_cost", "total_contacts",
               "n_pilots", "final_campaigns", "valid", "act_seconds"]
    pd.DataFrame([{k: row[k] for k in columns} for values in rows.values() for row in values]).to_csv(
        destination / "runs.csv", index=False)
    evaluator.verify_official_files()
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
