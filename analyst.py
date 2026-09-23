"""Optional analyst CLI; does not alter the scored agent or submission.csv."""
import argparse
import json
from pathlib import Path

from analyst_evidence import build_evidence, render_report
from openai_analyst import analyze, freeze_hypotheses, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Create a deterministic report, optionally with OpenAI commentary")
    report.add_argument("--openai", action="store_true", help="Allow one paid API request using OPENAI_API_KEY")
    report.add_argument("--model", default=None, help="Override OPENAI_MODEL and the default snapshot")
    report.add_argument("--seed", type=int, default=42)
    report.add_argument("--output-dir", default="artifacts/analyst")
    freeze = commands.add_parser("freeze", help="Freeze model-proposed candidates for subsequent evaluation")
    freeze.add_argument("analysis_json")
    freeze.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        try:
            bundle = json.loads(Path(args.analysis_json).read_text())
            policy = freeze_hypotheses(bundle)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            parser.error(f"Cannot freeze proposal: {type(exc).__name__}. Provide a valid successful analyst artifact.")
        target = Path(args.output)
        if target.exists():
            parser.error("Output exists; choose a new filename to preserve the frozen proposal.")
        write_json(target, policy)
        print(f"Frozen {len(policy['hypotheses'])} proposals: {target}")
        print("Saved for evaluation; the scored agent has not been changed.")
        return
    evidence = build_evidence(seed=args.seed)
    analysis = analyze(evidence, enabled=args.openai, model=args.model)
    output = Path(args.output_dir)
    write_json(output / "analysis.json", {"evidence": evidence, "analysis": analysis})
    report_text = render_report(evidence, commentary=analysis["commentary"])
    report_text += f"\n## Analyst mode\n\nStatus: {analysis['status']}. Reason: {analysis['reason'] or 'completed'}.\n"
    report_text += "\nThis report does not modify the campaign plan or submission CSV.\n"
    (output / "report.md").write_text(report_text)
    print(f"Report: {output / 'report.md'}")
    print(f"Evidence and suggestions: {output / 'analysis.json'}")
    print(f"Analyst mode: {analysis['status']} ({analysis['reason'] or 'completed'})")


if __name__ == "__main__":
    main()
