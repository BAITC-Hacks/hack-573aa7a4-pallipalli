"""Compare current grouped versus ungrouped planning with identical pilots.

The original comparison assumed the old greedy agent's exploration policy.
The current experiment uses prior_kg and combined, which differ only in the
final grouped planner. Full stage comparisons live in run_experiment.py.
"""
import sys

from run_experiment import main


if __name__ == "__main__":
    if "--stages" not in sys.argv:
        sys.argv.extend(["--stages", "prior_kg", "combined"])
    main()
