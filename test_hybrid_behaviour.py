"""Integration checks against public pilot contracts and fixed seed 42.

The controlled environment belongs to the participant and exposes no hidden
effects. Official build_submission is used only for reproducibility and the
frozen/refactored control comparison; these tests do not inspect private state.
"""
from pathlib import Path
import unittest

import pandas as pd

from frozen_agent import Agent as FrozenAgent
from hybrid_agent import Agent, RefactoredControl
from make_submission import build_submission


ROOT = Path(__file__).resolve().parent

# Published participant-guide parameters, independently declared for this fake.
PUBLIC_CHANNELS = {
    "push": {"cost_per_contact": 0.0, "conversion_multiplier": 0.50},
    "sms": {"cost_per_contact": 4.0, "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22.0, "conversion_multiplier": 0.85},
    "call": {"cost_per_contact": 160.0, "conversion_multiplier": 1.20},
}


class PublicScenario:
    """Fixed observable pilot signals with public resource accounting."""

    def __init__(self, signal, budget=100000, contacts=15000, profile=None):
        self.customer_profile = (pd.read_csv(ROOT / "customer_profile.csv") if profile is None
                                 else profile.copy())
        self.tariffs = pd.read_csv(ROOT / "data/dict_tariff.csv")
        self.channels = {key: dict(value) for key, value in PUBLIC_CHANNELS.items()}
        self.remaining_budget = budget
        self.remaining_contacts = contacts
        self.pilots_left = 20
        self.pilot_history = []
        self.calls = []
        self.signal = signal

    def audience(self, campaign):
        rows = self.customer_profile
        for key, column in (("filter_current_tariff", "current_tariff"),
                            ("filter_arpu_segment", "arpu_segment"),
                            ("filter_data_segment", "data_segment"),
                            ("filter_call_segment", "call_segment")):
            value = campaign.get(key)
            if value is not None and str(value):
                rows = rows[rows[column].isin(str(value).split(";"))]
        return rows.sort_values("ID_NUMBER")

    def run_pilot(self, **request):
        campaign = dict(request)
        requested = campaign.pop("n_customers")
        assert int(requested) == requested and 10 <= requested <= 200
        assert self.pilots_left > 0
        assert campaign["target_tariff"] in set(self.tariffs.tariff_plan_code)
        cost = self.channels[campaign["channel"]]["cost_per_contact"]
        n = min(requested, len(self.audience(campaign)), self.remaining_contacts)
        if cost:
            n = min(n, int(self.remaining_budget // cost))
        assert n > 0
        self.remaining_budget -= n * cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        result = {"observed_lift_ratio": self.signal, "n_customers": n}
        self.pilot_history.append(dict(result))
        self.calls.append({"request": dict(request), "result": dict(result)})
        return result


class RecordingAgent:
    """Observe the agent's public calls without reading the simulator model."""

    def __init__(self, agent):
        self.agent = agent
        self.trace = []

    def act(self, env):
        original = env.run_pilot

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            self.trace.append({"request": dict(kwargs), "result": dict(result)})
            return result

        env.run_pilot = record
        try:
            return self.agent.act(env)
        finally:
            env.run_pilot = original


class HybridBehaviourTests(unittest.TestCase):
    def check_public_contract(self, scenario, plan):
        self.assertTrue(1 <= len(plan) <= 10)
        self.assertTrue(1 <= len(scenario.calls) <= 20)
        self.assertGreaterEqual(scenario.remaining_budget, 0)
        self.assertGreaterEqual(scenario.remaining_contacts, 0)
        names = [campaign["campaign_name"] for campaign in plan]
        self.assertEqual(len(names), len(set(names)))
        covered_cells = set()
        for campaign in plan:
            self.assertIn(campaign["target_tariff"], set(scenario.tariffs.tariff_plan_code))
            self.assertIn(campaign["channel"], scenario.channels)
            self.assertGreater(len(scenario.audience(campaign)), 0)
            if ";" in campaign["filter_current_tariff"]:
                self.assertLessEqual(len(scenario.audience(campaign)), 5000)
            for current in campaign["filter_current_tariff"].split(";"):
                self.assertNotEqual(current, campaign["target_tariff"])
                cell = (current, campaign["filter_arpu_segment"])
                self.assertNotIn(cell, covered_cells)
                covered_cells.add(cell)

    def test_pilot_sign_changes_final_decision_and_exposure(self):
        positive, negative = PublicScenario(0.5), PublicScenario(-0.5)
        good, bad = Agent().act(positive), Agent().act(negative)
        self.check_public_contract(positive, good)
        self.check_public_contract(negative, bad)
        self.assertNotEqual(good, bad)
        self.assertGreater(len(good), 1)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["campaign_name"], "minimum_exposure_fallback")
        self.assertEqual(bad[0]["channel"], "push")
        positive_coverage = sum(min(5000, len(positive.audience(c))) for c in good)
        negative_coverage = len(negative.audience(bad[0]))
        self.assertGreater(positive_coverage, negative_coverage)

    def test_zero_budget_retains_free_pilots_and_valid_plan(self):
        scenario = PublicScenario(0.5, budget=0)
        plan = Agent().act(scenario)
        self.check_public_contract(scenario, plan)
        self.assertEqual(scenario.remaining_budget, 0)
        self.assertTrue(all(campaign["channel"] == "push" for campaign in plan))
        self.assertTrue(all(call["request"]["channel"] == "push" for call in scenario.calls))

    def test_single_eligible_small_cell_is_not_discarded(self):
        full = pd.read_csv(ROOT / "customer_profile.csv")
        _, group = next((key, rows) for key, rows in
                        full.groupby(["current_tariff", "arpu_segment"])
                        if len(rows) >= 59)
        for population in (10, 37, 59):
            with self.subTest(population=population):
                scenario = PublicScenario(0.5, profile=group.iloc[:population])
                plan = Agent().act(scenario)
                self.check_public_contract(scenario, plan)
                self.assertEqual(len(plan), 1)
                self.assertTrue(all(call["result"]["n_customers"] <= population
                                    for call in scenario.calls))

    def test_no_eligible_ten_person_cell_reports_clear_failure(self):
        full = pd.read_csv(ROOT / "customer_profile.csv")
        scenario = PublicScenario(0.5, profile=full.iloc[:9])
        with self.assertRaisesRegex(ValueError, "at least 10 subscribers"):
            Agent().act(scenario)
        self.assertEqual(scenario.calls, [])

    def test_seed_42_official_submission_and_pilots_repeat(self):
        first, second = RecordingAgent(Agent()), RecordingAgent(Agent())
        submission1 = build_submission(first, seed=42)
        submission2 = build_submission(second, seed=42)
        pd.testing.assert_frame_equal(submission1, submission2)
        self.assertEqual(first.trace, second.trace)
        self.assertGreater(len(first.trace), 0)

    def test_refactored_control_preserves_frozen_seed_42_behavior(self):
        frozen, control = RecordingAgent(FrozenAgent()), RecordingAgent(RefactoredControl())
        original = build_submission(frozen, seed=42)
        refactored = build_submission(control, seed=42)
        self.assertEqual(frozen.trace, control.trace)
        pd.testing.assert_frame_equal(original, refactored)


if __name__ == "__main__":
    unittest.main()
