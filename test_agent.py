"""Public-interface checks for policy behavior under controlled pilot signals."""
import unittest

import pandas as pd

from agent import Agent
from make_submission import build_submission
from mock_environment import CHANNELS
from scoring_core import validate_strategy


class PublicScenario:
    """No effect model or private judging state is exposed to the agent."""
    def __init__(self, signal, budget=100000, contacts=15000):
        self.customer_profile = pd.read_csv("customer_profile.csv")
        self.tariffs = pd.read_csv("data/dict_tariff.csv")
        self.channels = {k: dict(v) for k, v in CHANNELS.items()}
        self.remaining_budget = budget
        self.remaining_contacts = contacts
        self.pilots_left = 20
        self.pilot_history = []
        self.calls = []
        self.signal = signal

    def run_pilot(self, **campaign):
        requested = campaign.pop("n_customers")
        assert 10 <= requested <= 200 and self.pilots_left > 0
        rows = self.customer_profile
        for column, key in [("current_tariff", "filter_current_tariff"),
                            ("arpu_segment", "filter_arpu_segment")]:
            rows = rows[rows[column] == campaign[key]]
        cost = self.channels[campaign["channel"]]["cost_per_contact"]
        n = min(requested, len(rows), self.remaining_contacts)
        if cost:
            n = min(n, int(self.remaining_budget // cost))
        assert n > 0
        self.remaining_budget -= n * cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        result = {"observed_lift_ratio": self.signal, "n_customers": n}
        self.pilot_history.append(result)
        self.calls.append(campaign)
        return result


class NoHistoryAgent(Agent):
    def _history(self):
        return {}


class AgentTests(unittest.TestCase):
    def check_plan(self, env, plan):
        self.assertTrue(1 <= len(plan) <= 10)
        validate_strategy(pd.DataFrame(plan), env.tariffs)
        self.assertTrue(0 < len(env.calls) <= 20)
        self.assertGreaterEqual(env.remaining_budget, 0)
        self.assertGreaterEqual(env.remaining_contacts, 0)
        cells = [(c["filter_current_tariff"], c["filter_arpu_segment"]) for c in plan]
        self.assertEqual(len(cells), len(set(cells)))
        self.assertTrue(all(c["target_tariff"] != c["filter_current_tariff"] for c in plan))

    def test_pilot_results_change_plan(self):
        positive = PublicScenario(0.5)
        negative = PublicScenario(-0.5)
        good = Agent().act(positive)
        bad = Agent().act(negative)
        self.check_plan(positive, good)
        self.check_plan(negative, bad)
        self.assertNotEqual(good, bad)
        self.assertEqual(bad[0]["campaign_name"], "minimum_exposure_fallback")
        self.assertEqual(bad[0]["channel"], "push")

    def test_zero_budget_uses_free_channels(self):
        env = PublicScenario(0.5, budget=0)
        plan = Agent().act(env)
        self.check_plan(env, plan)
        self.assertTrue(all(c["channel"] == "push" for c in plan + env.calls))

    def test_missing_history_still_uses_pilots(self):
        env = PublicScenario(0.5)
        plan = NoHistoryAgent().act(env)
        self.check_plan(env, plan)

    def test_weak_pilots_trigger_bounded_fallback(self):
        env = PublicScenario(0.0)
        plan = NoHistoryAgent().act(env)
        self.check_plan(env, plan)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["channel"], "push")

    def test_default_submission_is_deterministic(self):
        first = build_submission(Agent()).to_csv(index=False)
        second = build_submission(Agent()).to_csv(index=False)
        self.assertEqual(first, second)

    def test_posterior_moves_with_pilot_evidence(self):
        agent = Agent()
        prior = {"prior": 0.1, "observations": []}
        negative = dict(prior, observations=[(-0.5, 200)])
        repeated = dict(prior, observations=[(-0.5, 200), (-0.5, 200)])
        self.assertLess(agent._estimate(negative)[0], 0)
        self.assertLess(agent._estimate(repeated)[1], agent._estimate(negative)[1])


if __name__ == "__main__":
    unittest.main()
