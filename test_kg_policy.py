"""Controlled public-interface scenarios for knowledge-gradient exploration."""
import math
import unittest

from kg_policy import explore, expected_excess, information_value


class PosteriorAgent:
    NOISE = 0.804
    KG_KAPPA = 1.0

    def __init__(self, contacts=4000, reserve=84000):
        self.pilot_contacts_left = contacts
        self.reserve_money = reserve

    def _estimate(self, c):
        precision = 1.0 / c["sd"] ** 2
        weighted = c["prior"] * precision
        for ratio, n, multiplier in c["observations"]:
            p = n * multiplier ** 2 / self.NOISE ** 2
            precision += p
            weighted += ratio / multiplier * p
        sd = math.sqrt(1.0 / precision)
        mean = weighted / precision
        return mean, sd, mean - self.KG_KAPPA * sd


class PublicScenario:
    def __init__(self, signals=None, budget=100000, contacts=15000, pilots=20,
                 invalid=None, channels=None):
        self.channels = channels if channels is not None else {
            "push": {"cost_per_contact": 0, "conversion_multiplier": 0.5},
            "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
        }
        self.remaining_budget = budget
        self.remaining_contacts = contacts
        self.pilots_left = pilots
        self.pilot_history = []
        self.signals = signals or {}
        self.invalid = invalid

    def run_pilot(self, **campaign):
        n = campaign["n_customers"]
        cost = self.channels[campaign["channel"]]["cost_per_contact"]
        multiplier = self.channels[campaign["channel"]]["conversion_multiplier"]
        assert 10 <= n <= 200
        assert self.pilots_left > 0 and n <= self.remaining_contacts
        assert n * cost <= self.remaining_budget
        self.pilots_left -= 1
        actual = 0 if self.invalid == "zero" else n
        observed = self.signals.get(campaign["target_tariff"], 0.1) * multiplier
        if self.invalid == "nan":
            observed = float("nan")
        self.remaining_budget -= actual * cost
        self.remaining_contacts -= actual
        self.pilot_history.append(dict(campaign))
        return {"n_customers": actual, "observed_lift_ratio": observed}


def candidate(target="target", cell="cell", mean=0.05, sd=0.25, population=1000,
              arpu=10000):
    return {"cell": cell, "population": population, "prior": mean, "sd": sd,
            "values": [float(arpu)] * min(population, 5000), "observations": [],
            "campaign": {"filter_current_tariff": cell, "filter_arpu_segment": "HIGH",
                         "target_tariff": target}}


class KnowledgeGradientTests(unittest.TestCase):
    def test_expected_excess_known_distribution(self):
        self.assertAlmostEqual(expected_excess(0, 1, 0), 1 / math.sqrt(2 * math.pi))
        self.assertEqual(expected_excess(1, 0, 2), 0)
        self.assertEqual(expected_excess(2, 0, 1), 1)
        self.assertEqual(expected_excess(-100, 0.01, 0), 0)

    def test_more_informative_probe_has_greater_gross_value(self):
        args = dict(mean=0.1, sd=0.25, alternative=0.0, stake=1000000,
                    probe_multiplier=0.65, deployment_multiplier=0.65)
        self.assertGreater(information_value(n=200, **args), information_value(n=40, **args))
        self.assertEqual(information_value(n=0, **args), 0)
        self.assertEqual(information_value(n=40, **dict(args, sd=0)), 0)

    def test_information_can_outrank_a_high_but_certain_mean(self):
        certain = candidate("certain", "known", mean=0.8, sd=0.001)
        uncertain = candidate("uncertain", "unknown", mean=0.05, sd=0.25)
        env = PublicScenario(pilots=1)
        explore(PosteriorAgent(), env, [certain, uncertain])
        self.assertEqual(env.pilot_history[0]["target_tariff"], "uncertain")

    def test_dominant_alternative_suppresses_information_value(self):
        args = dict(mean=0.05, sd=0.25, n=200, stake=1000000,
                    probe_multiplier=0.65, deployment_multiplier=0.65)
        self.assertGreater(information_value(alternative=0, **args),
                           information_value(alternative=0.8, **args))

    def test_negative_evidence_changes_next_target(self):
        a, b = candidate("a"), candidate("b")
        env = PublicScenario(signals={"a": -1.0, "b": 0.5}, pilots=2)
        explore(PosteriorAgent(), env, [a, b])
        self.assertEqual([c["target_tariff"] for c in env.pilot_history], ["a", "b"])
        self.assertLess(a["observations"][0][0], 0)

    def test_negative_evidence_stops_when_information_is_not_worth_cost(self):
        c = candidate()
        env = PublicScenario(signals={"target": -2.0})
        diagnostics = explore(PosteriorAgent(), env, [c])
        self.assertEqual(len(env.pilot_history), 1)
        self.assertEqual(diagnostics["stop_reason"], "information_value_below_cost")
        self.assertGreater(env.pilots_left, 0)

    def test_budget_contacts_population_and_reserve_limits(self):
        c = candidate(population=135)
        env = PublicScenario(budget=84020, contacts=83)
        agent = PosteriorAgent(contacts=67)
        diagnostics = explore(agent, env, [c])
        self.assertGreater(len(env.pilot_history), 0)
        self.assertLessEqual(diagnostics["pilot_contacts"], 67)
        self.assertGreaterEqual(env.remaining_budget, 84000)
        self.assertGreaterEqual(agent.pilot_contacts_left, 0)
        for call in env.pilot_history:
            self.assertTrue(10 <= call["n_customers"] <= 67)

    def test_zero_budget_uses_free_probe_and_normalizes_observation(self):
        c = candidate()
        env = PublicScenario(budget=0, pilots=1, signals={"target": 0.4})
        explore(PosteriorAgent(), env, [c])
        self.assertEqual(env.pilot_history[0]["channel"], "push")
        ratio, n, multiplier = c["observations"][0]
        self.assertEqual(multiplier, 0.5)
        self.assertAlmostEqual(ratio / multiplier, 0.4)
        self.assertTrue(10 <= n <= 200)

    def test_required_pilot_on_small_certain_negative_population(self):
        c = candidate(mean=-1, sd=0.001, population=10, arpu=100)
        env = PublicScenario(budget=0, contacts=10)
        diagnostics = explore(PosteriorAgent(), env, [c])
        self.assertEqual(len(env.pilot_history), 1)
        self.assertEqual(env.pilot_history[0]["n_customers"], 10)
        self.assertTrue(diagnostics["pilots"][0]["mandatory"])
        self.assertLess(diagnostics["pilots"][0]["acquisition_score"], 0)

    def test_required_first_pilot_can_use_reserved_cash_if_no_free_channel(self):
        env = PublicScenario(budget=40, contacts=10, channels={
            "sms": {"conversion_multiplier": 0.65, "cost_per_contact": 4}})
        diagnostics = explore(PosteriorAgent(), env, [candidate(population=10)])
        self.assertEqual(len(env.pilot_history), 1)
        self.assertTrue(diagnostics["pilots"][0]["reserve_override"])
        self.assertEqual(env.remaining_budget, 0)

    def test_invalid_results_do_not_divide_by_zero_or_corrupt_posterior(self):
        for invalid in ("zero", "nan"):
            with self.subTest(invalid=invalid):
                c = candidate()
                env = PublicScenario(invalid=invalid)
                agent = PosteriorAgent()
                diagnostics = explore(agent, env, [c])
                self.assertEqual(diagnostics["stop_reason"], "invalid_pilot_result")
                self.assertEqual(c["observations"], [])
                self.assertEqual(len(env.pilot_history), 1)
                self.assertTrue(all(math.isfinite(v) for v in agent._estimate(c)))
                self.assertEqual(agent.pilot_contacts_left, 4000 - diagnostics["pilot_contacts"])

    def test_no_feasible_pilot_reports_reason_without_calls(self):
        env = PublicScenario(contacts=9)
        diagnostics = explore(PosteriorAgent(), env, [candidate()])
        self.assertEqual(diagnostics["stop_reason"], "pilot_contact_limit")
        self.assertEqual(env.pilot_history, [])


if __name__ == "__main__":
    unittest.main()
