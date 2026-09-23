"""Small exhaustive references and regression cases for whole-plan search."""
import unittest
import numpy as np

from planner import _gain_curve, select_plan


def candidate(cell, values, cost=0, lower=1, pilots=0):
    return dict(cell=cell, values=np.asarray(values, dtype=float), cost=cost,
                lower=lower, pilot_contacts=pilots)


def exhaustive(candidates, budget, contacts, limit=10):
    """Independent exhaustive reference for tiny instances, same objective."""
    def visit(money, remaining, used, depth):
        best = 0.0
        if depth == limit:
            return best
        for c in candidates:
            if c["cell"] in used or c["lower"] <= 0:
                continue
            n = min(remaining, len(c["values"]), 5000)
            if c["cost"]:
                n = min(n, int(money // c["cost"]))
            if n <= 0:
                continue
            prefix = c["values"][:n]
            k = min(n, c["pilot_contacts"])
            fresh = prefix.sum() - (np.sort(prefix)[-k:].sum() if k else 0)
            gain = fresh * c["lower"] - n * c["cost"]
            if gain > 0:
                best = max(best, gain + visit(money - n * c["cost"], remaining - n,
                                             used | {c["cell"]}, depth + 1))
        return best
    return visit(budget, contacts, set(), 0)


class PlannerTests(unittest.TestCase):
    def test_combination_beats_best_individual_campaign(self):
        cs = [candidate("A", [29], 10), candidate("B", [22], 5), candidate("C", [22], 5)]
        chosen, d = select_plan(cs, budget=10, contacts=10)
        self.assertEqual(set(chosen), {1, 2})
        self.assertAlmostEqual(d["greedy_estimated_net"], 19)
        self.assertAlmostEqual(d["estimated_net"], 34)

    def test_seven_campaigns_without_padding(self):
        cs = [candidate(i, [2], cost=1) for i in range(7)]
        cs += [candidate(i, [0.5], cost=1) for i in range(7, 10)]
        chosen, d = select_plan(cs, budget=100, contacts=100)
        self.assertEqual(len(chosen), 7)
        self.assertEqual(d["estimated_net"], 7)
        self.assertEqual(set(d["best_estimated_net_by_count"]), set(range(8)))

    def test_ten_is_an_upper_limit(self):
        cs = [candidate(i, [1]) for i in range(12)]
        chosen, _ = select_plan(cs, budget=0, contacts=100)
        self.assertEqual(len(chosen), 10)

    def test_rejects_overlapping_cells(self):
        cs = [candidate("same", [10]), candidate("same", [20]), candidate("other", [1])]
        chosen, _ = select_plan(cs, budget=0, contacts=100)
        self.assertEqual(set(chosen), {1, 2})

    def test_order_handles_paid_truncation_and_free_followup(self):
        cs = [candidate("expensive", [12] * 10, 4),
              candidate("cheap", [6.5] * 10, 1), candidate("free", [5] * 10)]
        chosen, d = select_plan(cs, budget=10, contacts=7)
        self.assertAlmostEqual(d["estimated_net"], exhaustive(cs, 10, 7))
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(d["planned_cost"], 10)
        self.assertEqual(d["planned_contacts"], 7)

    def test_overlap_curve_matches_direct_prefix_calculation(self):
        values = np.array([10, 2, 30, 5, 1, 40], dtype=float)
        curve = _gain_curve(values, 0.4, 2, 1)
        for n in range(1, len(values) + 1):
            fresh = values[:n].sum() - np.sort(values[:n])[-min(2, n):].sum()
            self.assertAlmostEqual(curve[n], fresh * 0.4 - n)

    def test_exhaustive_agreement_on_small_instances(self):
        rng = np.random.default_rng(90210)
        for _ in range(30):
            cs = [candidate(i // 2, rng.uniform(1, 30, rng.integers(1, 10)),
                            int(rng.integers(0, 5)), float(rng.uniform(0.1, 1)),
                            int(rng.integers(0, 3))) for i in range(6)]
            budget, contacts = int(rng.integers(5, 30)), int(rng.integers(5, 30))
            _, d = select_plan(cs, budget, contacts)
            self.assertAlmostEqual(d["estimated_net"], exhaustive(cs, budget, contacts))

    def test_greedy_incumbent_survives_narrow_beam(self):
        cs = [candidate("A", [29], 10), candidate("B", [22], 5), candidate("C", [22], 5)]
        _, d = select_plan(cs, 10, 10, beam_width=1)
        self.assertGreaterEqual(d["estimated_net"], d["greedy_estimated_net"])

    def test_nonpositive_candidates_produce_no_plan(self):
        cs = [candidate("A", [10], lower=-1), candidate("B", [1], cost=2)]
        chosen, _ = select_plan(cs, 100, 100)
        self.assertEqual(chosen, [])


if __name__ == "__main__":
    unittest.main()
