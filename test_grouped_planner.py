"""Targeted allocation cases independent of organizer implementation details."""
import unittest
import numpy as np
from grouped_planner import _prepare_options, select_grouped_plan
from planner import select_plan


def candidate(current, ids, values, *, ratio=1.0, cost=0.0, target="destination",
              segment="HIGH", population=None, pilot_contacts=0):
    return {"cell": (current, segment), "ids": np.asarray(ids),
            "values": np.asarray(values, dtype=float), "gain_ratio": ratio,
            "cost": cost, "population": len(values) if population is None else population,
            "pilot_contacts": pilot_contacts,
            "campaign": {"filter_current_tariff": current, "filter_arpu_segment": segment,
                         "target_tariff": target, "channel": "sms"}}


class GroupedPlannerTests(unittest.TestCase):
    def test_full_population_cap_even_when_budget_would_truncate(self):
        options = [candidate("A", np.arange(3000), np.ones(3000), cost=0.1),
                   candidate("B", np.arange(3000, 6000), np.ones(3000), cost=0.1)]
        plan, diagnostics = select_grouped_plan(options, budget=1, contacts=100, max_campaigns=1)
        self.assertEqual(diagnostics["merge_alternatives_generated"], 0)
        self.assertEqual(len(plan), 1)
        self.assertNotIn(";", plan[0]["filter_current_tariff"])

    def test_exactly_five_thousand_is_allowed(self):
        options = [candidate("A", np.arange(2500), np.ones(2500)),
                   candidate("B", np.arange(2500, 5000), np.ones(2500))]
        plan, diagnostics = select_grouped_plan(options, 0, 10000, max_campaigns=1)
        self.assertEqual(diagnostics["estimated_net"], 5000)
        self.assertEqual(diagnostics["planned_contacts"], 5000)
        self.assertEqual(plan[0]["filter_current_tariff"], "A;B")

    def test_merged_prefix_interleaves_ids_and_preserves_distinct_cell_responses(self):
        options = [candidate("A", [2, 4], [20, 40], ratio=0.5, cost=1),
                   candidate("B", [1, 3], [50, 0.5], ratio=2.0, cost=1)]
        plan, diagnostics = select_grouped_plan(options, budget=2, contacts=4, max_campaigns=1)
        self.assertEqual(plan[0]["filter_current_tariff"], "A;B")
        self.assertAlmostEqual(diagnostics["estimated_net"], 108)
        self.assertEqual(diagnostics["planned_contacts"], 2)
        self.assertEqual(diagnostics["planned_cost"], 2)

    def test_overlapping_groups_cannot_charge_the_same_cell_twice(self):
        options = [candidate(current, np.arange(i * 2000, (i + 1) * 2000),
                             np.full(2000, 3 - i)) for i, current in enumerate("ABC")]
        _, diagnostics = select_grouped_plan(options, 0, 10000, max_campaigns=2)
        flattened = [index for group in diagnostics["selected_source_indices"] for index in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {0, 1, 2})
        self.assertEqual(diagnostics["estimated_net"], 12000)
        self.assertEqual(diagnostics["planned_contacts"], 6000)

    def test_grouping_can_cover_eleven_profitable_cells_under_ten_campaign_limit(self):
        options = [candidate(str(i), [i], [1]) for i in range(11)]
        plan, diagnostics = select_grouped_plan(options, 0, 20)
        self.assertLessEqual(len(plan), 10)
        self.assertEqual(diagnostics["estimated_net"], 11)
        self.assertEqual(diagnostics["ungrouped_estimated_net"], 10)
        self.assertGreater(diagnostics["selected_merged_campaigns"], 0)

    def test_ten_campaign_cap_applies_to_distinct_nonmergeable_targets(self):
        options = [candidate(str(i), [i], [1], target=str(i)) for i in range(12)]
        plan, diagnostics = select_grouped_plan(options, 0, 20, max_campaigns=20)
        self.assertEqual(len(plan), 10)
        self.assertEqual(diagnostics["estimated_net"], 10)

    def test_different_arpu_target_channel_or_other_filter_never_merge(self):
        options = [candidate("A", [1], [3]), candidate("B", [2], [3], segment="MID"),
                   candidate("C", [3], [3], target="elsewhere"),
                   candidate("D", [4], [3]), candidate("E", [5], [3])]
        options[3]["campaign"]["channel"] = "push"
        options[4]["campaign"]["filter_data_segment"] = "HEAVY"
        _, diagnostics = _prepare_options(options)
        self.assertEqual(diagnostics["merge_alternatives_generated"], 0)

    def test_merge_requires_complete_population_ids_and_corrected_overlap(self):
        options = [candidate("A", [1], [1]), candidate("B", [2], [1], population=4000),
                   candidate("C", [3], [1], pilot_contacts=1), candidate("D", [4], [1])]
        del options[3]["ids"]
        prepared, diagnostics = _prepare_options(options)
        self.assertEqual(len(prepared), 4)
        self.assertEqual(diagnostics["merge_alternatives_generated"], 0)
        self.assertEqual(diagnostics["merge_options_without_complete_identity_data"], 3)

    def test_incumbent_retains_singleton_objective_with_narrow_beam(self):
        rng = np.random.default_rng(111)
        for _ in range(15):
            options = [candidate(str(i), [2 * i, 2 * i + 1], rng.uniform(1, 15, 2),
                                 cost=float(rng.integers(0, 5)), target=str(i % 2))
                       for i in range(7)]
            budget, contacts = int(rng.integers(2, 20)), int(rng.integers(1, 15))
            _, singleton = select_plan(options, budget, contacts, beam_width=1)
            _, grouped = select_grouped_plan(options, budget, contacts, beam_width=1)
            self.assertGreaterEqual(grouped["estimated_net"] + 1e-9, singleton["estimated_net"])

    def test_empty_and_zero_limit_have_no_plan(self):
        for options, limit in [([], 10), ([candidate("A", [1], [1])], 0)]:
            plan, diagnostics = select_grouped_plan(options, 0, 100, max_campaigns=limit)
            self.assertEqual(plan, [])
            self.assertEqual(diagnostics["estimated_net"], 0)


if __name__ == "__main__":
    unittest.main()
