"""Checks for aggregate evidence and a deterministic offline report."""
import hashlib
import json
import unittest

from agent import Agent
from analyst_evidence import ROOT, build_evidence, render_report
from mock_environment import make_mock_env


class AnalystEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_digest = hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest()
        cls.evidence = build_evidence(seed=42)

    def test_json_safe_and_deterministic(self):
        encoded = json.dumps(self.evidence, sort_keys=True, allow_nan=False)
        repeated = json.dumps(build_evidence(seed=42), sort_keys=True, allow_nan=False)
        self.assertEqual(encoded, repeated)
        self.assertEqual(render_report(self.evidence), render_report(self.evidence))

    def test_no_subscriber_identifiers_or_records(self):
        forbidden = {"ID_NUMBER", "explicit_ids", "customer_profile", "customer_records",
                     "true_lift", "true_lift_ratio", "actual_net_gain", "net_arpu_gain"}

        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(self.evidence)
        self.assertEqual(sum(cell["population"] for cell in self.evidence["audience_summary"]["cells"]),
                         self.evidence["audience_summary"]["population"])

    def test_catalog_ids_and_valid_campaign_enums(self):
        catalog = self.evidence["candidate_catalog"]
        self.assertTrue(0 < len(catalog) <= 40)
        self.assertEqual(len(catalog), len({candidate["id"] for candidate in catalog}))
        tariffs = {row["tariff_plan_code"] for row in self.evidence["tariffs"]}
        channels = set(self.evidence["channels"])
        ranks = []
        for candidate in catalog:
            self.assertIn(candidate["target_tariff"], tariffs)
            self.assertIn(candidate["filter_current_tariff"], tariffs)
            self.assertNotEqual(candidate["target_tariff"], candidate["filter_current_tariff"])
            self.assertIn(candidate["filter_arpu_segment"], {"LOW", "MID", "HIGH"})
            self.assertIn(candidate["channel"], channels)
            self.assertGreaterEqual(candidate["history"]["migration_support"], 0)
            self.assertGreaterEqual(candidate["population"], 10)
            ranks.append(candidate["historical_ranking_net_proxy"])
        self.assertEqual(ranks, sorted(ranks, reverse=True))

    def test_resource_arithmetic_and_caps(self):
        evidence = self.evidence
        resources, initial = evidence["resources"], evidence["initial_limits"]
        self.assertEqual(resources["pilot_cost"], sum(p["result"]["cost"] for p in evidence["pilots"]))
        self.assertEqual(resources["pilot_contacts"], sum(p["result"]["n_customers"] for p in evidence["pilots"]))
        self.assertAlmostEqual(resources["budget_after_pilots"] + resources["pilot_cost"], initial["budget"])
        self.assertAlmostEqual(resources["estimated_deployment_cost"] + resources["estimated_budget_after_deployment"],
                               resources["budget_after_pilots"])
        self.assertTrue(1 <= len(evidence["final_campaigns"]) <= 10)
        self.assertTrue(1 <= len(evidence["pilots"]) <= 20)
        self.assertGreaterEqual(resources["estimated_budget_after_deployment"], 0)
        self.assertGreaterEqual(resources["estimated_contacts_after_deployment"], 0)
        for pilot in evidence["pilots"]:
            self.assertTrue(10 <= pilot["request"]["n_customers"] <= 200)
        for deployment in resources["campaigns"]:
            self.assertTrue(0 <= deployment["estimated_contacts"] <= 5000)

    def test_capture_does_not_change_agent_output_or_source(self):
        env = make_mock_env(seed=42, data_dir=str(ROOT / "data"),
                            profile_path=str(ROOT / "customer_profile.csv"))[0]
        self.assertEqual(Agent().act(env), self.evidence["final_campaigns"])
        self.assertEqual(hashlib.sha256((ROOT / "agent.py").read_bytes()).hexdigest(), self.agent_digest)

    def test_report_labels_estimates_and_optional_commentary(self):
        report = render_report(self.evidence)
        self.assertIn("Budget after pilots (observed)", report)
        self.assertIn("Deployment cost (estimated)", report)
        self.assertIn("Actual scored net gain | Not calculated", report)
        self.assertIn("excludes pilots", report)
        self.assertNotIn("## Optional LLM commentary", report)
        interpreted = render_report(self.evidence, "Consider another pilot for candidate_001.")
        self.assertIn("## Optional LLM commentary", interpreted)
        self.assertIn("proposals only", interpreted)
        structured = render_report(self.evidence, {
            "summary": "Pilot feedback informs the plan.",
            "observations": ["Estimates remain uncertain."],
            "limitations": ["No judging score was observed."],
            "hypotheses": [{"candidate_id": "candidate_001", "rationale": "Check the prior.",
                            "evidence_to_collect": "An independent pilot."}],
        })
        self.assertIn("**candidate_001**: Check the prior.", structured)
        self.assertIn("Evidence to collect: An independent pilot.", structured)


if __name__ == "__main__":
    unittest.main()
