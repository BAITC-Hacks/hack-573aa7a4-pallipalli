"""Contract and failure-path checks with a fake client; no paid API calls."""
import copy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openai_analyst import analyze, evidence_hash, freeze_hypotheses, validate_commentary, grounded_context, SUMMARIES, PILOT_QUESTIONS


EVIDENCE = {"seed": 42, "candidate_catalog": [{"id": "candidate_001", "campaign": {
    "filter_current_tariff": "tariff_4", "filter_arpu_segment": "MID",
    "target_tariff": "tariff_8", "channel": "sms"}}]}
CONTEXT = grounded_context(EVIDENCE)
COMMENTARY = {"summary": SUMMARIES[0],
              "observations": [CONTEXT["candidate_catalog"][0]["allowed_rationales"][0]],
              "limitations": CONTEXT["limitations"],
              "hypotheses": [{"candidate_id": "candidate_001",
                              "rationale": CONTEXT["candidate_catalog"][0]["allowed_rationales"][0],
                              "evidence_to_collect": PILOT_QUESTIONS[0]}]}



class FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = self
        self.response = response or SimpleNamespace(status="completed", output_text=json.dumps(COMMENTARY),
                                                   usage=SimpleNamespace(input_tokens=50, output_tokens=20, total_tokens=70))
        self.error = error
        self.calls = []
        self.closed = False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


class AnalystTests(unittest.TestCase):
    def call_with(self, client):
        factory_kwargs = []
        def factory(**kwargs):
            factory_kwargs.append(kwargs)
            return client
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret-value"}, clear=True):
            result = analyze(copy.deepcopy(EVIDENCE), enabled=True, client_factory=factory)
        return result, factory_kwargs

    def test_disabled_never_creates_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True):
            result = analyze(EVIDENCE, client_factory=lambda **kw: self.fail("Network was enabled"))
        self.assertEqual(result["reason"], "disabled")

    def test_missing_key_never_creates_client(self):
        with patch.dict(os.environ, {}, clear=True):
            result = analyze(EVIDENCE, enabled=True, client_factory=lambda **kw: self.fail("No key"))
        self.assertEqual(result["reason"], "missing_api_key")

    def test_valid_response_uses_strict_schema_and_bounded_request(self):
        client = FakeClient()
        result, config = self.call_with(client)
        self.assertEqual(result["status"], "openai")
        self.assertEqual(result["usage"]["total_tokens"], 70)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(config[0]["max_retries"], 0)
        self.assertEqual(config[0]["timeout"], 20)
        self.assertEqual(config[0]["base_url"], "https://api.openai.com/v1")
        self.assertFalse(client.calls[0]["store"])
        self.assertTrue(client.calls[0]["text"]["format"]["strict"])
        self.assertEqual(client.calls[0]["max_output_tokens"], 1800)
        self.assertNotIn("test-secret-value", json.dumps(result))
        self.assertNotIn("test-secret-value", json.dumps(client.calls))
        self.assertTrue(client.closed)

    def test_timeout_falls_back_without_leaking_error(self):
        client = FakeClient(error=TimeoutError("test-secret-value"))
        result, _ = self.call_with(client)
        self.assertEqual(result["status"], "offline")
        self.assertEqual(result["reason"], "api_error")
        self.assertNotIn("test-secret-value", json.dumps(result))
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.closed)

    def test_incomplete_and_refused_responses_fall_back(self):
        for status, text in [("incomplete", "{}"), ("completed", "")]:
            response = SimpleNamespace(status=status, output_text=text)
            result, _ = self.call_with(FakeClient(response=response))
            self.assertEqual(result["status"], "offline")
            self.assertIsNone(result["commentary"])

    def test_invalid_json_and_unknown_ids_are_rejected(self):
        bad = copy.deepcopy(COMMENTARY)
        bad["hypotheses"][0]["candidate_id"] = "made-up"
        for text in ("broken json", json.dumps(bad)):
            result, _ = self.call_with(FakeClient(response=SimpleNamespace(status="completed", output_text=text)))
            self.assertEqual(result["reason"], "invalid_response")

    def test_duplicate_candidates_are_rejected(self):
        bad = copy.deepcopy(COMMENTARY)
        bad["hypotheses"] *= 2
        with self.assertRaises(ValueError):
            validate_commentary(bad, {"candidate_001"})

    def test_extra_actions_are_rejected(self):
        bad = dict(COMMENTARY, execute_python="dangerous")
        with self.assertRaises(ValueError):
            validate_commentary(bad, {"candidate_001"})

    def test_freeze_is_deterministic_and_inactive(self):
        analysis, _ = self.call_with(FakeClient())
        bundle = {"evidence": EVIDENCE, "analysis": analysis}
        first = freeze_hypotheses(bundle)
        self.assertEqual(first, freeze_hypotheses(bundle))
        self.assertFalse(first["active_in_agent"])
        self.assertEqual(first["status"], "proposal_requires_evaluation")
        self.assertEqual(first["hypotheses"][0]["campaign"]["target_tariff"], "tariff_8")

    def test_freeze_rejects_mismatched_evidence(self):
        analysis, _ = self.call_with(FakeClient())
        altered = dict(EVIDENCE, seed=43)
        with self.assertRaises(ValueError):
            freeze_hypotheses({"evidence": altered, "analysis": analysis})

    def test_freeze_rejects_offline_or_empty_suggestions(self):
        with self.assertRaises(ValueError):
            freeze_hypotheses({"evidence": EVIDENCE, "analysis": analyze(EVIDENCE)})
        analysis, _ = self.call_with(FakeClient())
        analysis["commentary"]["hypotheses"] = []
        with self.assertRaises(ValueError):
            freeze_hypotheses({"evidence": EVIDENCE, "analysis": analysis})

    def test_unsupported_metrics_are_rejected(self):
        bad = copy.deepcopy(COMMENTARY)
        bad["hypotheses"][0]["evidence_to_collect"] = "Measure churn and conversions."
        with self.assertRaises(ValueError):
            validate_commentary(bad, {"candidate_001"}, CONTEXT)

    def test_unsupported_evidence_claim_is_rejected(self):
        bad = copy.deepcopy(COMMENTARY)
        bad["observations"] = ["This candidate has positive pilot results."]
        with self.assertRaises(ValueError):
            validate_commentary(bad, {"candidate_001"}, CONTEXT)

    def test_rationale_cannot_be_borrowed_from_another_candidate(self):
        evidence = copy.deepcopy(EVIDENCE)
        second = copy.deepcopy(evidence["candidate_catalog"][0])
        second["id"] = "candidate_002"
        evidence["candidate_catalog"].append(second)
        context = grounded_context(evidence)
        bad = copy.deepcopy(COMMENTARY)
        bad["hypotheses"][0]["rationale"] = context["candidate_catalog"][1]["allowed_rationales"][0]
        with self.assertRaises(ValueError):
            validate_commentary(bad, {"candidate_001", "candidate_002"}, context)

    def test_hash_is_independent_of_dict_order(self):
        self.assertEqual(evidence_hash({"a": 1, "b": 2}), evidence_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
