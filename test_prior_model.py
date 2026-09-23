"""Controlled checks of the public-data history prior; no simulator effects."""

import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prior_model import history_priors, PRIOR_ABS_SD, UNSEEN_SD


class HistoryPriorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.csv"
        self.tariffs = pd.DataFrame({"tariff_plan_code": ["a", "b", "c"],
                                     "price_tariff": [1000.0, 2000.0, 3000.0]})

    def write_history(self, rows):
        pd.DataFrame(rows, columns=["tariff_plan_code_from", "tariff_plan_code_to",
                                   "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M"]).to_csv(self.path, index=False)
        return self.path

    def test_sparse_segment_shrinks_toward_tariff_pair(self):
        self.write_history([("a", "b", 500, 1000)] + [("a", "b", 2000, 2000)] * 9)
        priors = history_priors(self.path, self.tariffs)
        mean, sd = priors[("a", "LOW", "b")]
        # Pair mean = 0.1; the sparse segment's unpooled mean is 1.0.
        self.assertAlmostEqual(mean, (1.0 + 5.0 * 0.1) / 6.0)
        self.assertGreater(sd, abs(mean))

    def test_more_support_reduces_sampling_uncertainty_with_shift_floor(self):
        rows = [("a", "b", 2000, 2200), ("a", "b", 2000, 2600)]
        self.write_history(rows * 2)
        small = history_priors(self.path, self.tariffs)[("a", "MID", "b")]
        self.write_history(rows * 20)
        large = history_priors(self.path, self.tariffs)[("a", "MID", "b")]
        self.assertAlmostEqual(small[0], large[0])
        self.assertGreater(small[1], large[1])
        self.assertGreaterEqual(large[1], math.sqrt(large[0] ** 2 + PRIOR_ABS_SD ** 2))

    def test_exact_segment_boundaries(self):
        self.write_history([("a", "b", 999, 999), ("a", "b", 1000, 2000),
                            ("a", "b", 5000, 10000), ("a", "b", 5001, 5001)])
        priors = history_priors(self.path, self.tariffs)
        self.assertAlmostEqual(priors[("a", "LOW", "b")][0], 5 * 0.5 / 6)
        self.assertAlmostEqual(priors[("a", "MID", "b")][0], (2 + 5 * 0.5) / 7)
        self.assertAlmostEqual(priors[("a", "HIGH", "b")][0], 5 * 0.5 / 6)

    def test_missing_history_has_capped_price_fallback(self):
        priors = history_priors(self.path, self.tariffs)
        self.assertEqual(len(priors), 18)
        mean, sd = priors[("a", "LOW", "c")]
        self.assertAlmostEqual(mean, 0.05)
        self.assertGreater(sd, UNSEEN_SD)
        self.assertLess(priors[("c", "HIGH", "a")][0], 0)

    def test_unseen_transition_uses_observed_share_proxy(self):
        self.write_history([("a", "b", 2000, 2400), ("a", "c", 2000, 2400)])
        priors = history_priors(self.path, self.tariffs)
        self.assertAlmostEqual(priors[("b", "MID", "c")][0], 0.5 * 0.5 * 0.5)

    def test_invalid_history_and_prices_remain_finite(self):
        self.write_history([("a", "b", float("inf"), 100),
                            ("a", "b", 1000, float("nan")),
                            ("a", "b", "bad", 100),
                            ("a", "b", 20, 100), (None, "b", 1000, 1200)])
        tariffs = pd.DataFrame({"tariff_plan_code": ["a", "b", "c", "a", None],
                               "price_tariff": [float("inf"), float("nan"), -3, "bad", 1000]})
        priors = history_priors(self.path, tariffs)
        self.assertEqual(len(priors), 18)
        self.assertTrue(all(mean == 0 and sd == UNSEEN_SD for mean, sd in priors.values()))

    def test_single_valid_row_and_duplicate_prices_are_supported(self):
        self.write_history([("a", "b", 2000, 2400)])
        tariffs = pd.concat([self.tariffs, self.tariffs.iloc[[0]]], ignore_index=True)
        priors = history_priors(self.path, tariffs)
        self.assertAlmostEqual(priors[("a", "MID", "b")][0], 0.2)
        self.assertTrue(all(math.isfinite(mean) and math.isfinite(sd) and sd > 0
                            for mean, sd in priors.values()))

    def test_invalid_file_schema_and_empty_catalogue(self):
        self.path.write_text("unrelated\n1\n")
        self.assertEqual(history_priors(self.path, pd.DataFrame()), {})
        priors = history_priors(self.path, self.tariffs.drop(columns="price_tariff"))
        self.assertEqual(len(priors), 18)
        self.assertTrue(all(value == (0.0, UNSEEN_SD) for value in priors.values()))
        self.assertEqual(history_priors(None, None), {})


if __name__ == "__main__":
    unittest.main()
