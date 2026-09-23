"""V2: adaptive Bayesian discovery, followed by conservative SMS deployment.

Copy this file to the supplied case folder as ``agent.py``.  Dependencies are
only numpy/pandas, already used by the case.  No history-derived effect table or
organizer state is read: historical analysis supplies the offer hypotheses only.

The zero-centred Normal(0, 0.25**2) prior has the precision of about 10.3 pilot
contacts.  Each actual observation updates it with variance 0.804**2 / n.
Discovery uses eight 100-contact SMS pilots; subsequent pilots use at most 200
contacts and a one-step, portfolio-level knowledge-gradient approximation.

Pilot IDs are private.  Repeated uniform samples imply target-specific exposure
probability 1-prod(1-n/N).  Planning integrates these probabilities with the
maximum-effect scoring rule, using posterior means for competing offers.  This
is an expected-overlap approximation, not knowledge of exact contacted IDs.
Final ordering/truncation uses the exact public profile sorted by ID_NUMBER.
"""

import itertools
import json
import math
import time

import numpy as np
import pandas as pd


class Agent:
    NOISE_VARIANCE = 0.804 ** 2
    PRIOR_SD = 0.25
    MAX_PILOTS = 20
    MAX_CAMPAIGNS = 10
    MAX_CAMPAIGN_CONTACTS = 5000
    # Stop exploration with twenty seconds reserved inside a five-minute limit.
    LEARNING_SECONDS = 280.0
    HYPOTHESES = (
        ("H1", "tariff_4", "MID", "tariff_8", True),
        ("H2", "tariff_13", "MID", "tariff_8", True),
        ("H3", "tariff_8", "MID", "tariff_10", True),
        ("H4", "tariff_10", "MID", "tariff_11", False),
        ("H5", "tariff_12", "MID", "tariff_8", True),
        ("H6", "tariff_8", "HIGH", "tariff_10", True),
        ("H7", "tariff_11", "HIGH", "tariff_12", True),
        ("H8", "tariff_8", "HIGH", "tariff_21", True),
        ("H9", "tariff_8", "LOW", "tariff_9", True),
        ("H10", "tariff_4", "HIGH", "tariff_21", False),
    )

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.risk_beta = 0.75
        self.trace = []
        self.plan_summary = {}
        self.cells = []

    def _log(self, event, **values):
        record = {"event": event, **values}
        self.trace.append(record)
        if self.verbose:
            print(json.dumps(record, ensure_ascii=False, default=str))

    def _new_cell(self, name, current, segment, target, frame, first_wave=False):
        ordered = frame.sort_values("ID_NUMBER").copy()
        baseline = pd.to_numeric(ordered["predicted_arpu"], errors="coerce")
        values = baseline.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        return {
            "id": name, "current": current, "segment": segment, "target": target,
            "first_wave": first_wave, "frame": ordered, "n": len(ordered),
            "baseline": float(values.sum()), "_prefix": np.r_[0.0, values.cumsum()],
            "mu": 0.0, "var": self.PRIOR_SD ** 2, "samples": 0,
            "pilot_sizes": [], "pilot_channels": [], "inclusion": 0.0, "exposures": {},
        }

    def _build_cells(self, env):
        profile = env.customer_profile
        required = {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"}
        if not required.issubset(profile.columns):
            raise ValueError("Public profile lacks required fields: " + str(sorted(required - set(profile.columns))))
        self._tariffs = set(env.tariffs["tariff_plan_code"].dropna().astype(str))
        self._channels = env.channels
        self.cells = []
        for name, current, segment, target, first_wave in self.HYPOTHESES:
            if current not in self._tariffs or target not in self._tariffs or current == target:
                self._log("candidate_skipped", hypothesis=name, reason="tariff unavailable")
                continue
            frame = profile[(profile["current_tariff"] == current) & (profile["arpu_segment"] == segment)]
            if frame.empty:
                self._log("candidate_skipped", hypothesis=name, reason="empty audience")
                continue
            self.cells.append(self._new_cell(name, current, segment, target, frame, first_wave))
        if not self.cells:
            # A deterministic, genuinely nonempty fallback for a changed catalog.
            # Mixed/missing ARPU is allowed only here and is never pilot-learned.
            for current, frame in profile.groupby("current_tariff", sort=True, observed=True):
                alternatives = sorted(self._tariffs - {str(current)})
                if current not in self._tariffs or not alternatives:
                    continue
                self.cells.append(self._new_cell("catalog_fallback", str(current), None, alternatives[0], frame))
        self._groups = {}
        for cell in self.cells:
            self._groups.setdefault((cell["current"], cell["segment"]), []).append(cell)

    @staticmethod
    def _resources(env):
        return max(0.0, float(env.remaining_budget)), max(0, int(env.remaining_contacts))

    def _campaign(self, cell, channel="sms", suffix="main"):
        result = {
            "campaign_name": "v2_{}_{}".format(suffix, cell["id"]),
            "filter_current_tariff": cell["current"],
            "target_tariff": cell["target"], "channel": channel,
        }
        if cell["segment"] is not None:
            result["filter_arpu_segment"] = cell["segment"]
        return result

    def _marginal_ratio(self, cell, ratios=None, channel="sms"):
        """Additional ratio beyond pilots, with every pilot cost still sunk.

        For untouched people the before effect is zero, including when the new
        offer is negative.  For already touched people it is their best pilot
        effect, which may itself be negative.  No artificial zero-floor is used.
        """
        group = self._groups[(cell["current"], cell["segment"])]
        values = {c["id"]: float(c["mu"]) for c in group}
        if ratios is not None:
            values.update(ratios)
        multiplier = (float(self._channels[channel]["conversion_multiplier"])
                      / float(self._channels["sms"]["conversion_multiplier"]))
        proposed = values[cell["id"]] * multiplier
        exposed = [(c, previous_channel, inclusion)
                   for c in group for previous_channel, inclusion in c["exposures"].items()
                   if inclusion > 0.0]
        marginal = 0.0
        for flags in itertools.product((False, True), repeat=len(exposed)):
            probability = 1.0
            previous = []
            for (c, previous_channel, inclusion), present in zip(exposed, flags):
                p = min(1.0, max(0.0, float(inclusion)))
                probability *= p if present else 1.0 - p
                if present:
                    previous_scale = (float(self._channels[previous_channel]["conversion_multiplier"])
                                      / float(self._channels["sms"]["conversion_multiplier"]))
                    previous.append(values[c["id"]] * previous_scale)
            before = max(previous) if previous else 0.0
            after = max(previous + [proposed]) if previous else proposed
            marginal += probability * (after - before)
        return float(marginal)

    def _lower_ratios(self, cell, beta):
        # Marginal value increases with this offer's effect and decreases with
        # the effect of a competing offer that may already have reached people.
        return {
            c["id"]: float(c["mu"] + (-beta if c is cell else beta) * math.sqrt(c["var"]))
            for c in self._groups[(cell["current"], cell["segment"])]
        }

    def _plan_rows(self, env, estimates=None, budget=None, contacts=None, risk_beta=None):
        """Approximate greedy SMS portfolio; resource/cap simulation is exact.

        Ranking uses conservative net value per counted contact, then total
        conservative value.  One winning offer per source/ARPU group prevents
        final-final overlap.  The greedy portfolio is not claimed to be an
        exact knapsack optimum.
        """
        estimates = self.cells if estimates is None else estimates
        actual_budget, actual_contacts = self._resources(env)
        budget = actual_budget if budget is None else max(0.0, float(budget))
        contacts = actual_contacts if contacts is None else max(0, int(contacts))
        beta = self.risk_beta if risk_beta is None else float(risk_beta)
        cost = float(self._channels["sms"]["cost_per_contact"])
        remaining = list(estimates)
        chosen = []
        while remaining and contacts > 0 and len(chosen) < self.MAX_CAMPAIGNS:
            options = []
            affordable = contacts if cost <= 0 else min(contacts, int(budget // cost))
            for cell in remaining:
                # VOI may simulate a positive future posterior for an untested
                # cell. Actual deployment applies its samples>0 guard in
                # _select_campaigns; this planning helper remains reusable.
                n = min(cell["n"], self.MAX_CAMPAIGN_CONTACTS, affordable)
                if n <= 0:
                    continue
                baseline = float(cell["_prefix"][n])
                expected = baseline * self._marginal_ratio(cell) - n * cost
                conservative = baseline * self._marginal_ratio(cell, self._lower_ratios(cell, beta)) - n * cost
                if not math.isfinite(conservative) or conservative <= 0.0:
                    continue
                options.append({
                    "cell": cell, "n": n, "baseline": baseline,
                    "expected_net": expected, "conservative_net": conservative,
                    "campaign": self._campaign(cell), "cost": n * cost,
                })
            if not options:
                break
            best = max(options, key=lambda r: (r["conservative_net"] / r["n"], r["conservative_net"], r["cell"]["id"]))
            chosen.append(best)
            budget -= best["cost"]
            contacts -= best["n"]
            key = (best["cell"]["current"], best["cell"]["segment"])
            remaining = [c for c in remaining if (c["current"], c["segment"]) != key]
        return chosen

    def _pilot_size(self, env, cell, wanted, channel="sms"):
        budget, contacts = self._resources(env)
        contact_cost = float(self._channels[channel]["cost_per_contact"])
        # At least one final contact remains; discovery must not consume the
        # whole final opportunity on unusually small resource budgets.
        reserve = max(1, min(5000, self._initial_contacts // 2))
        affordable = max(0, contacts - reserve)
        if contact_cost > 0:
            affordable = min(affordable, int(max(0.0, budget - contact_cost) // contact_cost))
        n = int(min(200, wanted, cell["n"], affordable))
        return n if n >= 10 else 0

    def _run_pilot(self, env, cell, n, phase, decision=None, channel="sms"):
        if n < 10 or int(env.pilots_left) <= 0 or self._calls >= self.MAX_PILOTS:
            return False
        specification = self._campaign(cell, channel=channel, suffix="pilot")
        kwargs = {k: v for k, v in specification.items() if k != "campaign_name"}
        kwargs["n_customers"] = int(n)
        try:
            result = env.run_pilot(**kwargs)
        except (RuntimeError, ValueError) as exc:
            self._log("pilot_failed", hypothesis=cell["id"], phase=phase,
                      specification=kwargs, error=str(exc))
            return False
        self._calls += 1
        actual = int(result.get("n_customers", 0))
        observed = float(result.get("observed_lift_ratio", float("nan")))
        if actual <= 0:
            self._log("pilot_invalid", hypothesis=cell["id"], specification=kwargs,
                      reason="nonpositive actual sample", result=result)
            return False
        # Contact ledger updates even if a malformed observation cannot be used.
        cell["pilot_sizes"].append(actual)
        cell["pilot_channels"].append(channel)
        cell["inclusion"] = 1.0 - (1.0 - cell["inclusion"]) * (1.0 - min(1.0, actual / cell["n"]))
        prior_exposure = cell["exposures"].get(channel, 0.0)
        cell["exposures"][channel] = 1.0 - (1.0 - prior_exposure) * (1.0 - min(1.0, actual / cell["n"]))
        if not math.isfinite(observed):
            self._log("pilot_invalid", hypothesis=cell["id"], specification=kwargs,
                      reason="nonfinite observation", result=result)
            return False
        old_var = cell["var"]
        old_mu = cell["mu"]
        # The emergency free-push probe has noise on push-relative lift.  Its
        # mean AND standard deviation must scale before the SMS posterior update.
        normalization = (float(self._channels["sms"]["conversion_multiplier"])
                         / float(self._channels[channel]["conversion_multiplier"]))
        normalized_observed = observed * normalization
        normalized_noise_variance = self.NOISE_VARIANCE * normalization ** 2
        variance = 1.0 / (1.0 / old_var + actual / normalized_noise_variance)
        cell["mu"] = variance * (old_mu / old_var + normalized_observed * actual / normalized_noise_variance)
        cell["var"] = variance
        cell["samples"] += actual
        self._log(
            "pilot", hypothesis=cell["id"], phase=phase, specification=kwargs,
            requested_n=n, actual_n=actual, observed_lift_ratio=observed,
            sms_normalized_observed_lift=normalized_observed,
            observation_variance_sms_units=normalized_noise_variance / actual,
            cost=float(result.get("cost", 0.0)), posterior_mean=cell["mu"],
            posterior_sd=math.sqrt(cell["var"]), total_observed_n=cell["samples"],
            expected_unique_pilot_contacts=cell["n"] * cell["inclusion"],
            expected_repeat_contacts=sum(cell["pilot_sizes"]) - cell["n"] * cell["inclusion"],
            decision=decision, remaining_budget=float(env.remaining_budget),
            remaining_contacts=int(env.remaining_contacts),
        )
        return True

    def _information_value(self, env, cell, n, current_value):
        """One-step expected portfolio improvement using nine-point quadrature.

        Future posterior mean has variance v-v_new.  Final resources and pilot
        exposure probabilities are recomputed for every possible observation.
        The pilot contributes its own expected incremental revenue, but costs
        both money and final contact capacity.  Competing offer effects use
        posterior plug-in means; this is a tractable knowledge-gradient policy.
        """
        budget, contacts = self._resources(env)
        cost = n * float(self._channels["sms"]["cost_per_contact"])
        immediate_gross = n * (cell["baseline"] / cell["n"]) * self._marginal_ratio(cell)
        old_mu, old_var, old_p = cell["mu"], cell["var"], cell["inclusion"]
        old_exposures = dict(cell["exposures"])
        new_var = 1.0 / (1.0 / old_var + n / self.NOISE_VARIANCE)
        mean_sd = math.sqrt(max(0.0, old_var - new_var))
        expected_future = 0.0
        try:
            cell["var"] = new_var
            cell["inclusion"] = 1.0 - (1.0 - old_p) * (1.0 - n / cell["n"])
            cell["exposures"]["sms"] = 1.0 - (1.0 - old_exposures.get("sms", 0.0)) * (1.0 - n / cell["n"])
            for node, weight in zip(self._quadrature_nodes, self._quadrature_weights):
                cell["mu"] = old_mu + math.sqrt(2.0) * mean_sd * float(node)
                rows = self._plan_rows(env, budget=budget - cost, contacts=contacts - n, risk_beta=0.0)
                expected_future += float(weight) * sum(row["expected_net"] for row in rows)
        finally:
            cell["mu"], cell["var"], cell["inclusion"] = old_mu, old_var, old_p
            cell["exposures"] = old_exposures
        value = expected_future + immediate_gross - cost - current_value
        return {
            "net_decision_value": float(value),
            "expected_future_final_net": float(expected_future),
            "current_final_net": float(current_value),
            "pilot_expected_marginal_gross": float(immediate_gross),
            "pilot_cost": float(cost), "actual_planned_n": n,
            "posterior_sd_after": math.sqrt(new_var),
            "expected_already_contacted_by_this_offer": n * old_p,
        }

    def _fallback_campaign(self, env):
        """Minimum-one rule: choose a real supported subset with least downside.

        Push has no communication cost and its smaller response multiplier
        reduces the downside of negative or uncertain offers.  Subdivision is
        restricted to the documented data/call filter values.  It is not an
        invented customer-count field or an intentionally empty campaign.
        """
        budget, contacts = self._resources(env)
        if contacts < 1 or "push" not in self._channels:
            raise ValueError("No feasible nonempty final campaign: no contacts or push channel")
        options = []
        for cell in self.cells:
            frame = cell["frame"]
            data_values = [None] + ([x for x in ("NON_USER", "LITE", "HEAVY") if (frame["data_segment"] == x).any()] if "data_segment" in frame else [])
            call_values = [None] + ([x for x in ("LOW", "MEDIUM", "HIGH") if (frame["call_segment"] == x).any()] if "call_segment" in frame else [])
            for data, call in itertools.product(data_values, call_values):
                subset = frame
                if data is not None:
                    subset = subset[subset["data_segment"] == data]
                if call is not None:
                    subset = subset[subset["call_segment"] == call]
                counted = subset.iloc[:min(contacts, self.MAX_CAMPAIGN_CONTACTS)]
                if counted.empty:
                    continue
                baseline = float(pd.to_numeric(counted["predicted_arpu"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).sum())
                expected = baseline * self._marginal_ratio(cell, channel="push")
                # For a negative offer, weaker push can improve on an earlier
                # SMS contact.  Its marginal value need not increase with that
                # offer's effect; examine all piecewise-linear breakpoints.
                bounds = self._lower_ratios(cell, self.risk_beta)
                low = cell["mu"] - self.risk_beta * math.sqrt(cell["var"])
                high = cell["mu"] + self.risk_beta * math.sqrt(cell["var"])
                candidates = [low, high]
                if low <= 0.0 <= high:
                    candidates.append(0.0)
                push_scale = (float(self._channels["push"]["conversion_multiplier"])
                              / float(self._channels["sms"]["conversion_multiplier"]))
                for other in self._groups[(cell["current"], cell["segment"])]:
                    if other is cell:
                        continue
                    for previous_channel in other["exposures"]:
                        previous_scale = (float(self._channels[previous_channel]["conversion_multiplier"])
                                          / float(self._channels["sms"]["conversion_multiplier"]))
                        breakpoint = bounds[other["id"]] * previous_scale / push_scale
                        if low <= breakpoint <= high:
                            candidates.append(breakpoint)
                lower = baseline * min(self._marginal_ratio(cell, {**bounds, cell["id"]: value}, channel="push") for value in candidates)
                campaign = self._campaign(cell, channel="push", suffix="minimum_one")
                if data is not None:
                    campaign["filter_data_segment"] = data
                if call is not None:
                    campaign["filter_call_segment"] = call
                options.append((lower, expected, -len(counted), campaign, len(counted), cell))
        if not options:
            raise ValueError("No feasible nonempty final campaign in the public catalog/profile")
        lower, expected, _, campaign, n, cell = max(options, key=lambda item: item[:3])
        self._log("minimum_one_fallback", reason="No positive conservative SMS opportunity; satisfying minimum-one campaign rule",
                  hypothesis=cell["id"], campaign=campaign, counted_contacts=n,
                  expected_marginal_net=expected, conservative_marginal_net=lower)
        self.plan_summary = {
            "policy": "least-downside supported push subset", "n_campaigns": 1,
            "final_contacts": n, "final_cost": 0.0,
            "expected_final_marginal_net": expected, "conservative_final_marginal_net": lower,
            "remaining_contacts_after_final": contacts - n, "remaining_budget_after_final": budget,
        }
        return [campaign]

    def _select_campaigns(self, env, estimates=None):
        candidates = self.cells if estimates is None else estimates
        observed = [cell for cell in candidates if cell["samples"] > 0]
        rows = self._plan_rows(env, estimates=observed)
        if not rows:
            return self._fallback_campaign(env)
        selected_ids = {row["cell"]["id"] for row in rows}
        for cell in self.cells:
            if cell["id"] not in selected_ids:
                self._log("candidate_not_deployed", hypothesis=cell["id"],
                          posterior_mean=cell["mu"], posterior_sd=math.sqrt(cell["var"]),
                          reason="Nonpositive conservative value, overlapping winner, or resource priority")
        for order, row in enumerate(rows, 1):
            cell = row["cell"]
            self._log("campaign", order=order, campaign=row["campaign"],
                      posterior_mean=cell["mu"], posterior_sd=math.sqrt(cell["var"]),
                      counted_contacts=row["n"], counted_baseline=row["baseline"],
                      expected_marginal_net=row["expected_net"], conservative_marginal_net=row["conservative_net"],
                      expected_overlap_with_same_offer_pilots=row["n"] * cell["inclusion"],
                      cost=row["cost"], reason="Positive conservative marginal value after pilot overlap")
        budget, contacts = self._resources(env)
        self.plan_summary = {
            "policy": "conservative SMS greedy portfolio", "risk_beta": self.risk_beta,
            "n_campaigns": len(rows), "final_contacts": sum(r["n"] for r in rows),
            "final_cost": sum(r["cost"] for r in rows),
            "expected_final_marginal_net": sum(r["expected_net"] for r in rows),
            "conservative_final_marginal_net": sum(r["conservative_net"] for r in rows),
            "remaining_contacts_after_final": contacts - sum(r["n"] for r in rows),
            "remaining_budget_after_final": budget - sum(r["cost"] for r in rows),
        }
        return [row["campaign"] for row in rows]

    def act(self, env):
        started = time.monotonic()
        self._deadline = started + self.LEARNING_SECONDS
        self.trace = []
        self.plan_summary = {}
        self._calls = 0
        self._initial_contacts = max(0, int(env.remaining_contacts))
        self._build_cells(env)
        if "sms" not in self._channels or "push" not in self._channels:
            raise ValueError("The supplied case requires SMS and push channels")
        self._quadrature_nodes, self._quadrature_weights = np.polynomial.hermite.hermgauss(9)
        self._quadrature_weights = self._quadrature_weights / math.sqrt(math.pi)
        self._log("start", candidate_count=len(self.cells), prior_mean=0.0,
                  prior_sd=self.PRIOR_SD, observation_noise_sd=math.sqrt(self.NOISE_VARIANCE),
                  final_risk_beta=self.risk_beta,
                  overlap_method="Independent uniform pilot inclusion; posterior-mean max-effect approximation")
        for cell in self.cells:
            if not cell["first_wave"] or time.monotonic() >= self._deadline:
                continue
            n = self._pilot_size(env, cell, 100)
            if n and int(env.pilots_left) > 0:
                self._run_pilot(env, cell, n, phase="discovery")
        if self._calls == 0 and int(env.pilots_left) > 0 and time.monotonic() < self._deadline:
            # On an exhausted-money fixture, still gather real evidence through
            # a free channel while preserving at least half the contact budget.
            possible = [c for c in self.cells if c["segment"] is not None]
            for cell in sorted(possible, key=lambda c: -c["baseline"]):
                n = self._pilot_size(env, cell, 100, channel="push")
                if n and self._run_pilot(env, cell, n, phase="emergency_discovery", channel="push"):
                    break
        adaptive_calls = 0
        while self._calls < self.MAX_PILOTS and int(env.pilots_left) > 0 and time.monotonic() < self._deadline:
            current_rows = self._plan_rows(env, risk_beta=0.0)
            current_value = sum(r["expected_net"] for r in current_rows)
            choices = []
            for cell in self.cells:
                if cell["segment"] is None or time.monotonic() >= self._deadline:
                    continue
                n = self._pilot_size(env, cell, 200)
                if not n:
                    continue
                decision = self._information_value(env, cell, n, current_value)
                if math.isfinite(decision["net_decision_value"]):
                    choices.append((decision["net_decision_value"], cell, n, decision))
            if not choices:
                self._log("learning_stopped", reason="No feasible informative pilot or time reserve reached")
                break
            value, cell, n, decision = max(choices, key=lambda x: (x[0], x[1]["id"]))
            self._log("adaptive_decision", chosen=cell["id"], candidates=[
                {"hypothesis": c["id"], **d} for _, c, _, d in sorted(choices, key=lambda x: -x[0])])
            if value <= 0.0:
                self._log("learning_stopped", reason="Expected decision improvement does not pay pilot cost/contact opportunity", best_net_decision_value=value)
                break
            phase = "confirmation" if adaptive_calls < 6 else "reserve"
            if not self._run_pilot(env, cell, n, phase=phase, decision=decision):
                break
            adaptive_calls += 1
        campaigns = self._select_campaigns(env, self.cells)
        if not 1 <= len(campaigns) <= self.MAX_CAMPAIGNS:
            raise ValueError("Final campaign count violates the case contract")
        self.plan_summary.update({
            "pilot_calls": self._calls, "pilot_contacts": sum(sum(c["pilot_sizes"]) for c in self.cells),
            "elapsed_seconds": time.monotonic() - started,
            "estimate_warning": "Posterior predictions and expected pilot overlap, not measured final score",
        })
        self._log("complete", **self.plan_summary)
        return campaigns
