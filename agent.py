"""V1: fixed marketing hypotheses, one SMS pilot each, and simple SMS rollout.

Copy this file to ``agent.py`` beside the participant's ``local_eval.py``.
Only the documented public environment is used. Historical switcher results
identify the eight hypotheses; they are not treated as campaign response rates.

This deliberately simple baseline ranks point estimates rather than Bayesian
posteriors. Pilot noise and repeated pilot/final contacts can therefore make its
predicted value optimistic. ``trace`` records evidence and ``plan_summary``
describes the returned plan; neither contains hidden customer or effect data.
"""

import json
import math
import time

import pandas as pd


class Agent:
    """Transparent rules baseline with a quiet, submission-compatible API."""

    HYPOTHESES = (
        ("H1", "tariff_4", "MID", "tariff_8"),
        ("H2", "tariff_13", "MID", "tariff_8"),
        ("H3", "tariff_8", "MID", "tariff_10"),
        ("H5", "tariff_12", "MID", "tariff_8"),
        ("H6", "tariff_8", "HIGH", "tariff_10"),
        ("H7", "tariff_11", "HIGH", "tariff_12"),
        ("H8", "tariff_8", "HIGH", "tariff_21"),
        ("H9", "tariff_8", "LOW", "tariff_9"),
    )
    DATA_SEGMENTS = ("NON_USER", "LITE", "HEAVY")
    CALL_SEGMENTS = ("LOW", "MEDIUM", "HIGH")
    ARPU_SEGMENTS = ("LOW", "MID", "HIGH")
    MAX_CAMPAIGNS = 10
    MAX_CAMPAIGN_CONTACTS = 5000
    PILOT_NOISE_STD = 0.804

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.trace = []
        self.plan_summary = {}

    def _log(self, event, **details):
        record = {"event": event, **details}
        self.trace.append(record)
        if self.verbose:
            print(json.dumps(record, sort_keys=True, default=str))

    @staticmethod
    def _filter(profile, filters):
        """Match precisely the four documented filter fields."""
        rows = profile
        for key, column in (
            ("filter_current_tariff", "current_tariff"),
            ("filter_arpu_segment", "arpu_segment"),
            ("filter_data_segment", "data_segment"),
            ("filter_call_segment", "call_segment"),
        ):
            if key in filters:
                rows = rows[rows[column] == filters[key]]
        return rows.sort_values("ID_NUMBER", kind="stable")

    def _subgroups(self, rows, base_filters):
        """Enumerate real, exactly addressable groups, never explicit ID lists.

        A broad group and its refinements overlap, so the planner must select
        only groups with disjoint IDs. Missing usage labels are not fabricated.
        """
        seen = set()
        for data in (None,) + self.DATA_SEGMENTS:
            for calls in (None,) + self.CALL_SEGMENTS:
                filters = dict(base_filters)
                if data is not None:
                    filters["filter_data_segment"] = data
                if calls is not None:
                    filters["filter_call_segment"] = calls
                matched = self._filter(rows, filters)
                if matched.empty:
                    continue
                ids = frozenset(matched["ID_NUMBER"].tolist())
                if ids in seen:
                    continue
                seen.add(ids)
                yield {
                    "filters": filters,
                    "ids": ids,
                    "n": len(matched),
                    "baseline": float(matched["predicted_arpu"].sum()),
                }

    def _fallback_options(self, profile, tariffs):
        """Find real low-exposure groups, including when hypotheses are absent."""
        options = []
        cells = profile.groupby(["current_tariff", "arpu_segment"], observed=True)
        for (source, arpu), rows in cells:
            if source not in tariffs or arpu not in self.ARPU_SEGMENTS:
                continue
            alternatives = sorted(t for t in tariffs if t != source)
            if not alternatives:
                continue
            filters = {"filter_current_tariff": source, "filter_arpu_segment": arpu}
            for group in self._subgroups(rows, filters):
                if group["n"] <= self.MAX_CAMPAIGN_CONTACTS:
                    options.append({
                        **group, "cell": (source, arpu),
                        "target": alternatives[0], "hypothesis": "fallback",
                    })
        return options

    def act(self, env):
        """Run up to eight fixed pilots and return 1–10 feasible campaigns.

        There are no adaptive repeats and no paid final channel except SMS.
        With too little cash to run any SMS pilot, one emergency free-push
        probe supplies evidence. Free push also provides a required nonempty
        final fallback. A contact reserve keeps that fallback feasible.
        """
        self.trace = []
        self.plan_summary = {}
        deadline = time.monotonic() + 280.0
        profile = env.customer_profile.copy()
        required = {"ID_NUMBER", "current_tariff", "arpu_segment",
                    "data_segment", "call_segment", "predicted_arpu"}
        missing = required.difference(profile.columns)
        if missing:
            raise ValueError("Missing public profile columns: " + ", ".join(sorted(missing)))
        numeric = pd.to_numeric(profile["predicted_arpu"], errors="coerce")
        valid = numeric.map(lambda value: math.isfinite(value))
        if not valid.all():
            self._log("invalid_forecasts", count=int((~valid).sum()),
                      handling="Zero valuation on a private copy; source unchanged.")
        profile["predicted_arpu"] = numeric.where(valid, 0.0)
        tariffs = set(env.tariffs["tariff_plan_code"].dropna().astype(str))
        if "sms" not in env.channels or "push" not in env.channels:
            raise ValueError("The documented SMS and push channels are required.")
        sms_cost = float(env.channels["sms"]["cost_per_contact"])
        if not math.isfinite(sms_cost) or sms_cost < 0:
            raise ValueError("Invalid public SMS contact cost.")

        fallback_options = self._fallback_options(profile, tariffs)
        feasible_fallbacks = [g for g in fallback_options
                              if g["n"] <= int(env.remaining_contacts)]
        if not feasible_fallbacks:
            raise ValueError("No supported nonempty campaign fits the remaining contact limit.")
        final_reserve = min(g["n"] for g in feasible_fallbacks)
        self._log("policy", variant="v1_rules", hypotheses=8, pilot_size=200,
                  final_contact_reserve=final_reserve,
                  ranking="Observed SMS lift × forecast baseline − SMS cost",
                  overlap_assumption="Pilot IDs are unavailable; no claim of exact incremental rollout lift.")

        observations = []
        sms_pilots_ran = 0
        for name, source, arpu, target in self.HYPOTHESES:
            filters = {"filter_current_tariff": source, "filter_arpu_segment": arpu}
            rows = self._filter(profile, filters)
            if source not in tariffs or target not in tariffs or rows.empty:
                self._log("candidate_rejected", hypothesis=name,
                          reason="Missing tariff or empty supported audience")
                continue
            if time.monotonic() >= deadline or int(env.pilots_left) <= 0:
                self._log("candidate_rejected", hypothesis=name, reason="Pilot or time limit")
                continue
            affordable = max(0, int(env.remaining_contacts) - final_reserve)
            if sms_cost > 0:
                affordable = min(affordable, int(float(env.remaining_budget) // sms_cost))
            n_request = min(200, len(rows), affordable)
            if n_request < 10:
                self._log("candidate_rejected", hypothesis=name,
                          reason="Cannot request the minimum 10 while reserving final contacts")
                continue
            try:
                result = env.run_pilot(
                    target_tariff=target, channel="sms", n_customers=int(n_request), **filters
                )
                sms_pilots_ran += 1
            except (RuntimeError, ValueError) as error:
                self._log("pilot_failed", hypothesis=name, reason=str(error))
                continue
            actual_n = int(result["n_customers"])
            ratio = float(result["observed_lift_ratio"])
            if actual_n <= 0 or not math.isfinite(ratio):
                self._log("pilot_rejected", hypothesis=name,
                          reason="Nonfinite observation or zero actual sample", result=dict(result))
                continue
            standard_error = self.PILOT_NOISE_STD / math.sqrt(actual_n)
            observation = {
                "hypothesis": name, "cell": (source, arpu), "target": target,
                "filters": filters, "ratio": ratio, "actual_n": actual_n,
                "standard_error": standard_error,
                "full_cell_net": ratio * float(rows["predicted_arpu"].sum()) - sms_cost * len(rows),
            }
            observations.append(observation)
            self._log("pilot", hypothesis=name, filters=filters, target_tariff=target,
                      channel="sms", requested_n=int(n_request), actual_n=actual_n,
                      cost=float(result["cost"]), observed_lift_ratio=ratio,
                      standard_error=standard_error,
                      nominal_95_interval=[ratio - 1.96 * standard_error,
                                           ratio + 1.96 * standard_error],
                      selection_warning="Single noisy estimate; no adaptive confirmation.")

        if (sms_pilots_ran == 0 and sms_cost > 0
                and float(env.remaining_budget) < 10 * sms_cost
                and int(env.pilots_left) > 0 and time.monotonic() < deadline):
            emergency = self._emergency_push_probe(
                env, profile, tariffs, fallback_options, final_reserve, sms_cost
            )
            if emergency is not None:
                observations.append(emergency)

        # One preferred target per source-tariff/ARPU cell prevents H6/H8 overlap.
        winners = {}
        for observation in observations:
            cell = observation["cell"]
            if cell not in winners or observation["full_cell_net"] > winners[cell]["full_cell_net"]:
                winners[cell] = observation
        for observation in observations:
            if winners[observation["cell"]] is not observation:
                self._log("candidate_rejected", hypothesis=observation["hypothesis"],
                          reason="Competing target has greater observed full-cell net value")

        options = []
        for observation in winners.values():
            rows = self._filter(profile, observation["filters"])
            profitable = False
            for group in self._subgroups(rows, observation["filters"]):
                if group["n"] > self.MAX_CAMPAIGN_CONTACTS:
                    continue
                # observed_lift_ratio already includes the SMS channel response.
                estimated_net = observation["ratio"] * group["baseline"] - sms_cost * group["n"]
                if math.isfinite(estimated_net) and estimated_net > 0:
                    profitable = True
                    options.append({**group, **{key: observation[key] for key in
                                               ("hypothesis", "cell", "target", "ratio")},
                                    "net": estimated_net, "cost": sms_cost * group["n"]})
            if not profitable:
                self._log("candidate_rejected", hypothesis=observation["hypothesis"],
                          reason="No supported subgroup has positive estimated SMS net value")

        remaining_contacts = int(env.remaining_contacts)
        remaining_budget = float(env.remaining_budget)
        used_ids = set()
        campaigns = []
        selected_options = []
        while len(campaigns) < self.MAX_CAMPAIGNS:
            feasible = [option for option in options
                        if option["n"] <= remaining_contacts
                        and option["cost"] <= remaining_budget + 1e-9
                        and not used_ids.intersection(option["ids"])]
            if not feasible:
                break
            best = max(feasible, key=lambda g: (g["net"], g["net"] / g["n"], -g["n"]))
            campaign = {
                "campaign_name": "v1_%s_%02d" % (best["hypothesis"], len(campaigns) + 1),
                **best["filters"], "target_tariff": best["target"], "channel": "sms",
            }
            campaigns.append(campaign)
            selected_options.append(best)
            used_ids.update(best["ids"])
            remaining_contacts -= best["n"]
            remaining_budget -= best["cost"]
            self._log("campaign_selected", campaign=campaign, n_contacts=best["n"],
                      forecast_baseline=best["baseline"], estimated_final_net=best["net"],
                      communication_cost=best["cost"],
                      reason="Largest positive feasible group; disjoint final contacts")

        if not campaigns:
            campaign, fallback = self._fallback(env, fallback_options, winners)
            campaigns.append(campaign)
            selected_options.append(fallback)
            remaining_contacts -= fallback["n"]
            remaining_budget -= fallback["cost"]

        final_n = sum(group["n"] for group in selected_options)
        final_cost = sum(group["cost"] for group in selected_options)
        self.plan_summary = {
            "variant": "v1_rules", "n_pilots": len(observations),
            "n_final_campaigns": len(campaigns), "final_contacts": final_n,
            "final_cost": float(final_cost),
            "estimated_final_net_before_pilot_deduplication": float(sum(g["net"] for g in selected_options)),
            "remaining_contacts_after_final": remaining_contacts,
            "remaining_budget_after_final": float(remaining_budget),
            "final_campaigns_overlap": False,
            "pilot_overlap": "Unknown pilot IDs; repeated contacts consume resources and do not add the same lift twice.",
            "uses_scoring_truncation": False,
        }
        self._log("plan", **self.plan_summary)
        return campaigns

    def _emergency_push_probe(self, env, profile, tariffs, fallback_options, reserve, sms_cost):
        """Buy evidence with contacts when no SMS pilot can be funded.

        The return value uses SMS-equivalent units, including the observation
        standard error. This one-off exception never changes ordinary V1 runs.
        """
        if float(env.channels["push"]["cost_per_contact"]) != 0.0:
            return None
        push_multiplier = float(env.channels["push"]["conversion_multiplier"])
        sms_multiplier = float(env.channels["sms"]["conversion_multiplier"])
        if not (push_multiplier > 0 and sms_multiplier > 0):
            return None
        scale = sms_multiplier / push_multiplier
        candidates = list(self.HYPOTHESES)
        seen_cells = {(source, arpu) for _, source, arpu, target in candidates
                      if source in tariffs and target in tariffs and source != target}
        for group in sorted(fallback_options, key=lambda g: (g["n"], g["baseline"])):
            source, arpu = group["cell"]
            if (source, arpu) not in seen_cells:
                seen_cells.add((source, arpu))
                candidates.append(("cash_fallback", source, arpu, group["target"]))
        for name, source, arpu, target in candidates:
            if source not in tariffs or target not in tariffs or source == target:
                continue
            filters = {"filter_current_tariff": source, "filter_arpu_segment": arpu}
            rows = self._filter(profile, filters)
            n_request = min(200, len(rows), max(0, int(env.remaining_contacts) - reserve))
            if n_request < 10:
                continue
            try:
                result = env.run_pilot(
                    target_tariff=target, channel="push", n_customers=int(n_request), **filters
                )
            except (RuntimeError, ValueError) as error:
                self._log("pilot_failed", hypothesis=name, channel="push", reason=str(error))
                return None
            actual_n = int(result["n_customers"])
            raw_ratio = float(result["observed_lift_ratio"])
            if actual_n <= 0 or not math.isfinite(raw_ratio):
                self._log("pilot_rejected", hypothesis=name, channel="push",
                          reason="Nonfinite emergency observation or zero actual sample")
                return None
            ratio = raw_ratio * scale
            raw_standard_error = self.PILOT_NOISE_STD / math.sqrt(actual_n)
            standard_error = raw_standard_error * scale
            self._log("pilot", hypothesis=name, filters=filters, target_tariff=target,
                      channel="push", reason="No cash for any SMS pilot; one emergency probe",
                      requested_n=int(n_request), actual_n=actual_n, cost=float(result["cost"]),
                      observed_lift_ratio=raw_ratio, raw_standard_error=raw_standard_error,
                      sms_equivalent_ratio=ratio, standard_error=standard_error,
                      nominal_95_interval=[ratio - 1.96 * standard_error,
                                           ratio + 1.96 * standard_error],
                      interval_units="SMS-equivalent relative lift", final_contact_reserve=reserve)
            return {
                "hypothesis": name, "cell": (source, arpu), "target": target,
                "filters": filters, "ratio": ratio, "actual_n": actual_n,
                "standard_error": standard_error,
                "full_cell_net": ratio * float(rows["predicted_arpu"].sum()) - sms_cost * len(rows),
            }
        return None

    def _fallback(self, env, fallback_options, winners):
        """Meet the guide's minimum one campaign with a real free-push group.

        Prefer a measured cell and the group with the least estimated loss.
        Without any measured cell, choose minimum forecast exposure and clearly
        label the zero estimate as an uninformed assumption, not positive proof.
        """
        push_cost = float(env.channels["push"]["cost_per_contact"])
        if push_cost != 0.0:
            raise ValueError("The documented free-push fallback is unavailable.")
        sms_multiplier = float(env.channels["sms"]["conversion_multiplier"])
        push_multiplier = float(env.channels["push"]["conversion_multiplier"])
        scale = push_multiplier / sms_multiplier
        feasible = [g for g in fallback_options if g["n"] <= int(env.remaining_contacts)]
        measured = []
        for group in feasible:
            evidence = winners.get(group["cell"])
            if evidence is not None:
                estimate = evidence["ratio"] * scale * group["baseline"]
                if math.isfinite(estimate):
                    measured.append({**group, "target": evidence["target"],
                                     "hypothesis": evidence["hypothesis"],
                                     "net": estimate, "cost": 0.0})
        if measured:
            best = max(measured, key=lambda g: (g["net"], -g["baseline"], -g["n"]))
            evidence_note = "SMS estimate scaled to free push; choose the least estimated downside."
        else:
            group = min(feasible, key=lambda g: (g["baseline"], g["n"], g["target"]))
            best = {**group, "net": 0.0, "cost": 0.0}
            evidence_note = "No pilot evidence; choose minimum forecast exposure, with unproven zero estimate."
        campaign = {
            "campaign_name": "v1_required_push_fallback", **best["filters"],
            "target_tariff": best["target"], "channel": "push",
        }
        self._log("no_positive_evidence", policy="Required nonempty final campaign",
                  reason="No profitable SMS group fits the remaining limits.",
                  campaign=campaign, n_contacts=best["n"], estimated_final_net=best["net"],
                  evidence=evidence_note)
        return campaign, best
