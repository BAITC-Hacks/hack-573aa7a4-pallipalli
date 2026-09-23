"""Offline campaign planner using public history and adaptive, noisy pilots."""
from pathlib import Path
import math

import numpy as np
import pandas as pd

from planner import select_plan


class Agent:
    # Publicly documented pilot noise. History is a weak prior because the
    # judging population differs from historical subscribers.
    NOISE = 0.804
    PRIOR_SD = 0.25
    CONFIDENCE_Z = 1.645

    def _history(self):
        path = Path(__file__).resolve().parent / "data" / "change_tariff.csv"
        if not path.exists():
            return {}
        h = pd.read_csv(path)
        before = pd.to_numeric(h["AVG_ARPU_PREV_3M"], errors="coerce")
        after = pd.to_numeric(h["AVG_ARPU_NEXT_3M"], errors="coerce")
        h = h.loc[(before >= 100) & np.isfinite(before) & np.isfinite(after)].copy()
        h["segment"] = np.where(before.loc[h.index] < 1000, "LOW",
                                np.where(before.loc[h.index] <= 5000, "MID", "HIGH"))
        h["lift"] = ((after.loc[h.index] - before.loc[h.index]) /
                     before.loc[h.index]).clip(-1, 3)
        keys = ["tariff_plan_code_from", "segment"]
        totals = h.groupby(keys, observed=True).size()
        priors = {}
        for key, rows in h.groupby(keys + ["tariff_plan_code_to"], observed=True):
            # Migration frequency is a ranking proxy, not an identified causal
            # conversion probability: non-migrating subscribers are absent.
            support = len(rows)
            share = support / float(totals.loc[key[:2]])
            shrinkage = support / (support + 30.0)
            priors[key] = float(rows.lift.mean() * share * shrinkage)
        return priors

    def _estimate(self, candidate):
        # Normalize unsaturated channel observations AND their noise variance.
        precision = 1.0 / self.PRIOR_SD ** 2
        weighted = candidate["prior"] * precision
        for observation in candidate["observations"]:
            ratio, n = observation[:2]
            multiplier = observation[2] if len(observation) > 2 else 1.0
            p = n * multiplier ** 2 / self.NOISE ** 2
            precision += p
            weighted += ratio / multiplier * p
        mean = weighted / precision
        sd = math.sqrt(1.0 / precision)
        return mean, sd, mean - self.CONFIDENCE_Z * sd

    def _pilot(self, env, candidate):
        n = min(200, candidate["population"], int(env.remaining_contacts),
                self.pilot_contacts_left)
        if n < 10 or env.pilots_left <= 0:
            return False
        allowance = max(0, float(env.remaining_budget) - self.reserve_money)
        # Strongest unsaturated probe at or below the public SMS price.
        # Free probes remain available when the cash allowance is exhausted.
        choices = [(float(v["conversion_multiplier"]), -float(v["cost_per_contact"]), k)
                   for k, v in sorted(env.channels.items())
                   if 0 < v["conversion_multiplier"] <= 1
                   and v["cost_per_contact"] <= 4
                   and n * v["cost_per_contact"] <= allowance]
        if not choices:
            return False
        multiplier, _, channel = max(choices)
        result = env.run_pilot(n_customers=n, channel=channel, **candidate["campaign"])
        actual = int(result["n_customers"])
        observed = float(result["observed_lift_ratio"])
        self.pilot_contacts_left -= actual
        if actual > 0 and math.isfinite(observed):
            candidate["observations"].append((observed, actual, multiplier))
            return True
        return False

    @staticmethod
    def _channel_multiplier(settings):
        # For positive response, conversion saturation can reduce a multiplier
        # above 1 to as little as 1. Do not assume that calls get the full 1.2.
        return min(1.0, float(settings["conversion_multiplier"]))

    @staticmethod
    def _incremental_response(deployment_ratio, pilot_credits):
        """Expected additional ratio after independent uniform pilot draws.

        Credits are (nonnegative estimated ratio, probability of contact).
        Attribute already-earned revenue to the strongest prior contact only.
        This is an expectation under the public sampling protocol, not a
        guarantee about unseen identities or heterogeneous individual effects.
        """
        deployment_ratio = max(0.0, float(deployment_ratio))
        unseen, credit = 1.0, 0.0
        for ratio, probability in sorted(pilot_credits, reverse=True):
            probability = min(1.0, max(0.0, float(probability)))
            credit += min(deployment_ratio, max(0.0, ratio)) * probability * unseen
            unseen *= 1 - probability
        return max(0.0, deployment_ratio - credit)

    def act(self, env):
        profile = env.customer_profile.sort_values("ID_NUMBER")
        priors = self._history()
        tariffs = sorted(str(t) for t in env.tariffs["tariff_plan_code"])
        self.reserve_money = float(env.remaining_budget) * 0.84
        self.pilot_contacts_left = min(4000, int(env.remaining_contacts * 0.27))
        candidates = []
        for (current, segment), rows in profile.groupby(
                ["current_tariff", "arpu_segment"], observed=True, sort=True):
            values = rows.predicted_arpu.to_numpy(dtype=float)[:5000]
            if len(rows) < 10 or not np.isfinite(values).all():
                continue
            cell = []
            for target in tariffs:
                if target == str(current):
                    continue
                prior = priors.get((str(current), str(segment), target), 0.0)
                ranks = []
                for settings in env.channels.values():
                    cost = float(settings["cost_per_contact"])
                    n = min(len(values), int(env.remaining_contacts))
                    if cost:
                        n = min(n, int(env.remaining_budget // cost))
                    ranks.append(float(values[:n].sum() * prior *
                                       self._channel_multiplier(settings) - n * cost))
                cell.append({
                    "cell": (str(current), str(segment)), "values": values,
                    "population": len(rows), "prior": prior,
                    "rank": max(ranks), "observations": [],
                    "campaign": {"filter_current_tariff": str(current),
                                 "filter_arpu_segment": str(segment),
                                 "target_tariff": target},
                })
            cell.sort(key=lambda c: -c["rank"])
            candidates.extend(cell[:3])
        candidates.sort(key=lambda c: -c["rank"])
        if not candidates:
            raise ValueError("No eligible audience with at least 10 subscribers")

        # Explore different groups before feedback-driven follow-up pilots.
        seen = set()
        for c in candidates:
            if c["cell"] in seen or len(seen) >= 10:
                continue
            if self._pilot(env, c):
                seen.add(c["cell"])
            if env.pilots_left <= 0 or self.pilot_contacts_left < 10:
                break

        while env.pilots_left > 0 and self.pilot_contacts_left >= 10:
            choices = []
            for c in candidates:
                if len(c["observations"]) >= 3:
                    continue
                if not c["observations"] and c["prior"] <= 0 and priors:
                    continue
                mean, sd, _ = self._estimate(c)
                optimistic = mean + sd
                if optimistic <= 0:
                    continue
                if c["observations"]:
                    # Prefer learning the scale of valuable observed winners;
                    # an untried offer cannot win solely through a wide prior.
                    opportunity = max(0, mean + 0.5 * sd)
                    priority = float(c["values"].sum()) * opportunity / (1 + len(c["observations"]))
                else:
                    priority = float(c["values"].sum()) * max(0, c["prior"]) * 0.25
                if priority > 0:
                    choices.append((priority, c))
            if not choices:
                break
            progressed = False
            for _, c in sorted(choices, key=lambda item: -item[0]):
                if self._pilot(env, c):
                    progressed = True
                    break
            if not progressed:
                break

        explored = [c for c in candidates if c["observations"]]
        options = []
        for c in explored:
            mean, sd, lower = self._estimate(c)
            # Revenue objective: expected gain, with a one-sigma positive
            # evidence gate. Uncertainty controls eligibility, not the value
            # of every customer after the candidate clears that gate.
            response = max(0, mean) if mean - sd > 0 else 0.0
            for channel, settings in sorted(env.channels.items()):
                deployment_ratio = response * self._channel_multiplier(settings)
                credits = []
                for previous in explored:
                    if previous["cell"] != c["cell"]:
                        continue
                    mean, sd, _ = self._estimate(previous)
                    for _, n, multiplier in previous["observations"]:
                        # Uniform random pilot draws: expected overlap, not
                        # the previous worst case of all pilots in the prefix.
                        ratio = max(0, mean + sd) * multiplier
                        credits.append((ratio, min(1, n / c["population"])))
                options.append(dict(c,
                    campaign=dict(c["campaign"], channel=channel),
                    gain_ratio=self._incremental_response(deployment_ratio, credits),
                    pilot_contacts=0,
                    cost=float(settings["cost_per_contact"])))
        selected, self.plan_diagnostics = select_plan(
            options, env.remaining_budget, env.remaining_contacts)
        self.plan_diagnostics.update(
            pilot_policy="affordable_normalized_probes",
            objective="posterior_expected_net_with_one_sigma_gate",
            overlap="expected_maximum_prior_pilot_credit")
        plan = [dict(options[i]["campaign"], campaign_name=f"revenue_{j + 1}")
                for j, i in enumerate(selected)]

        if not plan:
            # Required one-campaign fallback limits exposure, not ARPU risk.
            free = [k for k, v in sorted(env.channels.items()) if v["cost_per_contact"] == 0]
            if not free:
                raise ValueError("No free channel for mandatory conservative fallback")
            c = min(candidates, key=lambda c: (len(c["values"]), -c["rank"]))
            if not c["observations"] and env.pilots_left and self.pilot_contacts_left >= 10:
                self._pilot(env, c)
            same_cell = [x for x in candidates if x["cell"] == c["cell"] and x["observations"]]
            if same_cell:
                c = max(same_cell, key=lambda x: self._estimate(x)[2])
            plan = [dict(c["campaign"], channel=free[0], campaign_name="minimum_exposure_fallback")]
        return plan
