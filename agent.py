"""Offline campaign planner using public history and adaptive, noisy pilots."""
from pathlib import Path
import math

import numpy as np
import pandas as pd


class Agent:
    # Publicly documented pilot noise. History is a weak prior because the
    # judging population differs from historical subscribers.
    NOISE = 0.804
    PRIOR_SD = 0.15
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
        precision = 1.0 / self.PRIOR_SD ** 2
        weighted = candidate["prior"] * precision
        for observation, n in candidate["observations"]:
            p = n / self.NOISE ** 2
            precision += p
            weighted += observation * p
        mean = weighted / precision
        sd = math.sqrt(1.0 / precision)
        return mean, sd, mean - self.CONFIDENCE_Z * sd

    def _pilot(self, env, candidate):
        cost = float(env.channels[candidate["campaign"]["channel"]]["cost_per_contact"])
        n = min(200, len(candidate["values"]), int(env.remaining_contacts))
        # Keep most money and contacts available for the final plan.
        if cost:
            n = min(n, int(max(0, env.remaining_budget - self.reserve_money) // cost))
        if n < 10 or env.pilots_left <= 0:
            return False
        result = env.run_pilot(n_customers=n, **candidate["campaign"])
        observed = float(result["observed_lift_ratio"])
        actual = int(result["n_customers"])
        if actual > 0 and math.isfinite(observed):
            candidate["observations"].append((observed, actual))
            return True
        return False

    def act(self, env):
        profile = env.customer_profile.sort_values("ID_NUMBER")
        priors = self._history()
        tariffs = sorted(str(t) for t in env.tariffs["tariff_plan_code"])
        initial_budget = float(env.remaining_budget)
        self.reserve_money = initial_budget * 0.75
        pilot_contact_limit = min(4000, int(env.remaining_contacts * 0.27))
        starting_contacts = int(env.remaining_contacts)
        candidates = []
        for (current, segment), rows in profile.groupby(
                ["current_tariff", "arpu_segment"], observed=True, sort=True):
            values = rows.predicted_arpu.to_numpy(dtype=float)[:5000]
            if len(values) < 10 or not np.isfinite(values).all():
                continue
            cell_candidates = []
            for target in tariffs:
                if target == str(current):
                    continue
                base = priors.get((str(current), str(segment), target), 0.0)
                for channel, settings in sorted(env.channels.items()):
                    cost = float(settings["cost_per_contact"])
                    prior = base * float(settings["conversion_multiplier"])
                    affordable = len(values) if cost == 0 else min(
                        len(values), int(initial_budget // cost))
                    rank = float(values[:affordable].sum() * prior - affordable * cost)
                    # Paid pilots must leave the deployment reserve intact.
                    if cost * min(200, len(values)) > initial_budget * 0.25:
                        continue
                    cell_candidates.append({
                        "cell": (str(current), str(segment)), "values": values,
                        "prior": prior, "rank": rank, "observations": [],
                        "campaign": {"filter_current_tariff": str(current),
                                     "filter_arpu_segment": str(segment),
                                     "target_tariff": target, "channel": channel},
                    })
            # One preferred channel per destination, up to two alternatives per
            # cell. This prevents one large group consuming all exploration.
            cell_candidates.sort(key=lambda c: -c["rank"])
            targets = set()
            for candidate in cell_candidates:
                target = candidate["campaign"]["target_tariff"]
                if target not in targets:
                    candidates.append(candidate)
                    targets.add(target)
                if len(targets) == 2:
                    break
        candidates.sort(key=lambda c: -c["rank"])
        if not candidates:
            raise ValueError("No eligible audience with at least 10 subscribers")

        # Explore distinct cells before spending pilots on second choices.
        first, alternatives, seen = [], [], set()
        for c in candidates:
            if c["cell"] in seen:
                alternatives.append(c)
            else:
                first.append(c)
                seen.add(c["cell"])
        explored = []
        for c in (first[:8] + alternatives + first[8:]):
            if len(explored) >= 12 or env.pilots_left <= 0:
                break
            if starting_contacts - env.remaining_contacts + 200 > pilot_contact_limit:
                break
            if self._pilot(env, c):
                explored.append(c)

        # Recheck promising candidates: upper credible value prioritizes
        # uncertain opportunities, while the final plan uses a lower bound.
        while env.pilots_left > 0 and explored:
            if starting_contacts - env.remaining_contacts + 200 > pilot_contact_limit:
                break
            choices = []
            for c in explored:
                mean, sd, _ = self._estimate(c)
                cost = float(env.channels[c["campaign"]["channel"]]["cost_per_contact"])
                upper = float(c["values"].sum() * (mean + sd) - len(c["values"]) * cost)
                if upper > 0 and len(c["observations"]) < 4:
                    choices.append((upper / (len(c["observations"]) + 1), c))
            if not choices:
                break
            c = max(choices, key=lambda item: item[0])[1]
            if not self._pilot(env, c):
                break

        # Only one final campaign per disjoint tariff/ARPU cell. Actual pilot
        # identities are private, so pilot/final overlap cannot be removed;
        # discount the possible duplicated lift conservatively instead.
        plan, used_cells = [], set()
        budget, contacts = float(env.remaining_budget), int(env.remaining_contacts)
        while len(plan) < 10 and contacts > 0:
            options = []
            for c in explored:
                if c["cell"] in used_cells:
                    continue
                cost = float(env.channels[c["campaign"]["channel"]]["cost_per_contact"])
                n = min(len(c["values"]), contacts)
                if cost:
                    n = min(n, int(budget // cost))
                if n <= 0:
                    continue
                _, _, lower = self._estimate(c)
                pilot_n = sum(n0 for other in explored if other["cell"] == c["cell"]
                              for _, n0 in other["observations"])
                values = c["values"][:n]
                overlap = min(n, pilot_n)
                fresh_arpu = float(values.sum() - np.sort(values)[-overlap:].sum()) if overlap else float(values.sum())
                gain = fresh_arpu * lower - n * cost
                if lower > 0 and gain > 0:
                    options.append((gain, c, n, cost))
            if not options:
                break
            _, c, n, cost = max(options, key=lambda item: item[0])
            plan.append(dict(c["campaign"], campaign_name=f"history_pilot_{len(plan) + 1}"))
            used_cells.add(c["cell"])
            contacts -= n
            budget -= n * cost

        if not plan:
            # The case requires 1–10 campaigns even when evidence is weak.
            # Use the smallest existing cell and a free channel to limit the
            # unavoidable exposure. This is not a claim of profitability.
            free = [k for k, v in sorted(env.channels.items()) if v["cost_per_contact"] == 0]
            if not free:
                raise ValueError("No free channel for mandatory conservative fallback")
            c = min(candidates, key=lambda c: (len(c["values"]), -c["rank"]))
            fallback = dict(c["campaign"], channel=free[0])
            fallback_c = dict(c, campaign=fallback, observations=[])
            if env.pilots_left and env.remaining_contacts >= 10:
                self._pilot(env, fallback_c)
            # Pick the strongest observed destination within this small cell.
            same_cell = [x for x in explored + [fallback_c]
                         if x["cell"] == c["cell"] and x["observations"]]
            if same_cell:
                best = max(same_cell, key=lambda x: self._estimate(x)[2])
                fallback["target_tariff"] = best["campaign"]["target_tariff"]
            plan = [dict(fallback, campaign_name="minimum_exposure_fallback")]
        return plan
