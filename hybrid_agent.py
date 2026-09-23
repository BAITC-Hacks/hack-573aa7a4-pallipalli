"""Controlled additions to the frozen numerical agent.

The source branch supplies the ideas of evidence-dependent prior variance,
knowledge-gradient exploration, and merging compatible campaign audiences.
Flags isolate these additions while retaining our expected-value planner.
"""
from pathlib import Path
import math

import numpy as np

from frozen_agent import Agent as FrozenAgent
from planner import select_plan


class Agent(FrozenAgent):
    USE_HIERARCHY = True
    USE_KG = True
    USE_GROUPS = True
    OPTIMISTIC_SHORTLIST = True
    TARGETS_PER_CELL = 4

    def _estimate(self, candidate):
        precision = 1.0 / candidate.get("prior_sd", self.PRIOR_SD) ** 2
        weighted = candidate["prior"] * precision
        for observation in candidate["observations"]:
            ratio, n = observation[:2]
            multiplier = observation[2] if len(observation) > 2 else 1.0
            p = n * multiplier ** 2 / self.NOISE ** 2
            precision += p
            weighted += ratio / multiplier * p
        mean, sd = weighted / precision, math.sqrt(1.0 / precision)
        return mean, sd, mean - self.CONFIDENCE_Z * sd

    def _candidates(self, env):
        profile = env.customer_profile.sort_values("ID_NUMBER")
        if self.USE_HIERARCHY:
            from prior_model import history_priors
            priors = history_priors(Path(__file__).parent / "data/change_tariff.csv", env.tariffs)
        else:
            priors = self._history()
        candidates = []
        tariffs = sorted(str(t) for t in env.tariffs["tariff_plan_code"])
        for (current, segment), rows in profile.groupby(
                ["current_tariff", "arpu_segment"], observed=True, sort=True):
            values = rows.predicted_arpu.to_numpy(dtype=float)[:5000]
            ids = rows.ID_NUMBER.to_numpy()[:5000]
            if len(rows) < 10 or not np.isfinite(values).all():
                continue
            cell = []
            for target in tariffs:
                if target == str(current):
                    continue
                key = (str(current), str(segment), target)
                prior, sd = priors.get(key, (0.0, self.PRIOR_SD)) if self.USE_HIERARCHY else (
                    priors.get(key, 0.0), self.PRIOR_SD)
                ranking_response = prior + sd if self.OPTIMISTIC_SHORTLIST else prior
                ranks = []
                for settings in env.channels.values():
                    cost = float(settings["cost_per_contact"])
                    n = min(len(values), int(env.remaining_contacts))
                    if cost:
                        n = min(n, int(env.remaining_budget // cost))
                    ranks.append(float(values[:n].sum() * ranking_response *
                                       self._channel_multiplier(settings) - n * cost))
                cell.append({
                    "cell": (str(current), str(segment)), "values": values, "ids": ids,
                    "population": len(rows), "prior": prior, "prior_sd": sd,
                    "rank": max(ranks), "observations": [],
                    "campaign": {"filter_current_tariff": str(current),
                                 "filter_arpu_segment": str(segment), "target_tariff": target},
                })
            cell.sort(key=lambda c: -c["rank"])
            candidates.extend(cell[:self.TARGETS_PER_CELL])
        candidates.sort(key=lambda c: -c["rank"])
        return candidates, bool(priors)

    def _legacy_explore(self, env, candidates, has_history):
        seen = set()
        for candidate in candidates:
            if candidate["cell"] in seen or len(seen) >= 10:
                continue
            if self._pilot(env, candidate):
                seen.add(candidate["cell"])
            if env.pilots_left <= 0 or self.pilot_contacts_left < 10:
                break
        while env.pilots_left > 0 and self.pilot_contacts_left >= 10:
            choices = []
            for candidate in candidates:
                if len(candidate["observations"]) >= 3:
                    continue
                if not candidate["observations"] and candidate["prior"] <= 0 and has_history:
                    continue
                mean, sd, _ = self._estimate(candidate)
                if mean + sd <= 0:
                    continue
                if candidate["observations"]:
                    priority = float(candidate["values"].sum()) * max(0, mean + .5 * sd) / (
                        1 + len(candidate["observations"]))
                else:
                    priority = float(candidate["values"].sum()) * max(0, candidate["prior"]) * .25
                if priority > 0:
                    choices.append((priority, candidate))
            if not choices:
                break
            progressed = False
            for _, candidate in sorted(choices, key=lambda item: -item[0]):
                if self._pilot(env, candidate):
                    progressed = True
                    break
            if not progressed:
                break

    def _options(self, env, candidates):
        explored = [c for c in candidates if c["observations"]]
        options = []
        for candidate in explored:
            mean, sd, _ = self._estimate(candidate)
            response = max(0, mean) if mean - sd > 0 else 0.0
            for channel, settings in sorted(env.channels.items()):
                deployment_ratio = response * self._channel_multiplier(settings)
                credits = []
                for previous in explored:
                    if previous["cell"] != candidate["cell"]:
                        continue
                    mean, sd, _ = self._estimate(previous)
                    for _, n, multiplier in previous["observations"]:
                        ratio = max(0, mean + sd) * multiplier
                        credits.append((ratio, min(1, n / candidate["population"])))
                unseen, credit = 1.0, 0.0
                for ratio, probability in sorted(credits, reverse=True):
                    credit += min(deployment_ratio, ratio) * probability * unseen
                    unseen *= 1 - probability
                options.append(dict(candidate,
                    campaign=dict(candidate["campaign"], channel=channel),
                    lower=max(0, deployment_ratio - credit), pilot_contacts=0,
                    cost=float(settings["cost_per_contact"])))
        return options

    def act(self, env):
        self.reserve_money = float(env.remaining_budget) * .84
        self.pilot_contacts_left = min(4000, int(env.remaining_contacts * .27))
        candidates, has_history = self._candidates(env)
        if not candidates:
            raise ValueError("No eligible audience with at least 10 subscribers")
        if self.USE_KG:
            from kg_policy import explore
            explore(self, env, candidates)
        else:
            self._legacy_explore(env, candidates, has_history)
        options = self._options(env, candidates)
        if self.USE_GROUPS:
            from grouped_planner import select_grouped_plan
            plan, self.plan_diagnostics = select_grouped_plan(
                options, env.remaining_budget, env.remaining_contacts)
        else:
            selected, self.plan_diagnostics = select_plan(
                options, env.remaining_budget, env.remaining_contacts)
            plan = [dict(options[i]["campaign"], campaign_name=f"revenue_{j + 1}")
                    for j, i in enumerate(selected)]
        if not plan:
            free = [k for k, v in sorted(env.channels.items()) if v["cost_per_contact"] == 0]
            if not free:
                raise ValueError("No free channel for mandatory fallback")
            candidate = min(candidates, key=lambda c: (len(c["values"]), -c["rank"]))
            if not candidate["observations"] and env.pilots_left and self.pilot_contacts_left >= 10:
                self._pilot(env, candidate)
            same_cell = [c for c in candidates if c["cell"] == candidate["cell"] and c["observations"]]
            if same_cell:
                candidate = max(same_cell, key=lambda c: self._estimate(c)[2])
            plan = [dict(candidate["campaign"], channel=free[0], campaign_name="minimum_exposure_fallback")]
        self.plan_diagnostics["additions"] = {
            "hierarchical_prior": self.USE_HIERARCHY, "knowledge_gradient": self.USE_KG,
            "grouping": self.USE_GROUPS, "optimistic_shortlist": self.OPTIMISTIC_SHORTLIST,
            "targets_per_cell": self.TARGETS_PER_CELL,
        }
        self.plan_diagnostics["exploration"] = getattr(self, "exploration_diagnostics", {})
        return plan


class RefactoredControl(Agent):
    USE_HIERARCHY = USE_KG = USE_GROUPS = OPTIMISTIC_SHORTLIST = False
    TARGETS_PER_CELL = 3


class PriorOnly(RefactoredControl):
    USE_HIERARCHY = True


class KnowledgeGradientOnly(RefactoredControl):
    USE_KG = True


class GroupingOnly(RefactoredControl):
    USE_GROUPS = True


class PriorKnowledgeGradient(Agent):
    USE_GROUPS = False


class BroadOnly(RefactoredControl):
    OPTIMISTIC_SHORTLIST = True
    TARGETS_PER_CELL = 4
