"""Campaign grouping on public audience rows and already estimated responses.

Merges combine different current tariffs with an identical destination, channel,
ARPU segment and any other filters. Full audiences must fit under 5,000; a
budget-truncated count is never used to justify a merge. Search uses exact
ID-ordered gain prefixes and retains the ungrouped planner as an incumbent.

The search is heuristic: compatible pairs plus deterministic greedy packs are
generated, then at most MAX_MERGED_OPTIONS merges enter bounded beam search.
"""
from itertools import combinations
import math

import numpy as np

from planner import _gain_curve, select_plan


MAX_AUDIENCE = 5000
MAX_MERGED_OPTIONS = 128


def _prepare_options(options):
    """Return (campaign, member cells, cost, prefix curve) search alternatives."""
    prepared, buckets = [], {}
    skipped_incomplete, generated = 0, {}
    for index, option in enumerate(options):
        ratio = float(option.get("gain_ratio", option.get("lower", 0.0)))
        values = np.asarray(option["values"], dtype=float)[:MAX_AUDIENCE]
        cost = float(option["cost"])
        if not math.isfinite(ratio) or ratio <= 0 or not len(values):
            continue
        if cost < 0 or not math.isfinite(cost) or not np.isfinite(values).all():
            raise ValueError("Campaign costs and customer values must be finite; cost cannot be negative")
        campaign = dict(option["campaign"])
        curve = _gain_curve(values, ratio, option.get("pilot_contacts", 0), cost)
        prepared.append({"campaign": campaign, "cells": frozenset([option["cell"]]),
                         "cost": cost, "curve": curve, "source_indices": (index,)})

        # Merged audiences require all identities. Singleton values may still
        # represent the first 5,000 customers of a larger current-tariff cell.
        population = int(option.get("population", len(values)))
        ids = np.asarray(option.get("ids", []))
        current = str(campaign.get("filter_current_tariff", ""))
        if (population > MAX_AUDIENCE or population != len(values)
                or ids.ndim != 1 or len(ids) != population or not current
                or ";" in current or int(option.get("pilot_contacts", 0)) != 0):
            skipped_incomplete += 1
            continue
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Customer IDs within one cell must be unique")
        # Including all remaining filter fields prevents accidental merging of
        # audiences with different traffic/call filters in future extensions.
        signature = tuple(sorted((key, str(value)) for key, value in campaign.items()
                                 if key not in {"campaign_name", "filter_current_tariff"}))
        bucket = buckets.setdefault((signature, cost), [])
        bucket.append({"index": index, "cell": option["cell"], "current": current,
                       "campaign": campaign, "population": population, "ids": ids,
                       "lift": values * ratio, "cost": cost,
                       "net": float(np.sum(values) * ratio - population * cost)})

    def remember(members):
        if len(members) < 2:
            return
        if sum(m["population"] for m in members) > MAX_AUDIENCE:
            return
        if len({m["current"] for m in members}) != len(members):
            return
        if len({m["cell"] for m in members}) != len(members):
            return
        key = tuple(sorted(m["index"] for m in members))
        generated.setdefault(key, list(members))

    for members in buckets.values():
        # Pairs preserve small combinations; packs can free several campaign
        # slots simultaneously. No exponential subset enumeration is attempted.
        for pair in combinations(members, 2):
            remember(pair)
        orderings = [
            sorted(members, key=lambda m: (m["current"], m["index"])),
            sorted(members, key=lambda m: (-m["net"], m["index"])),
            sorted(members, key=lambda m: (-m["net"] / max(1, m["population"]), m["index"])),
            sorted(members, key=lambda m: (m["population"], m["index"])),
        ]
        for ordering in orderings:
            packs = []
            for member in ordering:
                for pack in packs:
                    if (sum(m["population"] for m in pack) + member["population"] <= MAX_AUDIENCE
                            and all(m["current"] != member["current"] for m in pack)):
                        pack.append(member)
                        break
                else:
                    packs.append([member])
            for pack in packs:
                remember(pack)

    ranked = sorted(generated.items(), key=lambda item: (
        -sum(m["net"] for m in item[1]), -len(item[1]), item[0]))
    skipped_identity_conflicts = 0
    for source_indices, members in ranked[:MAX_MERGED_OPTIONS]:
        ids = np.concatenate([m["ids"] for m in members])
        if len(np.unique(ids)) != len(ids):
            # A common identity contradicts disjoint tariff cells. Keep the
            # original options but do not use an incorrectly valued union.
            skipped_identity_conflicts += 1
            continue
        order = np.argsort(ids, kind="stable")
        lift = np.concatenate([m["lift"] for m in members])[order]
        cost = members[0]["cost"]
        curve = np.concatenate(([0.0], np.cumsum(lift))) - np.arange(len(lift) + 1) * cost
        campaign = dict(members[0]["campaign"])
        campaign["filter_current_tariff"] = ";".join(sorted(m["current"] for m in members))
        prepared.append({"campaign": campaign, "cells": frozenset(m["cell"] for m in members),
                         "cost": cost, "curve": curve, "source_indices": source_indices})
    diagnostics = {
        "group_generation": "compatible_pairs_and_four_deterministic_greedy_pack_orders",
        "max_merged_options": MAX_MERGED_OPTIONS,
        "merge_alternatives_generated": len(generated),
        "merge_alternatives_retained": min(len(generated), MAX_MERGED_OPTIONS) - skipped_identity_conflicts,
        "merge_alternatives_pruned": max(0, len(generated) - MAX_MERGED_OPTIONS),
        "merge_options_without_complete_identity_data": skipped_incomplete,
        "merge_options_with_identity_conflicts": skipped_identity_conflicts,
    }
    return prepared, diagnostics


def select_grouped_plan(options, budget, contacts, max_campaigns=10, beam_width=256):
    """Return ready campaign dictionaries and allocation diagnostics.

    Inputs follow planner.select_plan, adding campaign, population and IDs aligned
    with values. A gain_ratio (or lower) is one response estimate per original
    cell, already adjusted for pilot overlap; merge-eligible pilot_contacts is 0.
    The complete union audience must fit 5,000, even if deployment resources
    would reduce actual contacts. Alternative groups sharing a cell conflict.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    limit = max(0, min(10, int(max_campaigns)))
    budget, contacts = max(0.0, float(budget)), max(0, int(contacts))
    # This incumbent protects previous whole-plan performance if extra groups
    # crowd good singleton states out of the bounded beam.
    singleton_indices, incumbent_diagnostics = select_plan(
        options, budget, contacts, max_campaigns=limit, beam_width=beam_width)
    prepared, diagnostics = _prepare_options(options)
    cell_numbers = {cell: i for i, cell in enumerate(dict.fromkeys(o["cell"] for o in options))}
    for option in prepared:
        option["mask"] = sum(1 << cell_numbers[cell] for cell in option["cells"])
    initial = (0.0, budget, contacts, 0, ())

    def extend(state, index):
        score, money, remaining, mask, path = state
        option = prepared[index]
        if mask & option["mask"]:
            return None
        n = min(len(option["curve"]) - 1, remaining)
        if option["cost"]:
            n = min(n, int(money // option["cost"]))
        if n <= 0 or option["curve"][n] <= 0:
            return None
        return (score + float(option["curve"][n]), money - n * option["cost"], remaining - n,
                mask | option["mask"], path + (index,))

    def preference(state):
        return (-state[0], len(state[4]), -state[1], -state[2], state[4])

    singleton_map = {option["source_indices"][0]: index for index, option in enumerate(prepared)
                     if len(option["source_indices"]) == 1}
    incumbent = initial
    for index in singleton_indices:
        incumbent = extend(incumbent, singleton_map[index])
    greedy = initial
    while len(greedy[4]) < limit:
        choices = [child for i in range(len(prepared)) if (child := extend(greedy, i)) is not None]
        if not choices:
            break
        greedy = min(choices, key=preference)
    best = min(incumbent, greedy, key=preference)
    frontier, best_by_count = [initial], {0: 0.0}
    states_expanded, states_pruned = 0, 0
    for depth in range(1, limit + 1):
        deduplicated = {}
        for state in frontier:
            for index in range(len(prepared)):
                child = extend(state, index)
                if child is None:
                    continue
                states_expanded += 1
                key = (child[3], child[1], child[2])
                previous = deduplicated.get(key)
                if previous is None or preference(child) < preference(previous):
                    deduplicated[key] = child
        if not deduplicated:
            break
        ranked = sorted(deduplicated.values(), key=preference)
        best_by_count[depth] = ranked[0][0]
        best = min(best, ranked[0], key=preference)
        states_pruned += max(0, len(ranked) - beam_width)
        frontier = ranked[:beam_width]

    chosen = [prepared[index] for index in best[4]]
    diagnostics.update(
        method="grouped_beam_search_with_ungrouped_incumbent", beam_width=beam_width,
        estimated_net=best[0], ungrouped_estimated_net=incumbent_diagnostics["estimated_net"],
        greedy_estimated_net=greedy[0], selected_count=len(chosen),
        selected_merged_campaigns=sum(len(option["cells"]) > 1 for option in chosen),
        selected_source_indices=[list(option["source_indices"]) for option in chosen],
        best_estimated_net_by_count=best_by_count, states_expanded=states_expanded,
        states_pruned=states_pruned, planned_cost=budget - best[1],
        planned_contacts=contacts - best[2])
    plan = [dict(option["campaign"], campaign_name=f"grouped_{i + 1}") for i, option in enumerate(chosen)]
    return plan, diagnostics
