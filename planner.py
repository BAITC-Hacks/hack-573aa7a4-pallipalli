"""Deterministic beam search over variable-length, ordered campaign plans.

Only public candidate estimates enter this module. Beam pruning bounds runtime;
this is not a proof of global optimality. The greedy plan is kept as an incumbent.
"""
import heapq
import numpy as np


def _gain_curve(values, lower, pilot_contacts, cost):
    """Conservative incremental gain for each scorer-compatible ID prefix."""
    values = np.asarray(values, dtype=float)[:5000]
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    overlap = min(len(values), max(0, int(pilot_contacts)))
    if overlap:
        largest, largest_sum = [], 0.0
        for n, value in enumerate(values, 1):
            if len(largest) < overlap:
                heapq.heappush(largest, value)
                largest_sum += value
            elif value > largest[0]:
                largest_sum += value - heapq.heapreplace(largest, value)
            # The pilot identities are private; discount the largest possible
            # duplicated ARPU rather than pretend pilot contacts are all new.
            prefix[n] = max(0.0, prefix[n] - largest_sum)
    return prefix * lower - np.arange(len(prefix)) * cost


def select_plan(candidates, budget, contacts, max_campaigns=10, beam_width=256):
    """Return chosen candidate indices and search diagnostics.

    Each candidate has: cell, values (in ID order), lower, pilot_contacts, cost.
    Exactly the same monetary/contact truncation is used as in the public scorer.
    Final campaigns cannot share a cell. Plans may stop at any size up to ten.
    """
    limit = max(0, min(10, int(max_campaigns)))
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    budget, contacts = max(0.0, float(budget)), max(0, int(contacts))
    cells = {cell: i for i, cell in enumerate(dict.fromkeys(c["cell"] for c in candidates))}
    prepared = []
    for i, c in enumerate(candidates):
        if c["lower"] <= 0 or not len(c["values"]):
            continue
        curve = _gain_curve(c["values"], c["lower"], c["pilot_contacts"], c["cost"])
        prepared.append((i, 1 << cells[c["cell"]], float(c["cost"]), curve))

    # State: estimated gain, money left, contacts left, used-cell mask, indices.
    initial = (0.0, budget, contacts, 0, ())

    def extend(state, option):
        score, money, remaining, mask, path = state
        index, cell_bit, cost, curve = option
        if mask & cell_bit:
            return None
        n = min(len(curve) - 1, remaining)
        if cost > 0:
            n = min(n, int(money // cost))
        if n <= 0 or curve[n] <= 0:
            return None
        return (score + float(curve[n]), money - n * cost, remaining - n,
                mask | cell_bit, path + (index,))

    def preference(state):
        # Prefer higher gain, fewer campaigns, then more remaining resources.
        return (-state[0], len(state[4]), -state[1], -state[2], state[4])

    greedy = initial
    while len(greedy[4]) < limit:
        choices = [s for c in prepared if (s := extend(greedy, c)) is not None]
        if not choices:
            break
        greedy = min(choices, key=preference)

    best = greedy
    frontier = [initial]
    best_by_size = {0: 0.0}
    states_expanded, states_pruned = 0, 0
    for depth in range(1, limit + 1):
        deduplicated = {}
        for state in frontier:
            for option in prepared:
                child = extend(state, option)
                if child is None:
                    continue
                states_expanded += 1
                # These states have the same remaining options and resources,
                # even if their selected destination/channel choices differ.
                key = (child[3], child[1], child[2])
                previous = deduplicated.get(key)
                if previous is None or preference(child) < preference(previous):
                    deduplicated[key] = child
        if not deduplicated:
            break
        ranked = sorted(deduplicated.values(), key=preference)
        best_by_size[depth] = ranked[0][0]
        if preference(ranked[0]) < preference(best):
            best = ranked[0]
        states_pruned += max(0, len(ranked) - beam_width)
        frontier = ranked[:beam_width]

    diagnostics = {
        "method": "beam_search_with_greedy_incumbent",
        "beam_width": beam_width, "estimated_net": best[0],
        "greedy_estimated_net": greedy[0], "selected_count": len(best[4]),
        "best_estimated_net_by_count": best_by_size,
        "states_expanded": states_expanded, "states_pruned": states_pruned,
        "planned_cost": budget - best[1], "planned_contacts": contacts - best[2],
    }
    return list(best[4]), diagnostics
