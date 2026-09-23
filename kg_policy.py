"""Choose pilots by their expected value for the deployment decision.

This is a cellwise knowledge-gradient approximation, adapted from the supplied
branch.  The score values information against competing tariffs in that cell;
it does not solve the complete constrained campaign plan inside every probe.
Only public environment attributes and the caller's posterior are used.
"""
import math


def expected_excess(mean, sd, threshold):
    """E[(X - threshold)+] for a normal X, including a deterministic X."""
    if sd <= 1e-12:
        return max(mean - threshold, 0.0)
    z = (mean - threshold) / sd
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, (mean - threshold) * cdf + sd * pdf)


def information_value(mean, sd, n, alternative, stake, probe_multiplier,
                      deployment_multiplier, noise=0.804, kappa=1.0):
    """Expected improvement of the best conservative cell response.

    A prospective observation changes the posterior mean with variance equal
    to the reduction in posterior variance.  Probe and deployment multipliers
    are separate: a free push observation may inform a later SMS campaign.
    """
    if sd <= 1e-12 or n <= 0 or probe_multiplier <= 0 or stake <= 0:
        return 0.0
    variance = sd * sd
    new_variance = 1.0 / (1.0 / variance + n * probe_multiplier ** 2 / noise ** 2)
    new_sd = math.sqrt(new_variance)
    mean_sd = math.sqrt(max(0.0, variance - new_variance))
    threshold = max(0.0, alternative)
    before = max(0.0, mean - kappa * sd - threshold)
    after = expected_excess(mean - kappa * new_sd, mean_sd, threshold)
    return max(0.0, stake * deployment_multiplier * (after - before))


def _channels(env):
    """Use public SMS/push parameters without assuming the published prices."""
    channels = []
    for name in ("sms", "push"):
        settings = env.channels.get(name)
        if settings is None:
            continue
        multiplier = float(settings["conversion_multiplier"])
        cost = float(settings["cost_per_contact"])
        if math.isfinite(multiplier) and 0 < multiplier <= 1 and math.isfinite(cost) and cost >= 0:
            channels.append((name, multiplier, cost))
    return channels


def explore(agent, env, candidates):
    """Mutate candidate observations and return inspectable pilot diagnostics.

    Required caller state: ``_estimate(c) -> (mean, sd, lower)``, ``NOISE``,
    ``reserve_money``, and ``pilot_contacts_left``.  Candidate dictionaries use
    the frozen agent's cell/values/population/campaign/observations schema.

    Optional settings are KG_KAPPA (1), KG_CONTACT_COST (30), KG_PILOT_SIZES
    ((40, 80, 120, 200)), and KG_MAX_CELL_SHARE (0.5).  The contact cost is the
    reference branch's heuristic opportunity cost, not an organizer parameter.
    Kappa explicitly controls the conservative decision used by this policy.
    A first feasible pilot is mandatory even when its estimated value is low.
    """
    kappa = max(0.0, float(getattr(agent, "KG_KAPPA", 1.0)))
    contact_cost = max(0.0, float(getattr(agent, "KG_CONTACT_COST", 30.0)))
    noise = float(getattr(agent, "NOISE", 0.804))
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("Pilot noise must be finite and positive")
    sizes = tuple(sorted({int(n) for n in getattr(agent, "KG_PILOT_SIZES", (40, 80, 120, 200))
                          if 10 <= int(n) <= 200}))
    if not sizes:
        raise ValueError("At least one legal pilot size is required")
    cell_share = min(1.0, max(0.0, float(getattr(agent, "KG_MAX_CELL_SHARE", 0.5))))
    channels = _channels(env)
    diagnostics = {
        "policy": "cellwise_knowledge_gradient", "kappa": kappa,
        "contact_opportunity_cost": contact_cost, "pilot_sizes": list(sizes),
        "max_cell_share": cell_share, "pilots": [], "pilot_contacts": 0,
        "pilot_cost": 0.0, "stop_reason": None,
        "value_approximation": "best_conservative_tariff_per_cell",
    }
    agent.exploration_diagnostics = diagnostics
    if not channels:
        diagnostics["stop_reason"] = "no_supported_probe_channel"
        return diagnostics
    deployment_multiplier = max(mult for _, mult, _ in channels)
    by_cell = {}
    for candidate in candidates:
        by_cell.setdefault(candidate["cell"], []).append(candidate)
    valid_pilots = sum(bool(c["observations"]) for c in candidates)

    # This bound also protects against a malformed environment that forgets to
    # decrement its public pilot counter.
    for _ in range(max(0, int(env.pilots_left))):
        if env.pilots_left <= 0:
            diagnostics["stop_reason"] = "pilot_limit"
            break
        contact_cap = min(int(env.remaining_contacts), int(agent.pilot_contacts_left))
        if contact_cap < 10:
            diagnostics["stop_reason"] = "pilot_contact_limit"
            break
        estimates = {id(c): agent._estimate(c)[:2] for c in candidates}
        allowance = max(0.0, float(env.remaining_budget) - float(agent.reserve_money))
        mandatory = valid_pilots == 0

        def choose(cash, permit_minimum, minimum_only=False):
            best = None
            for c in candidates:
                mean, sd = estimates[id(c)]
                if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
                    continue
                alternatives = [m - kappa * s for other in by_cell[c["cell"]] if other is not c
                                for m, s in [estimates[id(other)]]
                                if math.isfinite(m) and math.isfinite(s) and s >= 0]
                alternative = max(alternatives, default=0.0)
                population = max(0, int(c["population"]))
                max_cell_n = int(population * cell_share)
                if permit_minimum and population >= 10:
                    max_cell_n = max(10, max_cell_n)
                count = min(len(c["values"]), 5000, max(0, int(env.remaining_contacts)))
                stake = float(sum(c["values"][:count]))
                if not math.isfinite(stake) or stake < 0:
                    continue
                for channel, multiplier, cost in channels:
                    cap = min(max(sizes), population, max_cell_n, contact_cap)
                    if cost > 0:
                        cap = min(cap, int(cash // cost))
                    if minimum_only:
                        feasible_sizes = {10} if cap >= 10 else set()
                    else:
                        feasible_sizes = {n for n in sizes if n <= cap}
                        if cap >= min(sizes):
                            feasible_sizes.add(cap)
                        elif permit_minimum and cap >= 10:
                            feasible_sizes.add(cap)
                    for n in sorted(feasible_sizes):
                        value = information_value(mean, sd, n, alternative, stake,
                                                  multiplier, deployment_multiplier, noise, kappa)
                        score = value - n * (cost + contact_cost)
                        option = (score, value, c, channel, multiplier, cost, n)
                        if best is None or score > best[0]:
                            best = option
            return best

        choice = choose(allowance, mandatory)
        reserve_override = False
        if choice is None and mandatory:
            # Pilots are a required part of the interface.  If the cash reserve
            # prevents every legal first probe, spend the smallest feasible
            # first probe from the public remaining budget instead.
            choice = choose(max(0.0, float(env.remaining_budget)), True, minimum_only=True)
            reserve_override = choice is not None
        if choice is None:
            diagnostics["stop_reason"] = "no_feasible_pilot"
            break
        score, value, candidate, channel, multiplier, cost, n = choice
        if score <= 0 and not mandatory:
            diagnostics["stop_reason"] = "information_value_below_cost"
            diagnostics["next_pilot_score"] = score
            break
        mean_before, sd_before = estimates[id(candidate)]
        try:
            result = env.run_pilot(n_customers=n, channel=channel, **candidate["campaign"])
        except (RuntimeError, ValueError) as error:
            diagnostics["stop_reason"] = "pilot_error"
            diagnostics["pilot_error"] = str(error)
            break
        actual = max(0, int(result["n_customers"]))
        observed = float(result["observed_lift_ratio"])
        agent.pilot_contacts_left = max(0, int(agent.pilot_contacts_left) - actual)
        diagnostics["pilot_contacts"] += actual
        diagnostics["pilot_cost"] += actual * cost
        record = {
            "cell": candidate["cell"], "target": candidate["campaign"]["target_tariff"],
            "channel": channel, "requested": n, "actual": actual,
            "mean_before": mean_before, "sd_before": sd_before,
            "expected_information_value": value, "acquisition_score": score,
            "mandatory": mandatory, "reserve_override": reserve_override,
            "observed_ratio": observed if math.isfinite(observed) else None,
        }
        diagnostics["pilots"].append(record)
        if actual <= 0 or not math.isfinite(observed):
            diagnostics["stop_reason"] = "invalid_pilot_result"
            break
        candidate["observations"].append((observed, actual, multiplier))
        valid_pilots += 1
        mean_after, sd_after, _ = agent._estimate(candidate)
        record.update(mean_after=mean_after, sd_after=sd_after)
    if diagnostics["stop_reason"] is None:
        diagnostics["stop_reason"] = "pilot_limit"
    return diagnostics
