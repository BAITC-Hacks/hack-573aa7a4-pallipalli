"""Aggregate public evidence for an optional analyst outside Agent.act.

The analyst receives summaries and noisy pilot feedback. Neither private
environment state nor scored outcomes are read by this module.
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

from agent import Agent
from mock_environment import make_mock_env


ROOT = Path(__file__).resolve().parent
MAX_CATALOG = 40
MAX_CAMPAIGN_CONTACTS = 5000


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _audience_summary(profile):
    cells = []
    for (current, segment), rows in profile.groupby(
            ["current_tariff", "arpu_segment"], observed=True, sort=True, dropna=False):
        values = pd.to_numeric(rows["predicted_arpu"], errors="coerce")
        cells.append({
            "current_tariff": None if pd.isna(current) else str(current),
            "arpu_segment": None if pd.isna(segment) else str(segment),
            "population": len(rows), "predicted_arpu_sum": float(values.sum()),
            "predicted_arpu_mean": float(values.mean()),
            "data_segment_counts": {str(k): int(v) for k, v in
                                    rows["data_segment"].fillna("UNKNOWN").value_counts().sort_index().items()},
            "call_segment_counts": {str(k): int(v) for k, v in
                                    rows["call_segment"].fillna("UNKNOWN").value_counts().sort_index().items()},
        })
    return {"population": len(profile),
            "baseline_predicted_arpu": float(profile["predicted_arpu"].sum()),
            "cells": cells}


def _history_summary():
    path = ROOT / "data" / "change_tariff.csv"
    if not path.exists():
        return {}, 0
    history = pd.read_csv(path, usecols=[
        "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M", "tariff_plan_code_from",
        "tariff_plan_code_to"])
    before = pd.to_numeric(history["AVG_ARPU_PREV_3M"], errors="coerce")
    after = pd.to_numeric(history["AVG_ARPU_NEXT_3M"], errors="coerce")
    valid = (before >= 100) & np.isfinite(before) & np.isfinite(after)
    history = history.loc[valid].copy()
    history["segment"] = np.where(before.loc[valid] < 1000, "LOW",
                                   np.where(before.loc[valid] <= 5000, "MID", "HIGH"))
    history["lift"] = ((after.loc[valid] - before.loc[valid]) /
                        before.loc[valid]).clip(-1, 3)
    keys = ["tariff_plan_code_from", "segment"]
    totals = history.groupby(keys, observed=True).size()
    summary = {}
    for key, rows in history.groupby(keys + ["tariff_plan_code_to"], observed=True):
        support = len(rows)
        share = support / float(totals.loc[key[:2]])
        mean = float(rows["lift"].mean())
        summary[tuple(str(item) for item in key)] = {
            "migration_support": support,
            "source_segment_migrations": int(totals.loc[key[:2]]),
            "migration_share": share, "mean_clipped_relative_arpu_change": mean,
            "shrunk_ranking_prior": mean * share * support / (support + 30.0),
        }
    return summary, len(history)


def _candidate_catalog(audience, tariffs, channels, initial_budget):
    history, valid_rows = _history_summary()
    source_totals = {key[:2]: values["source_segment_migrations"]
                     for key, values in history.items()}
    candidates = []
    tariff_ids = sorted(str(item) for item in tariffs["tariff_plan_code"])
    for cell in audience["cells"]:
        if cell["population"] < 10 or not math.isfinite(cell["predicted_arpu_mean"]):
            continue
        if cell["current_tariff"] not in tariff_ids or cell["arpu_segment"] not in (
                "LOW", "MID", "HIGH"):
            continue
        for target in tariff_ids:
            if target == cell["current_tariff"]:
                continue
            prior = history.get((cell["current_tariff"], cell["arpu_segment"], target), {
                "migration_support": 0, "source_segment_migrations": source_totals.get(
                    (cell["current_tariff"], cell["arpu_segment"]), 0),
                "migration_share": 0.0, "mean_clipped_relative_arpu_change": 0.0,
                "shrunk_ranking_prior": 0.0,
            })
            variants = []
            for channel, settings in sorted(channels.items()):
                cost = float(settings["cost_per_contact"])
                count = min(cell["population"], MAX_CAMPAIGN_CONTACTS)
                if cost:
                    count = min(count, int(initial_budget // cost))
                channel_prior = prior["shrunk_ranking_prior"] * float(
                    settings["conversion_multiplier"])
                variants.append({
                    "filter_current_tariff": cell["current_tariff"],
                    "filter_arpu_segment": cell["arpu_segment"],
                    "target_tariff": target, "channel": channel,
                    "population": cell["population"],
                    "population_mean_predicted_arpu": cell["predicted_arpu_mean"],
                    "history": dict(prior), "channel_adjusted_prior": channel_prior,
                    "historical_ranking_net_proxy": (
                        count * cell["predicted_arpu_mean"] * channel_prior - count * cost),
                    "ranking_contact_count": count,
                })
            # One channel per destination keeps the bounded catalog diverse.
            variants.sort(key=lambda c: (-c["historical_ranking_net_proxy"], c["channel"]))
            candidates.append(variants[0])
    candidates.sort(key=lambda c: (-c["historical_ranking_net_proxy"],
                                  c["filter_current_tariff"], c["filter_arpu_segment"],
                                  c["target_tariff"], c["channel"]))
    return [dict(candidate, id=f"candidate_{i:03d}")
            for i, candidate in enumerate(candidates[:MAX_CATALOG], start=1)], valid_rows


def _deployment_estimate(profile, campaigns, channels, budget, contacts):
    """Cost/contact arithmetic from public filters; no effect prediction."""
    remaining_budget, remaining_contacts = float(budget), int(contacts)
    details = []
    for campaign in campaigns:
        rows = profile
        for column, key in [("arpu_segment", "filter_arpu_segment"),
                            ("data_segment", "filter_data_segment"),
                            ("call_segment", "filter_call_segment")]:
            value = campaign.get(key)
            if value is not None:
                rows = rows[rows[column] == value]
        if campaign.get("filter_current_tariff") is not None:
            wanted = [item.strip() for item in str(
                campaign["filter_current_tariff"]).split(";") if item.strip()]
            rows = rows[rows["current_tariff"].isin(wanted)]
        cost = float(channels[campaign["channel"]]["cost_per_contact"])
        count = min(len(rows), MAX_CAMPAIGN_CONTACTS, remaining_contacts)
        if cost:
            count = min(count, int(remaining_budget // cost))
        count = max(0, int(count))
        spend = count * cost
        remaining_budget -= spend
        remaining_contacts -= count
        details.append({"campaign_name": campaign.get("campaign_name", ""),
                        "matching_population": len(rows),
                        "estimated_contacts": count, "estimated_cost": spend})
    return {"estimated_deployment_cost": float(budget) - remaining_budget,
            "estimated_deployment_contacts": int(contacts) - remaining_contacts,
            "estimated_budget_after_deployment": remaining_budget,
            "estimated_contacts_after_deployment": remaining_contacts,
            "campaigns": details}


def build_evidence(seed=42):
    """Run the unchanged offline agent and return deterministic public evidence.

    The factory's organizer-only second return value is immediately discarded.
    Creating this local evidence runs pilots in a fresh local mock environment;
    it never executes LLM-proposed campaigns or calls an external API.
    """
    env = make_mock_env(seed=seed, data_dir=str(ROOT / "data"),
                        profile_path=str(ROOT / "customer_profile.csv"))[0]
    initial = {"budget": float(env.remaining_budget),
               "contacts": int(env.remaining_contacts), "pilots": int(env.pilots_left),
               "max_final_campaigns": 10, "max_contacts_per_campaign": MAX_CAMPAIGN_CONTACTS}
    audience = _audience_summary(env.customer_profile)
    catalog, history_rows = _candidate_catalog(audience, env.tariffs, env.channels,
                                               initial["budget"])
    pilots = []
    public_run_pilot = env.run_pilot

    def capture_pilot(target_tariff, channel, n_customers=100,
                      filter_arpu_segment=None, filter_data_segment=None,
                      filter_call_segment=None, filter_current_tariff=None):
        request = dict(target_tariff=target_tariff, channel=channel,
                       n_customers=n_customers, filter_arpu_segment=filter_arpu_segment,
                       filter_data_segment=filter_data_segment,
                       filter_call_segment=filter_call_segment,
                       filter_current_tariff=filter_current_tariff)
        result = public_run_pilot(**request)
        pilots.append({"request": dict(request), "result": dict(result)})
        return result

    env.run_pilot = capture_pilot
    agent = Agent()
    campaigns = agent.act(env)
    diagnostics = dict(getattr(agent, "plan_diagnostics", {}))
    deployment = _deployment_estimate(env.customer_profile, campaigns, env.channels,
                                      env.remaining_budget, env.remaining_contacts)
    estimate_applies = (diagnostics.get("selected_count") == len(campaigns)
                       and not any(c.get("campaign_name") == "minimum_exposure_fallback"
                                   for c in campaigns))
    evidence = {
        "schema_version": 1, "seed": int(seed),
        "scope": "Local mock run; public data and noisy pilot observations only.",
        "initial_limits": initial, "audience_summary": audience,
        "tariffs": env.tariffs.sort_values("tariff_plan_code").to_dict(orient="records"),
        "channels": {key: dict(value) for key, value in sorted(env.channels.items())},
        "pilots": pilots, "final_campaigns": campaigns, "plan_diagnostics": diagnostics,
        "plan_estimate_applies_to_returned_plan": estimate_applies,
        "resources": {
            "budget_after_pilots": float(env.remaining_budget),
            "contacts_after_pilots": int(env.remaining_contacts),
            "pilots_left": int(env.pilots_left),
            "pilot_cost": initial["budget"] - float(env.remaining_budget),
            "pilot_contacts": initial["contacts"] - int(env.remaining_contacts),
            **deployment,
        },
        "candidate_catalog": catalog, "historical_valid_migration_rows": history_rows,
        "limitations": [
            "Local mock evidence is not the hidden judging score.",
            "Pilot effects are noisy observations, not true model effects.",
            "Historical migration shares are ranking proxies, not causal conversion probabilities.",
            "Catalog ranking uses population-average ARPU; it is not a scored campaign result.",
            "Deployment cost and contacts are estimates from public filters and remaining limits.",
            "Planner estimated_net is a conservative incremental final-plan estimate; it excludes pilot net effects.",
            "No actual scored net gain, true effects, or subscriber records are included.",
            "Catalog hypotheses are suggestions only and do not change or execute the submission.",
        ],
    }
    return _json_safe(evidence)


def render_report(evidence, commentary=None):
    """Render authoritative numbers in Python; keep optional commentary separate."""
    resources = evidence["resources"]
    initial = evidence["initial_limits"]
    estimated_net = evidence["plan_diagnostics"].get("estimated_net")
    number = lambda value: f"{value:,.2f}" if value is not None else "Unavailable"
    rows = [
        ("Audience", f"{evidence['audience_summary']['population']:,}"),
        ("Pilots completed", str(len(evidence["pilots"]))),
        ("Final campaigns", str(len(evidence["final_campaigns"]))),
        ("Initial budget", number(initial["budget"])),
        ("Pilot cost (observed)", number(resources["pilot_cost"])),
        ("Budget after pilots (observed)", number(resources["budget_after_pilots"])),
        ("Deployment cost (estimated)", number(resources["estimated_deployment_cost"])),
        ("Budget after deployment (estimated)", number(resources["estimated_budget_after_deployment"])),
        ("Pilot contacts (observed)", f"{resources['pilot_contacts']:,}"),
        ("Contacts after pilots (observed)", f"{resources['contacts_after_pilots']:,}"),
        ("Deployment contacts (estimated)", f"{resources['estimated_deployment_contacts']:,}"),
        ("Conservative final-plan net (estimated, excludes pilots)",
         number(estimated_net) if evidence["plan_estimate_applies_to_returned_plan"] else "Unavailable"),
        ("Actual scored net gain", "Not calculated by this analyst"),
    ]
    lines = ["# Campaign analyst report", "", f"Local mock seed: **{evidence['seed']}**.",
             "", "## Authoritative run summary", "",
             "Numbers below are rendered by Python from public evidence. Estimated deployment",
             "spending is separate from observed pilot spending; no scorer result is reported.",
             "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    lines += ["", "## Selected campaigns", "",
              "| Campaign | Current tariff | ARPU segment | Target | Channel | Estimated contacts | Estimated cost |",
              "|---|---|---|---|---|---:|---:|"]
    for campaign, deployment in zip(evidence["final_campaigns"], resources["campaigns"]):
        fields = [campaign.get("campaign_name", ""), campaign.get("filter_current_tariff", "All"),
                  campaign.get("filter_arpu_segment", "All"), campaign["target_tariff"],
                  campaign["channel"], f"{deployment['estimated_contacts']:,}",
                  number(deployment["estimated_cost"])]
        lines.append("| " + " | ".join(str(item).replace("|", "\\|") for item in fields) + " |")
    lines += ["", "## Evidence limitations", ""]
    lines += [f"- {item}" for item in evidence["limitations"]]
    if commentary:
        lines += ["", "## Optional LLM commentary", "",
                  "Generated interpretation; the authoritative numerical summary above takes precedence.",
                  "Hypotheses are proposals only and have not changed or executed the agent's plan.",
                  ""]
        if isinstance(commentary, dict):
            lines += [str(commentary.get("summary", "")).strip(), "", "### Observations", ""]
            lines += [f"- {item}" for item in commentary.get("observations", [])]
            lines += ["", "### Suggested hypotheses (not executed)", ""]
            for hypothesis in commentary.get("hypotheses", []):
                lines += [f"- **{hypothesis['candidate_id']}**: {hypothesis['rationale']}",
                          f"  Evidence to collect: {hypothesis['evidence_to_collect']}"]
            lines += ["", "### Interpretation limitations", ""]
            lines += [f"- {item}" for item in commentary.get("limitations", [])]
        else:
            lines.append(str(commentary).strip())
    else:
        lines += ["", "## Analyst status", "",
                  "Deterministic local report. No OpenAI commentary was provided."]
    return "\n".join(lines) + "\n"
