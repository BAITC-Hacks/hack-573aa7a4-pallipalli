"""History priors using observed migrations and the public tariff catalogue.

Migration shares are a ranking proxy, not causal conversion estimates: the
history contains migrations and omits the denominator of all eligible people.
The uncertainty explicitly allows the campaign population to differ.
"""

import math

import numpy as np
import pandas as pd


SHRINK_K0 = 5.0
PRIOR_REL_SD = 1.0
PRIOR_ABS_SD = 0.05
UNSEEN_SD = 0.08
SEGMENTS = ("LOW", "MID", "HIGH")
HISTORY_COLUMNS = (
    "tariff_plan_code_from", "tariff_plan_code_to",
    "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
)


def _codes(series):
    """Keep missing codes missing rather than converting them to 'nan'."""
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def _catalogue(tariffs):
    if not isinstance(tariffs, pd.DataFrame) or "tariff_plan_code" not in tariffs:
        return {}, set()
    codes = _codes(tariffs["tariff_plan_code"])
    known = set(codes.dropna().astype(str))
    if "price_tariff" not in tariffs:
        return {}, known
    prices = pd.to_numeric(tariffs["price_tariff"], errors="coerce")
    valid = codes.notna() & np.isfinite(prices) & (prices >= 0)
    frame = pd.DataFrame({"code": codes[valid], "price": prices[valid]})
    # Duplicate catalogue rows must not make a scalar lookup return a Series.
    return frame.groupby("code")["price"].median().to_dict(), known


def _history_frame(history_path):
    if history_path is None:
        return pd.DataFrame()
    try:
        history = pd.read_csv(history_path)
    except (OSError, ValueError, TypeError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()
    if not set(HISTORY_COLUMNS).issubset(history.columns):
        return pd.DataFrame()
    before = pd.to_numeric(history["AVG_ARPU_PREV_3M"], errors="coerce")
    after = pd.to_numeric(history["AVG_ARPU_NEXT_3M"], errors="coerce")
    source = _codes(history["tariff_plan_code_from"])
    target = _codes(history["tariff_plan_code_to"])
    valid = (np.isfinite(before) & np.isfinite(after) & (before >= 100)
             & source.notna() & target.notna())
    history = pd.DataFrame({"source": source[valid], "target": target[valid]})
    before, after = before[valid], after[valid]
    history["segment"] = np.where(before < 1000, "LOW",
                                  np.where(before <= 5000, "MID", "HIGH"))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lift = (after - before) / before
    history["lift"] = lift.clip(-1, 3)
    # A finite input can still overflow during subtraction; do not let that
    # become an apparently valid clipped observation.
    return history.loc[np.isfinite(lift)].copy()


def history_priors(history_path, tariffs):
    """Return ``(current, segment, target) -> (mean, sd)`` for known tariffs.

    Sparse segment estimates shrink toward the same tariff pair across segments.
    Uncertainty combines observed sampling variation, a distribution-shift term
    proportional to the mean, and an absolute floor. Unobserved transitions get
    a weak, capped price-gap estimate with broad uncertainty. Missing/invalid
    files or absent prices degrade to finite price-based or zero-mean priors.
    """
    prices, known = _catalogue(tariffs)
    history = _history_frame(history_path)
    priors = {}
    share_proxy = 0.1
    if not history.empty:
        known.update(history["source"].astype(str))
        known.update(history["target"].astype(str))
        totals = history.groupby(["source", "segment"], observed=True).size()
        pair_means = history.groupby(["source", "target"], observed=True)["lift"].mean()
        groups = history.groupby(["source", "segment", "target"], observed=True)["lift"]
        global_sd = float(history["lift"].std())
        if not math.isfinite(global_sd):
            global_sd = 1.0
        shares = []
        for (source, segment, target), values in groups:
            n = len(values)
            share = n / float(totals.loc[(source, segment)])
            pair_mean = float(pair_means.loc[(source, target)])
            smoothed_lift = (n * float(values.mean()) + SHRINK_K0 * pair_mean) / (n + SHRINK_K0)
            mean = smoothed_lift * share
            observed_sd = float(values.std()) if n > 2 else global_sd
            if not math.isfinite(observed_sd):
                observed_sd = global_sd
            sampling_variance = (share * observed_sd) ** 2 / n
            sd = math.sqrt(sampling_variance + (PRIOR_REL_SD * mean) ** 2 + PRIOR_ABS_SD ** 2)
            priors[(str(source), str(segment), str(target))] = (mean, sd)
            shares.append(share)
        share_proxy = float(np.median(shares))

    median_price = float(np.median(list(prices.values()))) if prices else 1.0
    for source in sorted(known):
        for target in sorted(known):
            if source == target:
                continue
            mean = 0.0
            if source in prices and target in prices:
                base = max(float(prices[source]), median_price, 1.0)
                gap = max(-1.0, min(1.0, (float(prices[target]) - float(prices[source])) / base))
                mean = 0.5 * gap * share_proxy
            # Price difference alone is not evidence of customer response.
            sd = math.sqrt(UNSEEN_SD ** 2 + (PRIOR_REL_SD * mean) ** 2)
            for segment in SEGMENTS:
                priors.setdefault((source, segment, target), (mean, sd))
    return priors
