"""
Агент тарифных кампаний v2: разведка, которая сама решает, насколько верить истории.

Отличия от agent.py:

1. ДОВЕРИЕ К ИСТОРИИ УЧИТСЯ. Вместо зашитой погрешности приора агент считает,
   что истинный эффект ≈ β·h + b ± τ ± ρ·|h| (h — оценка по истории), и после
   каждого пилота уточняет β, b, τ, ρ по всем пилотам сразу (байесовская
   сетка). τ — общий разброс, ρ — ошибка отдельного перехода, растущая с его
   силой (например, перевёрнутый знак). История
   подтвердилась — β≈1, τ мал, агент действует узко. История врёт — β→0, τ
   растёт, приор всех гипотез расширяется и разведка сама становится широкой.

2. КАНДИДАТЫ — ВСЕ ТАРИФЫ. Если история врёт, лучший тариф может быть любым,
   поэтому гипотезы заранее не отсекаем: выбор пилота делает knowledge gradient.

3. ПИЛОТ — РЕАЛЬНЫЕ ЛЮДИ. В цену пилота входит ожидаемый ущерб его абонентам,
   если эффект окажется отрицательным. В «плохом» мире разведка сворачивается.

4. ОСТАТОК ОХВАТА — В PUSH. Бесплатные контакты не пропадают: их получают
   группы с положительной оценкой, не вошедшие в основной план.
"""

import math
import os

import numpy as np
import pandas as pd

# Шум пилота: std наблюдаемого эффекта ≈ PILOT_NOISE / sqrt(n).
PILOT_NOISE = 0.8
PILOT_CHANNEL = "sms"
PILOT_SIZES = (40, 80, 120, 200)
PILOT_CONTACT_SHARE = 0.25     # не больше четверти общего охвата уходит на разведку
PILOT_OPPORTUNITY_COST = 30    # цена «сожжённого» контакта при выборе размера пилота
MIN_CELL_SIZE = 60             # ячейки меньше не пилотируем и не таргетируем

SHRINK_K0 = 5                  # сглаживание малых групп истории к среднему по паре тарифов
UNSEEN_SD = 0.08               # разброс ценовой прикидки для переходов без истории

# Насколько история похожа на правду: эффект = β·h + b + N(0, τ² + ρ²·h²).
# Сетка гиперпараметров и их априорное распределение.
BETA_GRID = np.linspace(-0.5, 1.5, 21)
BETA_PRIOR = (0.8, 0.5)        # по умолчанию история скорее верна, но с запасом
BIAS_GRID = np.linspace(-0.15, 0.10, 26)
BIAS_PRIOR_SD = 0.03
TAU_GRID = np.geomspace(0.005, 0.25, 16)   # равномерно по log τ
RHO_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])  # относительная ошибка отдельного перехода

KAPPA = 1.0                    # запускаем кампанию, если mean − κ·σ > 0
KAPPA_PUSH = 0.5               # бесплатный push на остаток охвата — порог мягче
MAX_CAMPAIGNS = 10
MAX_PER_CAMPAIGN = 5000

ARPU_BINS = [-math.inf, 1000, 5000, math.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]

_B, _BIAS, _TAU, _RHO = np.meshgrid(BETA_GRID, BIAS_GRID, TAU_GRID, RHO_GRID, indexing="ij")
_LOG_PRIOR = (-0.5 * ((_B - BETA_PRIOR[0]) / BETA_PRIOR[1]) ** 2
              - 0.5 * (_BIAS / BIAS_PRIOR_SD) ** 2)


def _norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _expected_excess(mu, sigma, c):
    """E[max(X, c)] − c для X ~ N(mu, sigma²)."""
    if sigma <= 1e-12:
        return max(mu - c, 0.0)
    z = (mu - c) / sigma
    return sigma * (z * _norm_cdf(z) + _norm_pdf(z))


class Agent:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------ act
    def act(self, env):
        self.env = env
        self.channels = env.channels
        cells = self._build_cells(env.customer_profile)
        prior = self._history_prior()
        cands = self._build_candidates(cells, prior, env.tariffs)
        self._refresh(cands)
        self.log(f"[agent] ячеек: {len(cells)}, гипотез для разведки: {len(cands)}")

        self._explore(env, cells, cands)
        campaigns = self._plan(env, cells, cands)
        return campaigns

    # ---------------------------------------------------------------- cells
    def _build_cells(self, profile):
        cells = {}
        grouped = profile.sort_values("ID_NUMBER").groupby(["current_tariff", "arpu_segment"], observed=True)
        for (tariff, seg), g in grouped:
            n = len(g)
            if n < MIN_CELL_SIZE:
                continue
            capped = g.iloc[:MAX_PER_CAMPAIGN]
            cells[(tariff, seg)] = {
                "tariff": tariff, "seg": seg, "n": n,
                "arpu_mean": float(capped["predicted_arpu"].mean()),
                "stake": float(capped["predicted_arpu"].sum()),
            }
        return cells

    # ---------------------------------------------------------------- prior
    def _history_prior(self):
        """(from, to, seg) -> (h, v_h): оценка эффекта по истории и её выборочная дисперсия."""
        paths = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "change_tariff.csv"),
                 os.path.join("data", "change_tariff.csv")]
        df = None
        for p in paths:
            try:
                df = pd.read_csv(p)
                break
            except Exception:
                continue
        if df is None:
            self.log("[agent] история не найдена — работаем только на ценовом приоре")
            return {}

        df = df[df["AVG_ARPU_PREV_3M"] >= 100].copy()
        df["seg"] = pd.cut(df["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS).astype(str)
        df["pct"] = ((df["AVG_ARPU_NEXT_3M"] - df["AVG_ARPU_PREV_3M"]) / df["AVG_ARPU_PREV_3M"]).clip(-1, 3)
        pct_sd_all = float(df["pct"].std())

        g = (df.groupby(["tariff_plan_code_from", "tariff_plan_code_to", "seg"])["pct"]
             .agg(["mean", "std", "size"]).reset_index())
        totals = g.groupby(["tariff_plan_code_from", "seg"])["size"].sum().rename("total").reset_index()
        pair = (df.groupby(["tariff_plan_code_from", "tariff_plan_code_to"])["pct"].mean()
                .rename("pair_mean").reset_index())
        g = g.merge(totals, on=["tariff_plan_code_from", "seg"]).merge(
            pair, on=["tariff_plan_code_from", "tariff_plan_code_to"])

        prior = {}
        for r in g.itertuples(index=False):
            k = r.size
            conv = k / r.total
            pct = (k * r.mean + SHRINK_K0 * r.pair_mean) / (k + SHRINK_K0)
            pct_sd = r.std if k > 2 and not math.isnan(r.std) else pct_sd_all
            prior[(r.tariff_plan_code_from, r.tariff_plan_code_to, r.seg)] = (pct * conv, (conv * pct_sd) ** 2 / k)
        self._median_conv = float((g["size"] / g["total"]).median())
        return prior

    def _price_prior(self, tariffs, t_from, t_to):
        """Для переходов без истории: знак по разнице цен, эффект небольшой, неопределённость широкая."""
        price = tariffs.set_index("tariff_plan_code")["price_tariff"]
        conv = getattr(self, "_median_conv", 0.1)
        if t_from not in price.index or t_to not in price.index:
            return 0.0, UNSEEN_SD ** 2
        base = max(float(price[t_from]), float(price.median()), 1.0)
        pct = max(min((float(price[t_to]) - float(price[t_from])) / base, 1.0), -1.0) * 0.5
        return pct * conv, UNSEEN_SD ** 2

    def _build_candidates(self, cells, prior, tariffs):
        cands = []
        for key, cell in cells.items():
            for t in tariffs["tariff_plan_code"]:
                if t == cell["tariff"]:
                    continue
                h, vh = prior.get((cell["tariff"], t, cell["seg"])) or self._price_prior(tariffs, cell["tariff"], t)
                cands.append({"cell": key, "target": t, "h": h, "vh": vh, "ys": [], "vs": [],
                              "m": h, "s": math.sqrt(vh), "n_pilots": 0, "pilot_n": 0})
        return cands

    def _refresh(self, cands):
        """
        Пересчитывает оценки всех гипотез.

        1) По всем проведённым пилотам уточняем, насколько история похожа на
           правду (β, b, τ, ρ) — апостериорное распределение на сетке.
        2) Приор каждой гипотезы = β·h + b, разброс = v_h + τ² + ρ²·h² + неопределённость β и b.
        3) Если гипотезу пилотировали, объединяем приор с её замерами по весам точности.
        """
        seen = [c for c in cands if c["vs"]]
        log_w = _LOG_PRIOR
        if seen:
            h = np.array([c["h"] for c in seen])
            vh = np.array([c["vh"] for c in seen])
            vo = np.array([1.0 / sum(1.0 / v for v in c["vs"]) for c in seen])
            y = np.array([o * sum(yi / v for yi, v in zip(c["ys"], c["vs"])) for c, o in zip(seen, vo)])
            mean = _B[..., None] * h + _BIAS[..., None]
            var = vh + _TAU[..., None] ** 2 + _RHO[..., None] ** 2 * h ** 2 + vo
            log_w = log_w + (-0.5 * np.log(var) - 0.5 * (y - mean) ** 2 / var).sum(axis=-1)
        w = np.exp(log_w - log_w.max())
        w /= w.sum()

        eb, ebias = float((w * _B).sum()), float((w * _BIAS).sum())
        vb = float((w * (_B - eb) ** 2).sum())
        vbias = float((w * (_BIAS - ebias) ** 2).sum())
        cov = float((w * (_B - eb) * (_BIAS - ebias)).sum())
        et2 = float((w * _TAU ** 2).sum())
        er2 = float((w * _RHO ** 2).sum())
        self.trust = (eb, ebias, math.sqrt(et2), math.sqrt(er2))

        for c in cands:
            mu = eb * c["h"] + ebias
            var = max(c["vh"] + et2 + (er2 + vb) * c["h"] ** 2 + vbias + 2 * cov * c["h"], 1e-8)
            if c["vs"]:
                prec = 1.0 / var + sum(1.0 / v for v in c["vs"])
                c["m"] = (mu / var + sum(yi / v for yi, v in zip(c["ys"], c["vs"]))) / prec
                c["s"] = math.sqrt(1.0 / prec)
            else:
                c["m"], c["s"] = mu, math.sqrt(var)

    # ------------------------------------------------------------ explore
    def _lb(self, c):
        return c["m"] - KAPPA * c["s"]

    def _pilot_value(self, cand, cell, n, others_best, mult):
        """Ожидаемый прирост финального плана от пилота размера n (knowledge gradient)."""
        s_obs = PILOT_NOISE / math.sqrt(n) / mult
        s_new = 1.0 / math.sqrt(1.0 / cand["s"] ** 2 + 1.0 / s_obs ** 2)
        sig_t = math.sqrt(max(cand["s"] ** 2 - s_new ** 2, 0.0))
        c = max(others_best, 0.0)
        before = max(self._lb(cand), c) - c
        after = _expected_excess(cand["m"] - KAPPA * s_new, sig_t, c)
        return cell["stake"] * mult * (after - before)

    def _pilot_downside(self, cand, cell, n, mult):
        """Ожидаемый ущерб абонентам пилота: они реальные, и отрицательный эффект идёт в счёт."""
        e_neg = cand["m"] - _expected_excess(cand["m"], cand["s"], 0.0)   # E[min(θ, 0)] ≤ 0
        return n * cell["arpu_mean"] * mult * e_neg

    def _explore(self, env, cells, cands):
        if PILOT_CHANNEL not in self.channels:
            return
        mult = self.channels[PILOT_CHANNEL]["conversion_multiplier"]
        cost = self.channels[PILOT_CHANNEL]["cost_per_contact"]
        pilot_contact_cap = int(PILOT_CONTACT_SHARE * env.remaining_contacts)
        used = 0

        by_cell = {}
        for c in cands:
            by_cell.setdefault(c["cell"], []).append(c)

        while env.pilots_left > 0:
            best = None
            for key, group in by_cell.items():
                cell = cells[key]
                lbs = sorted((self._lb(o) for o in group), reverse=True)
                n_max = min(max(PILOT_SIZES), cell["n"] // 2, pilot_contact_cap - used)
                if cost > 0:
                    n_max = min(n_max, int(env.remaining_budget // cost))
                sizes = sorted({s for s in PILOT_SIZES if s <= n_max} | ({n_max} if n_max >= min(PILOT_SIZES) else set()))
                if not sizes:
                    continue
                for c in group:
                    lb = self._lb(c)
                    others_best = lbs[1] if lb == lbs[0] and len(lbs) > 1 else lbs[0]
                    for n in sizes:
                        score = (self._pilot_value(c, cell, n, others_best, mult)
                                 + self._pilot_downside(c, cell, n, mult)
                                 - n * (cost + PILOT_OPPORTUNITY_COST))
                        if best is None or score > best[0]:
                            best = (score, c, n)
            if best is None or best[0] <= 0:
                break

            _, c, n = best
            cell = cells[c["cell"]]
            try:
                res = env.run_pilot(target_tariff=c["target"], channel=PILOT_CHANNEL, n_customers=n,
                                    filter_arpu_segment=cell["seg"], filter_current_tariff=cell["tariff"])
            except (RuntimeError, ValueError) as e:
                self.log(f"[agent] пилот не запущен: {e}")
                break

            n_real = res["n_customers"]
            used += n_real
            y = res["observed_lift_ratio"] / mult
            m_old = c["m"]
            c["ys"].append(y)
            c["vs"].append((PILOT_NOISE / math.sqrt(n_real) / mult) ** 2)
            c["n_pilots"] += 1
            c["pilot_n"] += n_real
            self._refresh(cands)
            beta, bias, tau, rho = self.trust
            self.log(f"[agent] пилот {cell['tariff']}/{cell['seg']} → {c['target']} n={n_real}: "
                     f"наблюдали {y:+.3f}, оценка {m_old:+.3f} → {c['m']:+.3f} ± {c['s']:.3f}; "
                     f"доверие к истории β={beta:.2f} b={bias:+.3f} τ={tau:.3f} ρ={rho:.2f}")

    # --------------------------------------------------------------- plan
    def _plan(self, env, cells, cands):
        channels = self.channels
        by_cost = sorted(channels, key=lambda ch: channels[ch]["cost_per_contact"])
        base_channels = by_cost[:2]  # дешёвые каналы для массового охвата

        # лучший целевой тариф для каждой ячейки по консервативной оценке
        best = {}
        for c in cands:
            if c["cell"] not in best or self._lb(c) > self._lb(best[c["cell"]]):
                best[c["cell"]] = c

        units = []
        for key, c in best.items():
            lb = self._lb(c)
            if lb <= 0:
                continue
            cell = cells[key]
            n = min(cell["n"], MAX_PER_CAMPAIGN)
            value = {ch: cell["arpu_mean"] * lb * channels[ch]["conversion_multiplier"]
                     - channels[ch]["cost_per_contact"] for ch in channels}
            base = max(base_channels, key=lambda ch: value[ch])
            if value[base] <= 0:
                continue
            units.append({"cell": cell, "cand": c, "n": n, "value": value, "channel": base})

        # охват: сначала самые ценные на контакт
        units.sort(key=lambda u: u["value"][u["channel"]], reverse=True)
        contacts_left = env.remaining_contacts
        money_left = env.remaining_budget
        chosen = []
        for u in units:
            cost = channels[u["channel"]]["cost_per_contact"]
            take = min(u["n"], contacts_left)
            if cost > 0:
                take = min(take, int(money_left // cost))
            if take <= 0:
                continue
            u["take"] = take
            contacts_left -= take
            money_left -= take * cost
            chosen.append(u)

        # апгрейд канала на оставшиеся деньги: лучший прирост на единицу доп. затрат
        upgrades = []
        for u in chosen:
            if u["take"] < u["n"]:
                continue  # частично охваченную ячейку не трогаем
            for ch in channels:
                extra = u["n"] * (channels[ch]["cost_per_contact"] - channels[u["channel"]]["cost_per_contact"])
                gain = u["n"] * (u["value"][ch] - u["value"][u["channel"]])
                if extra > 0 and gain > 0:
                    upgrades.append((gain / extra, gain, extra, u, ch))
        upgrades.sort(key=lambda x: x[0], reverse=True)
        upgraded = set()
        for _, gain, extra, u, ch in upgrades:
            if id(u) in upgraded or extra > money_left:
                continue
            u["channel"] = ch
            money_left -= extra
            upgraded.add(id(u))

        # остаток охвата — бесплатным каналом в ячейки, не вошедшие в план
        free = by_cost[0]
        if channels[free]["cost_per_contact"] == 0 and contacts_left > 0:
            taken = {(u["cell"]["tariff"], u["cell"]["seg"]) for u in chosen}
            fill = []
            for key in cells:
                if key in taken:
                    continue
                c = max((x for x in cands if x["cell"] == key), key=lambda x: x["m"] - KAPPA_PUSH * x["s"])
                est = c["m"] - KAPPA_PUSH * c["s"]
                if est <= 0:
                    continue
                cell = cells[key]
                value = {ch: cell["arpu_mean"] * est * channels[ch]["conversion_multiplier"]
                         - channels[ch]["cost_per_contact"] for ch in channels}
                fill.append({"cell": cell, "cand": c, "n": min(cell["n"], MAX_PER_CAMPAIGN),
                             "value": value, "channel": free})
            fill.sort(key=lambda u: u["value"][free], reverse=True)
            for u in fill:
                take = min(u["n"], contacts_left)
                if take <= 0:
                    break
                u["take"] = take
                contacts_left -= take
                chosen.append(u)

        # объединяем ячейки в кампании: один тариф + канал + ARPU-сегмент
        groups = {}
        for u in chosen:
            key = (u["cand"]["target"], u["channel"], u["cell"]["seg"])
            bucket = groups.setdefault(key, [[]])
            if sum(x["take"] for x in bucket[-1]) + u["take"] > MAX_PER_CAMPAIGN:
                bucket.append([])
            bucket[-1].append(u)

        campaigns = []
        for (target, ch, seg), buckets in groups.items():
            for members in buckets:
                total_value = sum(m["take"] * m["value"][ch] for m in members)
                n = sum(m["take"] for m in members)
                campaigns.append((total_value / n, total_value, {
                    "campaign_name": f"{seg.lower()}_{'_'.join(m['cell']['tariff'] for m in members)}_to_{target}_{ch}",
                    "filter_arpu_segment": seg,
                    "filter_current_tariff": ";".join(m["cell"]["tariff"] for m in members),
                    "target_tariff": target,
                    "channel": ch,
                }))
        # сначала самые ценные на контакт: лимиты охвата и бюджета режут хвост
        campaigns.sort(key=lambda x: x[0], reverse=True)
        if len(campaigns) > MAX_CAMPAIGNS:
            campaigns = sorted(campaigns, key=lambda x: x[1], reverse=True)[:MAX_CAMPAIGNS]
            campaigns.sort(key=lambda x: x[0], reverse=True)

        result = [c for _, _, c in campaigns]
        if not result:
            result = [self._fallback_campaign(cells, cands)]

        for per_contact, total, c in campaigns:
            self.log(f"[agent] кампания {c['campaign_name']}: ~{per_contact:,.0f}/контакт, ~{total:,.0f} итого")
        return result

    def _fallback_campaign(self, cells, cands):
        """Ничего не прошло порог: одна бесплатная кампания на самую надёжную гипотезу."""
        c = max(cands, key=lambda x: x["m"])
        cell = cells[c["cell"]]
        free = min(self.channels, key=lambda ch: self.channels[ch]["cost_per_contact"])
        self.log(f"[agent] нет уверенно прибыльных кампаний — fallback {cell['tariff']}/{cell['seg']} → {c['target']}")
        return {"campaign_name": "fallback", "filter_arpu_segment": cell["seg"],
                "filter_current_tariff": cell["tariff"], "target_tariff": c["target"], "channel": free}
