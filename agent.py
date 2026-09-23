"""
Агент тарифных кампаний: байесовская разведка пилотами + распределение охвата.

Логика в трёх шагах:

1. ПРИОР. Для каждой ячейки аудитории (current_tariff × arpu_segment) и каждого
   целевого тарифа оцениваем эффект по истории data/change_tariff.csv:
   средний % изменения ARPU × доля переходов. История описывает другую выборку,
   поэтому неопределённость приора берём широкой (±100% от оценки).

2. РАЗВЕДКА. Каждый пилот выбирается адаптивно — тот, чей результат с наибольшей
   ожидаемой пользой может изменить финальный план (knowledge gradient). Результат
   пилота объединяется с приором по весам точности: пилот на 200 клиентах
   сдвигает оценку сильно, на 40 — слабо.

3. ПЛАН. Кампанию запускаем только если консервативная оценка (среднее − κ·σ)
   положительна. Охват (15 000 контактов) отдаём ячейкам с наибольшей ценностью
   на контакт, затем оставшиеся деньги тратим на апгрейд канала там, где
   дорогой канал окупается (дорогие абоненты × сильный эффект).
"""

import math
import os

import pandas as pd

# Шум пилота: std наблюдаемого эффекта ≈ PILOT_NOISE / sqrt(n) (≈0.07 на 150 клиентах).
PILOT_NOISE = 0.8
PILOT_CHANNEL = "sms"
PILOT_SIZES = (40, 80, 120, 200)
PILOT_CONTACT_SHARE = 0.25     # не больше четверти общего охвата уходит на разведку
PILOT_OPPORTUNITY_COST = 30    # цена «сожжённого» контакта при выборе размера пилота
MIN_CELL_SIZE = 60             # ячейки меньше не пилотируем и не таргетируем

# Приор из истории: насколько мы ей доверяем
PRIOR_REL_SD = 1.0             # история — другая выборка: σ ≈ 100% от оценки
PRIOR_ABS_SD = 0.05
UNSEEN_SD = 0.08               # переходы, которых нет в истории
SHRINK_K0 = 5                  # сглаживание малых групп к среднему по паре тарифов
TARGETS_PER_CELL = 4

KAPPA = 1.0                    # запускаем кампанию, если mean − κ·σ > 0
MAX_CAMPAIGNS = 10
MAX_PER_CAMPAIGN = 5000

ARPU_BINS = [-math.inf, 1000, 5000, math.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]


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
        """(from, to, seg) -> (mean, sd) эффекта в «базовых» единицах (множитель канала = 1)."""
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
            m0 = pct * conv
            var_hist = (conv * pct_sd) ** 2 / k
            s0 = math.sqrt(var_hist + (PRIOR_REL_SD * m0) ** 2 + PRIOR_ABS_SD ** 2)
            prior[(r.tariff_plan_code_from, r.tariff_plan_code_to, r.seg)] = (m0, s0)
        self._median_conv = float((g["size"] / g["total"]).median())
        return prior

    def _price_prior(self, tariffs, t_from, t_to):
        """Для переходов без истории: знак по разнице цен, эффект небольшой, неопределённость широкая."""
        price = tariffs.set_index("tariff_plan_code")["price_tariff"]
        conv = getattr(self, "_median_conv", 0.1)
        if t_from not in price.index or t_to not in price.index:
            return 0.0, UNSEEN_SD
        base = max(float(price[t_from]), float(price.median()), 1.0)
        pct = max(min((float(price[t_to]) - float(price[t_from])) / base, 1.0), -1.0) * 0.5
        return pct * conv, UNSEEN_SD

    def _build_candidates(self, cells, prior, tariffs):
        targets = list(tariffs["tariff_plan_code"])
        cands = []
        for key, cell in cells.items():
            options = []
            for t in targets:
                if t == cell["tariff"]:
                    continue
                m0, s0 = prior.get((cell["tariff"], t, cell["seg"])) or self._price_prior(tariffs, cell["tariff"], t)
                options.append({"cell": key, "target": t, "m": m0, "s": s0, "n_pilots": 0, "pilot_n": 0})
            # оставляем самые многообещающие по оптимистичной оценке
            options.sort(key=lambda o: o["m"] + o["s"], reverse=True)
            cands.extend(options[:TARGETS_PER_CELL])
        return cands

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
            for c in cands:
                cell = cells[c["cell"]]
                others = [self._lb(o) for o in by_cell[c["cell"]] if o is not c]
                others_best = max(others) if others else 0.0
                n_max = min(max(PILOT_SIZES), cell["n"] // 2, pilot_contact_cap - used)
                if cost > 0:
                    n_max = min(n_max, int(env.remaining_budget // cost))
                for n in sorted({s for s in PILOT_SIZES if s <= n_max} | ({n_max} if n_max >= min(PILOT_SIZES) else set())):
                    score = self._pilot_value(c, cell, n, others_best, mult) - n * (cost + PILOT_OPPORTUNITY_COST)
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
            var_obs = (PILOT_NOISE / math.sqrt(n_real) / mult) ** 2
            prec = 1.0 / c["s"] ** 2 + 1.0 / var_obs
            m_old = c["m"]
            c["m"] = (c["m"] / c["s"] ** 2 + y / var_obs) / prec
            c["s"] = math.sqrt(1.0 / prec)
            c["n_pilots"] += 1
            c["pilot_n"] += n_real
            self.log(f"[agent] пилот {cell['tariff']}/{cell['seg']} → {c['target']} n={n_real}: "
                     f"наблюдали {y:+.3f}, оценка {m_old:+.3f} → {c['m']:+.3f} ± {c['s']:.3f}")

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
