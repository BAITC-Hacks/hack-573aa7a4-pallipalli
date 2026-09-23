"""
Стресс-тест агента на «альтернативных мирах».

Мок-среда организаторов построена прямо из data/change_tariff.csv, поэтому агент,
который просто верит истории, выглядит на моке отлично — а на судействе эффекты
другие. Здесь мы проверяем агента в мирах, где эффекты отличаются от истории
разными способами (шум, завышение, смена знака, другие лучшие тарифы, ...).

Всё считается ОФИЦИАЛЬНЫМ кодом: среда — environment.make_environment (тот же
шум пилотов), результат — scoring_core.score_campaigns (те же лимиты,
дедупликация и формула). Меняется только impact-модель.

    python stress_eval.py                       # все миры × 5 seed
    python stress_eval.py --seeds 10 --worlds mixed,fresh
    python stress_eval.py --agents ours=agent:Agent,new=agent_v2:Agent
    python stress_eval.py --csv stress_results.csv
    python stress_eval.py --list                # описание миров

Для сравнения всегда считаются референсы:
  oracle        — знает истинные эффекты мира, пилотов не делает (жадный план);
                  ориентир потолка, а не точный оптимум;
  template      — agent_template.py организаторов;
  history-only  — верит истории, пилот делает «для галочки». Показывает, как
                  выглядит переобучение под мок.
"""

import argparse
import contextlib
import importlib
import io
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from environment import make_environment
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET, _mock_fallback, _mock_impact_model
from scoring_core import MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN, sanitize_campaigns, score_campaigns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FROM, TO, SEG = "tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"
PCT, CONV = "arpu_change_pct", "conversion_rate"
FILTER_COLUMNS = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff"]
RUNTIME_LIMIT = 300          # консервативно: шаблон говорит о 5 минутах
ORACLE_MIN = 50_000          # ниже этого % от оракула не показываем — делить почти на ноль
DEFAULT_AGENTS = "ours=agent:Agent,template=agent_template:Agent,history-only=history"


# ------------------------------------------------------------------ данные
_DATA = None


def _data():
    """Загружается один раз на процесс."""
    global _DATA
    if _DATA is None:
        profile = pd.read_csv(os.path.join(BASE_DIR, "customer_profile.csv"))
        tariffs = pd.read_csv(os.path.join(BASE_DIR, "data", "dict_tariff.csv"))
        history = _mock_impact_model(pd.read_csv(os.path.join(BASE_DIR, "data", "change_tariff.csv")))
        _DATA = {
            "profile": profile,
            "tariffs": tariffs,
            "baseline": float(profile["predicted_arpu"].sum()),
            "grid": _full_grid(profile, tariffs, history),
            "cells": _cells(profile),
        }
    return _DATA


def _full_grid(profile, tariffs, history):
    """
    Полная impact-модель: каждая ячейка аудитории (тариф × ARPU-сегмент) × каждый
    из 21 целевого тарифа. Где истории нет — то же ценовое правило, что в моке,
    поэтому мир "mock" совпадает с local_eval.py один в один. Полная сетка нужна,
    чтобы искажать все переходы, а не только встречавшиеся в истории.
    """
    cells = profile[["current_tariff", SEG]].drop_duplicates().rename(columns={"current_tariff": FROM})
    grid = cells.merge(pd.DataFrame({TO: tariffs["tariff_plan_code"]}), how="cross")
    hist = history.assign(**{SEG: history[SEG].astype(str)})[[FROM, TO, SEG, PCT, CONV, "count"]]
    grid = grid.merge(hist, on=[FROM, TO, SEG], how="left")

    missing = grid[PCT].isna()
    fallback_conv = history[CONV].median()
    fb = [_mock_fallback(r[FROM], r[TO], r[SEG], tariffs, fallback_conv) for _, r in grid[missing].iterrows()]
    grid.loc[missing, PCT] = [f[0] for f in fb]
    grid.loc[missing, CONV] = [f[1] for f in fb]
    grid["count"] = grid["count"].fillna(0).astype(int)
    return grid.reset_index(drop=True)


def _cells(profile):
    """Ячейки аудитории с размером и средним ARPU (первые 5000 по ID — как режет скоринг)."""
    rows = []
    for (tariff, seg), g in profile.sort_values("ID_NUMBER").groupby(["current_tariff", SEG]):
        g = g.iloc[:MAX_CUSTOMERS_PER_CAMPAIGN]
        rows.append({"current_tariff": tariff, SEG: seg, "n": len(g), "arpu_mean": float(g["predicted_arpu"].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- миры
def _permute_within(g, by, rng):
    """Перемешивает (эффект, конверсию) между строками внутри каждой группы."""
    out = g.copy()
    for idx in g.groupby(by).groups.values():
        out.loc[idx, [PCT, CONV]] = g.loc[rng.permutation(idx), [PCT, CONV]].to_numpy()
    return out


def _w_mock(g, rng):
    return g


def _w_noisy(g, rng):
    g[PCT] += rng.normal(0, 0.15, len(g))
    g[CONV] *= rng.lognormal(0, 0.3, len(g))
    return g


def _w_weaker(g, rng):
    g[PCT] = g[PCT] * 0.5 - 0.03
    return g


def _w_downsell(g, rng):
    g[PCT] -= 0.15
    return g


def _w_sign_flip(g, rng):
    flip = rng.random(len(g)) < 0.3
    g.loc[flip, PCT] *= -1
    return g


def _w_shuffled(g, rng):
    return _permute_within(g, [FROM, SEG], rng)


def _w_segment_swap(g, rng):
    return _permute_within(g, [FROM, TO], rng)


def _w_sparse_noise(g, rng):
    sd = 0.6 / np.sqrt(1 + g["count"])
    g[PCT] += rng.normal(0, 1, len(g)) * sd
    g[CONV] *= np.exp(rng.normal(0, 1, len(g)) * sd)
    return g


def _w_mixed(g, rng):
    sd = 0.4 / np.sqrt(1 + g["count"])
    g[PCT] = g[PCT] * rng.uniform(0.5, 1.2) + rng.normal(0, 0.08, len(g)) + rng.normal(0, 1, len(g)) * sd
    g.loc[rng.random(len(g)) < 0.15, PCT] *= -1
    g[CONV] *= rng.lognormal(0, 0.3, len(g))
    return g


def _w_fresh(g, rng):
    seen = g[g["count"] > 0]
    g[PCT] = rng.choice(seen[PCT].to_numpy(), len(g))
    g[CONV] = rng.choice(seen[CONV].to_numpy(), len(g))
    return g


def _w_hostile(g, rng):
    g[PCT] = -g[PCT].abs() - 0.02
    return g


WORLDS = {
    "mock":         ("мок организаторов = средние по истории", _w_mock),
    "noisy":        ("история верна в среднем, каждый переход сдвинут шумом ±15 п.п.", _w_noisy),
    "weaker":       ("история завышает: все эффекты вдвое слабее и −3 п.п.", _w_weaker),
    "downsell":     ("рынок хуже: −15 п.п. к каждому переходу", _w_downsell),
    "sign_flip":    ("у 30% переходов знак эффекта обратный", _w_sign_flip),
    "shuffled":     ("лучшие целевые тарифы перемешаны внутри каждой ячейки", _w_shuffled),
    "segment_swap": ("эффекты перемешаны между ARPU-сегментами (HIGH может стать ценным)", _w_segment_swap),
    "sparse_noise": ("чем меньше наблюдений в истории, тем сильнее ошибка", _w_sparse_noise),
    "mixed":        ("похоже на судейство: общий масштаб, шум, 15% ошибок знака", _w_mixed),
    "fresh":        ("история бесполезна: эффекты заново из общего распределения", _w_fresh),
    "hostile":      ("все переходы убыточны — лучшее решение почти ничего не делать", _w_hostile),
}


def make_world(name, seed):
    """Impact-модель мира. Каждый seed — свой экземпляр случайного мира."""
    base = _data()["grid"]
    rng = np.random.default_rng([zlib.crc32(name.encode()), seed])
    g = WORLDS[name][1](base.copy(), rng)
    same = g[FROM] == g[TO]                     # «перейти на свой же тариф» не искажаем
    g.loc[same, [PCT, CONV]] = base.loc[same, [PCT, CONV]]
    g[PCT] = g[PCT].clip(-1, 3)
    g[CONV] = g[CONV].clip(0, 1)
    return g


# --------------------------------------------------------- референсы
def plan_from_beliefs(beliefs, cells, channels, contacts, budget):
    """
    Жадный план по заданным «убеждениям» об эффектах: лучший тариф на ячейку,
    охват — самым ценным на контакт, остаток денег — на апгрейд канала,
    упаковка в ≤10 кампаний. Для оракула убеждения = истина мира.
    """
    cost = {ch: spec["cost_per_contact"] for ch, spec in channels.items()}
    cheap = sorted(channels, key=cost.get)[:2]

    m = cells.merge(beliefs.rename(columns={FROM: "current_tariff"}), on=["current_tariff", SEG])
    m = m[m["current_tariff"] != m[TO]]
    m = (m.assign(base=m[PCT] * m[CONV]).sort_values("base", ascending=False)
          .drop_duplicates(["current_tariff", SEG]))

    units = []
    for r in m.itertuples(index=False):
        pct, conv = getattr(r, PCT), getattr(r, CONV)
        val = {ch: pct * min(conv * spec["conversion_multiplier"], 1.0) * r.arpu_mean - cost[ch]
               for ch, spec in channels.items()}
        ch = max(cheap, key=val.get)
        if val[ch] > 0:
            units.append({"tariff": r.current_tariff, "seg": getattr(r, SEG), "target": getattr(r, TO),
                          "n": int(r.n), "val": val, "ch": ch})

    units.sort(key=lambda u: u["val"][u["ch"]], reverse=True)
    chosen = []
    for u in units:  # частично ячейки не берём: фильтр не умеет выбрать «половину ячейки»
        if u["n"] <= contacts and u["n"] * cost[u["ch"]] <= budget:
            contacts -= u["n"]
            budget -= u["n"] * cost[u["ch"]]
            chosen.append(u)

    upgrades = []
    for u in chosen:
        for ch in channels:
            gain = u["n"] * (u["val"][ch] - u["val"][u["ch"]])
            extra = u["n"] * (cost[ch] - cost[u["ch"]])
            if gain > 0 and extra > 0:
                upgrades.append((gain / extra, extra, u, ch))
    upgrades.sort(key=lambda x: x[0], reverse=True)
    done = set()
    for _, extra, u, ch in upgrades:
        if id(u) not in done and extra <= budget:
            budget -= extra
            u["ch"] = ch
            done.add(id(u))

    groups = {}
    for u in chosen:
        buckets = groups.setdefault((u["target"], u["ch"], u["seg"]), [[]])
        if sum(x["n"] for x in buckets[-1]) + u["n"] > MAX_CUSTOMERS_PER_CAMPAIGN:
            buckets.append([])
        buckets[-1].append(u)

    campaigns = []
    for (target, ch, seg), buckets in groups.items():
        for members in buckets:
            total = sum(x["n"] * x["val"][ch] for x in members)
            n = sum(x["n"] for x in members)
            campaigns.append((total / n, total, {
                "campaign_name": f"ref_{seg.lower()}_{target}_{ch}_{len(campaigns)}",
                "filter_arpu_segment": seg,
                "filter_current_tariff": ";".join(x["tariff"] for x in members),
                "target_tariff": target,
                "channel": ch,
            }))
    campaigns = sorted(campaigns, key=lambda x: x[1], reverse=True)[:MAX_CAMPAIGNS]
    campaigns.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in campaigns]


class HistoryOnlyAgent:
    """Верит истории как истине; один крошечный пилот — только чтобы выполнить правило."""

    def act(self, env):
        history = _data()["grid"]
        cells = _cells(env.customer_profile)
        plan = plan_from_beliefs(history, cells, env.channels, env.remaining_contacts, env.remaining_budget)
        if plan:
            first = plan[0]
            try:
                env.run_pilot(target_tariff=first["target_tariff"], channel="push", n_customers=10,
                              filter_arpu_segment=first["filter_arpu_segment"],
                              filter_current_tariff=first["filter_current_tariff"].split(";")[0])
            except (RuntimeError, ValueError):
                pass
        return plan_from_beliefs(history, cells, env.channels, env.remaining_contacts, env.remaining_budget)


ORACLE = "oracle"
BUILTIN_AGENTS = {"history": HistoryOnlyAgent}


def _make_agent(spec):
    if spec in BUILTIN_AGENTS:
        return BUILTIN_AGENTS[spec]()
    module, cls = spec.split(":")
    klass = getattr(importlib.import_module(module), cls)
    try:
        return klass(verbose=False)
    except TypeError:
        return klass()


# ------------------------------------------------------------ прогон
def _score(campaigns, grid, d):
    df = pd.DataFrame(campaigns)
    if df.empty:
        return {"net_arpu_gain": 0.0, "gross_arpu_lift": 0.0, "total_cost": 0.0,
                "total_contacts": 0, "unique_customers_targeted": 0, "risk_score_pct": 0.0}
    for col in FILTER_COLUMNS + ["explicit_ids"]:
        if col not in df.columns:
            df[col] = None
    return score_campaigns(df, d["profile"], grid, d["tariffs"], d["baseline"], _mock_fallback)


def run_task(label, spec, world, seed):
    """Один прогон агента в одном мире. Ошибка агента = пилоты в зачёт, финал пустой (как на судействе)."""
    d = _data()
    grid = make_world(world, seed)
    row = {"agent": label, "world": world, "seed": seed, "error": "", "runtime": 0.0,
           "n_raw": 0, "discarded": 0, "pilot_net": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        if spec == ORACLE:
            final = plan_from_beliefs(grid, d["cells"], CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET)
            pilots = []
        else:
            env, internals = make_environment(d["profile"], grid, d["tariffs"], CHANNELS, TOTAL_BUDGET,
                                              MAX_TOTAL_CONTACTS, _mock_fallback, seed=seed)
            t0 = time.perf_counter()
            try:
                raw = _make_agent(spec).act(env)
            except Exception as e:
                raw, row["error"] = [], f"{type(e).__name__}: {e}"
            row["runtime"] = time.perf_counter() - t0
            raw = list(raw) if isinstance(raw, (list, tuple)) else []
            final = sanitize_campaigns(raw, d["tariffs"])
            row["n_raw"], row["discarded"] = len(raw), len(raw) - len(final)
            final = final[:MAX_CAMPAIGNS]
            pilots = internals.executed_pilot_campaigns()

        res = _score(pilots + final, grid, d)
        if pilots:
            row["pilot_net"] = _score(pilots, grid, d)["net_arpu_gain"]

    row.update(net=res["net_arpu_gain"], gross=res["gross_arpu_lift"], cost=res["total_cost"],
               contacts=res["total_contacts"], unique=res["unique_customers_targeted"],
               risk=res["risk_score_pct"], n_campaigns=len(final), n_pilots=len(pilots))
    return row


def _run_all(tasks, jobs):
    rows = []
    total = len(tasks)
    step = max(total // 10, 1)

    def progress():
        if sys.stderr.isatty():
            print(f"\r  прогонов: {len(rows)}/{total}", end="", file=sys.stderr, flush=True)
        elif len(rows) % step == 0 or len(rows) == total:
            print(f"  прогонов: {len(rows)}/{total}", file=sys.stderr, flush=True)

    if jobs <= 1:
        for t in tasks:
            rows.append(run_task(*t))
            progress()
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for fut in as_completed([pool.submit(run_task, *t) for t in tasks]):
                rows.append(fut.result())
                progress()
    if sys.stderr.isatty():
        print(file=sys.stderr)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- отчёт
def _k(x):
    return f"{x / 1000:,.0f}k"


def report(df, worlds, labels, n_seeds):
    has_oracle = ORACLE in set(df["agent"])
    agents = [a for a in labels if a != ORACLE]
    g = df.groupby(["world", "agent"])["net"]
    mean, worst = g.mean(), g.min()
    neg = df.assign(neg=df["net"] < 0).groupby(["world", "agent"])["neg"].sum()

    def oracle_mean(w):
        return mean.get((w, ORACLE), np.nan) if has_oracle else np.nan

    w0 = max(len(w) for w in worlds) + 2
    col = 17
    print(f"\nСтресс-тест: {len(worlds)} миров × {n_seeds} seed. Чистый результат в тыс. у.е.")
    if has_oracle:
        print("oracle знает истинные эффекты мира (жадный план без пилотов) — ориентир потолка.\n")

    print("СРЕДНИЙ результат и % от оракула")
    print("мир".ljust(w0) + ("oracle".rjust(10) if has_oracle else "") + "".join(a.rjust(col) for a in agents))
    for w in worlds:
        o = oracle_mean(w)
        line = w.ljust(w0) + (_k(o).rjust(10) if has_oracle else "")
        for a in agents:
            m = mean.get((w, a), np.nan)
            share = f"{100 * m / o:.0f}%" if has_oracle and o > ORACLE_MIN else "—"
            line += f"{_k(m)} {share:>5}".rjust(col)
        print(line)

    print("\nХУДШИЙ прогон и число прогонов в минус")
    print("мир".ljust(w0) + "".join(a.rjust(col) for a in agents))
    for w in worlds:
        line = w.ljust(w0)
        for a in agents:
            line += f"{_k(worst.get((w, a), np.nan))} {int(neg.get((w, a), 0))}/{n_seeds}".rjust(col)
        print(line)

    print("\nИТОГ ПО АГЕНТАМ")
    for a in agents:
        d = df[df["agent"] == a]
        shares = [mean[(w, a)] / oracle_mean(w) for w in worlds if has_oracle and oracle_mean(w) > ORACLE_MIN]
        worst_row = d.loc[d["net"].idxmin()]
        print(f"- {a}:")
        if shares:
            print(f"    средний % от оракула: {100 * np.mean(shares):.0f}%   "
                  f"(медиана по мирам {100 * np.median(shares):.0f}%)")
        print(f"    в минус: {int((d['net'] < 0).sum())}/{len(d)} прогонов   "
              f"худший: {_k(worst_row['net'])} ({worst_row['world']}, seed {worst_row['seed']})")
        print(f"    пилотов в среднем: {d['n_pilots'].mean():.1f}   результат пилотов: {_k(d['pilot_net'].mean())}   "
              f"контактов: {d['contacts'].mean():,.0f}   бюджет: {d['cost'].mean():,.0f}")
        print(f"    время: среднее {d['runtime'].mean():.1f}с, максимум {d['runtime'].max():.1f}с")

        problems = []
        if (d["error"] != "").any():
            errs = d.loc[d["error"] != "", "error"]
            problems.append(f"падений {len(errs)} (пример: {errs.iloc[0][:120]})")
        if (d["n_pilots"] == 0).any():
            problems.append(f"без пилотов {int((d['n_pilots'] == 0).sum())} прогонов — нарушение must-have")
        if (d["n_campaigns"] == 0).any():
            problems.append(f"пустой план {int((d['n_campaigns'] == 0).sum())} прогонов")
        if (d["discarded"] > 0).any():
            problems.append(f"отброшенные кампании в {int((d['discarded'] > 0).sum())} прогонах")
        if (d["n_raw"] > MAX_CAMPAIGNS).any():
            problems.append(f"больше {MAX_CAMPAIGNS} кампаний в {int((d['n_raw'] > MAX_CAMPAIGNS).sum())} прогонах")
        if (d["runtime"] > RUNTIME_LIMIT).any():
            problems.append(f"дольше {RUNTIME_LIMIT}с в {int((d['runtime'] > RUNTIME_LIMIT).sum())} прогонах")
        for p in problems:
            print(f"    ⚠ {p}")


def main():
    ap = argparse.ArgumentParser(description="Стресс-тест агента на мирах с эффектами, отличными от истории.")
    ap.add_argument("--seeds", type=int, default=5, help="прогонов на мир (default 5)")
    ap.add_argument("--worlds", default="all", help="через запятую; см. --list")
    ap.add_argument("--agents", default=DEFAULT_AGENTS, help="label=module:Class,... (default: %(default)s)")
    ap.add_argument("--no-oracle", action="store_true", help="не считать оракула")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="параллельных процессов")
    ap.add_argument("--csv", help="сохранить все прогоны в CSV")
    ap.add_argument("--list", action="store_true", help="показать миры и выйти")
    args = ap.parse_args()

    os.chdir(BASE_DIR)  # агенты читают data/ по относительному пути
    if args.list:
        for name, (desc, _) in WORLDS.items():
            print(f"{name:14} {desc}")
        return

    worlds = list(WORLDS) if args.worlds == "all" else [w.strip() for w in args.worlds.split(",")]
    unknown = [w for w in worlds if w not in WORLDS]
    if unknown:
        sys.exit(f"Неизвестные миры: {unknown}. Доступны: {list(WORLDS)}")

    agents = []
    for item in args.agents.split(","):
        label, _, spec = item.strip().partition("=")
        agents.append((label, spec or label))
    for label, spec in agents:  # ошибка импорта должна быть видна сразу, а не как 50 «падений»
        try:
            _make_agent(spec)
        except Exception as e:
            sys.exit(f"Не удалось создать агента {label} ({spec}): {type(e).__name__}: {e}")

    runners = ([] if args.no_oracle else [(ORACLE, ORACLE)]) + agents
    tasks = [(label, spec, w, s) for w in worlds for s in range(args.seeds) for label, spec in runners]
    t0 = time.perf_counter()
    df = _run_all(tasks, args.jobs)
    print(f"  готово за {time.perf_counter() - t0:.0f}с", file=sys.stderr)

    report(df, worlds, [label for label, _ in runners], args.seeds)
    if args.csv:
        df.sort_values(["world", "agent", "seed"]).to_csv(args.csv, index=False)
        print(f"\nВсе прогоны: {args.csv}")


if __name__ == "__main__":
    main()
