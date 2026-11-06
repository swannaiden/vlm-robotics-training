"""Score every rollout and draw the figures.

    python plot/make_plots.py [post|overview|compare|openai|anthropic|google ...]   (default: all)

Writes figures/*.png, each with a CSV of every plotted value. Uses Google Sans for the overview figure if it is
installed or placed in fonts/ (from Google Fonts); otherwise DejaVu Sans.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGURES = ROOT / "data", ROOT / "figures"
RUNS = ("panda", "panda_strict")  # "don't calculate" prompt; the same plus "DO NOT USE ANY REASONING"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import fk_position, load_arms, score  # noqa: E402

# Google Sans (from Google Fonts) for the overview figure, if downloaded to data/fonts; DejaVu Sans otherwise.
from matplotlib import font_manager  # noqa: E402

for _font in sorted((ROOT / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(_font))
OVERVIEW_FONT = ["Google Sans", "DejaVu Sans"]
BLACK = "#000000"

# Models in release order, with short axis labels.
LINEUPS = {
    "openai": [
        ("gpt-4o-2024-11-20", "4o"), ("gpt-4.1", "4.1"), ("gpt-5", "5"), ("gpt-5.1", "5.1"), ("gpt-5.2", "5.2"),
        ("gpt-5.4", "5.4"), ("gpt-5.5", "5.5"), ("gpt-5.6-sol", "5.6 Sol"), ("gpt-6-astra", "6 Astra"),
        ("gpt-6-sol", "6 Sol"), ("gpt-6.1-sol", "6.1 Sol"),
    ],
    "anthropic": [
        ("claude-sonnet-4.5", "Sonnet 4.5"), ("claude-opus-4.5", "Opus 4.5"), ("claude-opus-4.6", "Opus 4.6"),
        ("claude-opus-4.7", "Opus 4.7"), ("claude-opus-4.8", "Opus 4.8"), ("claude-opus-5", "Opus 5"),
        ("claude-fable-5.1", "Fable 5.1"), ("claude-opus-5.5", "Opus 5.5"),
    ],
    "google": [
        ("gemini-2.5-flash", "2.5 Flash"), ("gemini-3-flash-preview", "3 Flash"), ("gemini-3.1-pro-preview", "3.1 Pro"),
        ("gemini-3.1-flash-lite", "3.1 Lite"), ("gemini-3.5-flash", "3.5 Flash"), ("gemini-3.6-flash", "3.6 Flash"),
        ("gemini-3.7-flash", "3.7 Flash"), ("gemini-3.8-flash", "3.8 Flash"), ("gemini-4-argon", "4 Argon"),
    ],
}
# Every model, grouped for the overview dot plot: (group, vendor prefix, [(model id, display name)]).
GROUPS = [
    ("OpenAI", "openai", [(m, f"GPT-{n}") for m, n in LINEUPS["openai"]]),
    ("Anthropic", "anthropic", [(m, f"Claude {n}") for m, n in LINEUPS["anthropic"]]),
    ("Google", "google", [(m, "Gemini 3.1 Flash-Lite" if m == "gemini-3.1-flash-lite" else f"Gemini {n}") for m, n in LINEUPS["google"]]),
    ("DeepSeek", "deepseek", [("deepseek-chat", "DeepSeek V3"), ("deepseek-chat-v3.1", "DeepSeek V3.1"),
                              ("deepseek-v3.2", "DeepSeek V3.2"), ("deepseek-v4-pro", "DeepSeek V4 Pro"),
                              ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash")]),
    ("Qwen", "qwen", [("qwen-2.5-72b-instruct", "Qwen2.5 72B"), ("qwen3-235b-a22b-2507", "Qwen3 235B 2507"),
                      ("qwen3.5-397b-a17b", "Qwen3.5 397B"), ("qwen3.7-max", "Qwen3.7 Max"), ("qwen3.8-flash", "Qwen3.8 Flash")]),
    ("Kimi", "moonshotai", [("kimi-k2", "Kimi K2"), ("kimi-k2.5", "Kimi K2.5"), ("kimi-k3", "Kimi K3"), ("kimi-k3.1-pro", "Kimi K3.1 Pro")]),
    ("GLM", "z-ai", [("glm-4.5", "GLM 4.5"), ("glm-4.7", "GLM 4.7"), ("glm-5.1", "GLM 5.1"), ("glm-5.2", "GLM 5.2")]),
    ("Other Chinese labs", None, [("minimax/minimax-m3", "MiniMax M3"), ("bytedance-seed/seed-2-1-turbo", "Seed 2.1 Turbo"),
                                  ("xiaomi/mimo-v2.6-pro", "MiMo 2.6 Pro"), ("tencent/hy4", "Hunyuan Hy4")]),
]
# Colours are reference-palette slots 1-3 (validated all-pairs), each paired with its own marker shape.
SLOTS = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D")]
CONDITIONS = [
    ("off", "Reasoning off"),
    ("brief", "Brief reasoning (lowest effort; can't be off)"),
    ("strict", 'Told "DO NOT USE ANY REASONING" (0 reasoning tokens)'),
]
PROVIDERS = [("openai", "OpenAI"), ("anthropic", "Anthropic (Claude)"), ("google", "Google (Gemini)")]
# Points named in the comparison figure, with label offsets (points) chosen to stay clear of each other.
LABEL_AT = {"gpt-4o-2024-11-20": (7, 0, "left"), "claude-sonnet-4.5": (0, -13, "center"), "gemini-2.5-flash": (8, 0, "left")}
# Late-2026 points sit too close together for adjacent labels: these go in the empty band at the bottom of the
# panel with a leader line to the point (text date, text height in cm, alignment).
LEADER_AT = {"gpt-6-astra": ("2026-07-01", 6.5, "right"), "gpt-6.1-sol": ("2026-08-20", 3.0, "right"),
             "gemini-4-argon": ("2026-11-05", 6.5, "right")}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

PANELS = [
    ("FK error, real Panda with its table", "cm", "fk", "real", "spec", "err_m", np.median),
    ("Multiple choice correct, real Panda", "%", "fk_mc", "real", "spec", "correct", np.mean),
    ("IK error, real Panda with its table", "cm", "ik", "real", "spec", "err_m", np.median),
    ("Twisted arm: distance from the Panda's answer", "cm", "fk", "twisted", "spec", "err_twin_m", np.median),
    ("Alien arm (no Panda numbers): distance from the Panda's answer", "cm", "fk", "alien", "spec", "err_twin_m", np.median),
    ("Joint angles only: distance from the Panda's answer", "cm", "fk", "real", "anon", "err_m", np.median),
]


def condition_of(run: str, effort: str) -> str:
    if run == "panda_strict":
        return "strict"
    return "brief" if effort == "low" else "off"  # GPT-5 runs at "minimal" but used 0 reasoning tokens


def load_rows() -> list[dict]:
    """Every rollout, scored against its task's ground truth."""
    rows = []
    for run in RUNS:
        tasks = {t["task_id"]: t for t in map(json.loads, (DATA / run / "tasks.jsonl").open())}
        arms = load_arms(DATA / run / "arms.json")
        scored = []
        for path in sorted((DATA / run / "rollouts").glob("*.jsonl")):
            for record in map(json.loads, path.open()):
                record.setdefault("spec", path.stem)
                scored.append(score(tasks[record["task_id"]], record, arms))
        for r in scored:
            r["condition_run"] = condition_of(run, r["effort"])
            r["provider"], rest = r["label"].split("/", 1)
            r["model_id"] = rest.split("@")[0]
            rows.append(r)
    return rows


def release_dates() -> dict[str, dt.date]:
    out = {}
    for run in RUNS:
        for mid, entry in json.loads((DATA / run / "models.json").read_text()).items():
            date = dt.datetime.fromtimestamp(entry["created"], dt.UTC).date()
            out[mid] = out[mid.split("/", 1)[1]] = date
    return out


def values(rows, task, kind, cond, field) -> list[float]:
    out = []
    for r in rows:
        if r["task"] == task and r["kind"] == kind and r["condition"] == cond:
            if field == "correct":
                out.append(100.0 * bool(r["correct"]))
            elif r["parsed"]:
                out.append(r[field] * 100)
    return out


def bootstrap(vals, stat, rng, n=2000):
    vals = np.asarray(vals)
    boot = [stat(rng.choice(vals, len(vals))) for _ in range(n)]
    return float(stat(vals)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def baselines() -> dict[str, tuple[float, str]]:
    arms = load_arms(DATA / "panda" / "arms.json")
    tasks = [json.loads(l) for l in (DATA / "panda" / "tasks.jsonl").open()]
    real_fk = np.array([t["answer"] for t in tasks if t["task"] == "fk" and t["kind"] == "real" and t["condition"] == "spec"])
    constant = float(np.median(np.linalg.norm(real_fk - real_fk.mean(axis=0), axis=1)) * 100)
    q_median = json.loads((DATA / "droid_median_q.json").read_text())["q"]  # median DROID joint configuration
    p_med = fk_position(arms["panda"], np.array(q_median))
    targets = [t["target"] for t in tasks if t["task"] == "ik" and t["kind"] == "real" and t["condition"] == "spec"]
    ik = float(np.median([np.linalg.norm(p_med - t) for t in targets]) * 100)
    guess = (constant, f"constant guess, {constant:.0f} cm")
    return {"fk": guess, "fk_mc": (25.0, "chance, 25%"), "ik": (ik, f"always DROID's median pose, {ik:.0f} cm")}


def style_axis(ax, title, unit, y0, label, xlim=None):
    ax.set_facecolor(SURFACE)
    ax.axhline(y0, color=MUTED, lw=1.2, zorder=1)
    # Name the baseline in a corner the data never reach (models sit near the line at both ends): bottom left
    # for errors, bottom right for multiple choice.
    right = unit == "%"
    ax.text(0.99 if right else 0.01, 0.03, f"gray line: {label}", transform=ax.transAxes,
            ha="right" if right else "left", va="bottom", fontsize=8, color=INK2)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel(unit, rotation=0, ha="right", va="center", labelpad=6)
    if unit == "%":
        ax.set_ylim(0, 102)
    else:
        ax.set_ylim(0, max(ax.get_ylim()[1], y0 * 1.3))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_points(ax, pts, color, marker, connect):
    xs, cs, los, his = zip(*pts)
    if connect:
        ax.plot(xs, cs, color=color, lw=2, alpha=0.45, solid_capstyle="round", zorder=2)
    ax.vlines(xs, los, his, color=color, lw=1.5, alpha=0.8, zorder=3)
    ax.plot(xs, cs, linestyle="none", marker=marker, ms=7.5, color=color, mec=SURFACE, mew=1.5, zorder=4)


def new_figure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    return plt.subplots(3, 2, figsize=(13, 12.5), facecolor=SURFACE)


def finish(fig, title, subtitle, footnote, handles, out, table):
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.94), ncol=4, frameon=False, fontsize=8.5)
    fig.text(0.055, 0.985, title, fontsize=14, weight="semibold", va="top")
    fig.text(0.055, 0.963, subtitle, fontsize=9, color=INK2, va="top", linespacing=1.5)
    fig.text(0.055, 0.008, footnote, fontsize=7.5, color=MUTED, va="bottom", linespacing=1.4)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.885 if handles else 0.9, bottom=0.06, hspace=0.42, wspace=0.12)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {out} and {out.with_suffix('.csv')}")


def handle(color, marker, label):
    return plt.Line2D([], [], color=color, marker=marker, ms=7.5, mec=SURFACE, mew=1.5, lw=0, label=label)


def lineup_figure(provider: str, rows, base, dates, out: Path) -> None:
    """One provider's models on a categorical axis, one series per run condition. A provider with a single
    condition is drawn in its colour from the comparison figure, so each provider keeps one colour."""
    models = [(m, name) for m, name in LINEUPS[provider] if m in dates]
    rows = [r for r in rows if r["provider"] == provider]
    present = [c for c in CONDITIONS if any(r["condition_run"] == c[0] for r in rows)]
    offsets = dict(zip([c for c, _ in present], np.linspace(-0.17, 0.17, len(present)) if len(present) > 1 else [0.0]))
    rng = np.random.default_rng(0)
    fig, axes = new_figure()
    table = []
    for ax, (title, unit, task, kind, cond, field, stat) in zip(axes.flat, PANELS):
        for (ckey, clabel), (color, marker) in zip(CONDITIONS, SLOTS):
            if ckey not in offsets:
                continue
            if len(offsets) == 1:
                color, marker = SLOTS[[p for p, _ in PROVIDERS].index(provider)]
            pts = []
            for i, (model, _) in enumerate(models):
                vals = values([r for r in rows if r["model_id"] == model and r["condition_run"] == ckey], task, kind, cond, field)
                if len(vals) >= 10:
                    c, lo, hi = bootstrap(vals, stat, rng)
                    pts.append((i + offsets[ckey], c, lo, hi))
                    table.append({"panel": title, "model": model, "condition": clabel, "n": len(vals),
                                  "value": round(c, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2), "unit": unit})
            if pts:
                draw_points(ax, pts, color, marker, connect=ckey == "off" or len(offsets) == 1)
        xlim = (-0.6, len(models) - 0.4)
        ax.set_xlim(*xlim)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([f"{name}\n{dates[m]:%b %y}" for m, name in models], fontsize=8)
        y0, label = base.get(task if kind == "real" and cond == "spec" else "fk")
        style_axis(ax, title, unit, y0, label, xlim)
    handles = [handle(color, marker, clabel) for (ckey, clabel), (color, marker) in zip(CONDITIONS, SLOTS) if ckey in offsets]
    if len(handles) == 1:
        handles = []  # a single series is named in the subtitle
    names = dict(PROVIDERS)
    first, last = models[0][1], models[-1][1]
    if provider == "openai":
        subtitle = (f"{first} → {last} on 30 DROID joint configurations per cell, with the \"don't calculate\" prompt.\n"
                    "Points: median (rate for multiple choice) with 95% bootstrap interval. Lower is better, except multiple choice.")
        footnote = ("GPT-5 runs at minimal effort but used 0 reasoning tokens, so it is shown as reasoning off. Twisted: Panda link "
                    "lengths with joints 5/7 twists swapped. Alien: twist swap on made-up lengths. Joint angles only: no name, no table.")
    else:
        subtitle = (f"{first} → {last} on 30 DROID joint configurations per cell, all told \"DO NOT USE ANY REASONING\".\n"
                    "Points: median (rate for multiple choice) with 95% bootstrap interval. Lower is better, except multiple choice.")
        thinkers = [name for m, name in models if any(r["model_id"] == m and r["effort"] not in ("none", "n/a") for r in rows)]
        tokens = sum((r.get("reasoning_tokens") or 0) > 0 for r in rows if r["model_id"] in dict(models))
        must = (f"{', '.join(thinkers)} must think (lowest effort); " if thinkers else "")
        footnote = (f"{must}{tokens} of {len(rows)} answers used any reasoning tokens.\nTwisted: Panda link lengths with joints 5/7 "
                    "twists swapped. Alien: twist swap on made-up lengths. Joint angles only: no name, no table.")
    finish(fig, f"Panda kinematics across {names[provider]} model generations", subtitle, footnote, handles, out, table)


def compare_figure(rows, base, dates, out: Path) -> None:
    """Both providers on one release-date axis. Each point is a zero-reasoning run: the strict prompt where it
    exists, otherwise reasoning off."""
    rng = np.random.default_rng(0)
    fig, axes = new_figure()
    table = []
    all_dates = [dates[m] for p, _ in PROVIDERS for m, _ in LINEUPS[p] if m in dates]
    pad = dt.timedelta(days=25)
    xlim = (mdates.date2num(min(all_dates) - pad), mdates.date2num(max(all_dates) + pad))
    for ai, (ax, (title, unit, task, kind, cond, field, stat)) in enumerate(zip(axes.flat, PANELS)):
        for (provider, pname), (color, marker) in zip(PROVIDERS, SLOTS):
            pts, labels = [], []
            for model, name in LINEUPS[provider]:
                mrows = [r for r in rows if r["provider"] == provider and r["model_id"] == model]
                for ckey in ("strict", "off"):
                    vals = values([r for r in mrows if r["condition_run"] == ckey], task, kind, cond, field)
                    if len(vals) >= 10:
                        c, lo, hi = bootstrap(vals, stat, rng)
                        x = mdates.date2num(dates[model])
                        pts.append((x, c, lo, hi))
                        labels.append((x, c, name))
                        table.append({"panel": title, "provider": pname, "model": model, "release": str(dates[model]),
                                      "condition": dict(CONDITIONS)[ckey], "n": len(vals), "value": round(c, 2),
                                      "ci_low": round(lo, 2), "ci_high": round(hi, 2), "unit": unit})
                        break
            if pts:
                draw_points(ax, sorted(pts), color, marker, connect=True)
            if ai == 0:  # name the first and the jumping models once, in the first panel
                prefix = {"openai": "GPT-", "anthropic": "", "google": "Gemini "}[provider]
                for x, c, name in labels:
                    for model, mname in LINEUPS[provider]:
                        if mname != name:
                            continue
                        if model in LABEL_AT:
                            dx, dy, ha = LABEL_AT[model]
                            ax.annotate(prefix + name, (x, c), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                                        color=INK2, ha=ha, va="center")
                        elif model in LEADER_AT:
                            tdate, ty, ha = LEADER_AT[model]
                            ax.annotate(prefix + name, (x, c), xytext=(mdates.date2num(dt.date.fromisoformat(tdate)), ty),
                                        textcoords="data", fontsize=7.5, color=INK2, ha=ha, va="center",
                                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8, "shrinkA": 2, "shrinkB": 5})
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        y0, label = base.get(task if kind == "real" and cond == "spec" else "fk")
        style_axis(ax, title, unit, y0, label, xlim)
    handles = [handle(color, marker, pname) for (_, pname), (color, marker) in zip(PROVIDERS, SLOTS)]
    subtitle = ("Every point answers with zero reasoning tokens, on 30 DROID joint configurations per cell, by release date.\n"
                "Points: median (rate for multiple choice) with 95% bootstrap interval. Lower is better, except multiple choice.")
    footnote = ("Prompt: \"don't calculate\" with reasoning off for OpenAI models up to GPT-5.6 Sol; additionally told \"DO NOT USE ANY "
                "REASONING\" for GPT-6 Astra, GPT-6 Sol, GPT-6.1 Sol and all Claude and Gemini models (it did not change GPT-6 Sol).")
    finish(fig, "Panda kinematics: OpenAI vs Claude vs Gemini model generations", subtitle, footnote, handles, out, table)


def zero_reasoning_values(mrows, task, kind, cond, field):
    """A model's values from its zero-reasoning run: the strict prompt where it exists, else reasoning off."""
    for ckey in ("strict", "off"):
        vals = values([r for r in mrows if r["condition_run"] == ckey], task, kind, cond, field)
        if len(vals) >= 10:
            return vals, ckey
    return None, None


def dotplot_figure(rows, base, dates, out: Path) -> None:
    """Every model as a row, grouped by lab in release order; one column per headline metric."""
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in rows:
        by_model[f"{r['provider']}/{r['model_id']}"].append(r)
    entries = []  # (y label, full id or None for a group header)
    for gname, vendor, models in GROUPS:
        present = [(f"{vendor}/{m}" if vendor else m, name) for m, name in models if (f"{vendor}/{m}" if vendor else m) in by_model]
        if present:
            entries.append((gname, None))
            entries += [(f"{name}  ·  {dates[full]:%b %y}", full) for full, name in present]
    rng = np.random.default_rng(0)
    panels = PANELS[:4]
    # Narrow columns: short titles and baseline notes.
    short = ["FK error, real Panda\ngray: constant guess, 24 cm", "Multiple choice, real Panda\ngray: chance, 25%",
             "IK error, real Panda\ngray: DROID's median pose, 23 cm", "Novel arm vs Panda's answer\ngray: constant guess, 24 cm"]
    # All text black, in Google Sans; no figure title (the post or page around it says what it shows).
    plt.rcParams.update({"font.family": OVERVIEW_FONT, "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK, "text.color": BLACK})
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 1.9 + 0.235 * len(entries)), sharey=True, facecolor=SURFACE)
    ys = np.arange(len(entries))[::-1]
    color, marker = SLOTS[0]
    table = []
    for ax, ptitle, (title, unit, task, kind, cond, field, stat) in zip(axes, short, panels):
        ax.set_facecolor(SURFACE)
        for y, (label, full) in zip(ys, entries):
            if full is None:
                continue
            vals, ckey = zero_reasoning_values(by_model[full], task, kind, cond, field)
            if vals is None:
                continue
            c, lo, hi = bootstrap(vals, stat, rng)
            ax.hlines(y, lo, hi, color=color, lw=1.5, alpha=0.8, zorder=3)
            ax.plot(c, y, marker=marker, ms=6.5, color=color, mec=SURFACE, mew=1.5, zorder=4)
            table.append({"panel": title, "model": full, "release": str(dates[full]), "condition": dict(CONDITIONS)[ckey],
                          "n": len(vals), "value": round(c, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2), "unit": unit})
        y0, blabel = base.get(task if kind == "real" and cond == "spec" else "fk")
        ax.axvline(y0, color=MUTED, lw=1.2, zorder=1)
        ax.set_title(ptitle, loc="left", fontsize=9.5, color=BLACK, pad=8)
        ax.set_xlabel(unit)
        ax.set_xlim(0, 102 if unit == "%" else None)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for y, (label, full) in zip(ys, entries):
            if full is None:
                ax.axhline(y - 0.5, color=GRID, lw=0.8, zorder=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for label, _ in entries], fontsize=8)
    for tick, (_, full) in zip(axes[0].get_yticklabels(), entries):
        if full is None:
            tick.set_fontweight("semibold")
    axes[0].set_ylim(-0.7, len(entries) - 0.3)
    fig.subplots_adjust(left=0.19, right=0.99, top=1 - 0.62 / fig.get_figheight(), bottom=0.55 / fig.get_figheight(), wspace=0.08)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {out} and {out.with_suffix('.csv')}")


# Labels for the X post image, placed in the empty band under the data with a leader line to each point.
POST_LEADERS = {"gpt-6-astra": ("2026-08-08", 7.0, "right"), "gpt-6.1-sol": ("2026-09-22", 3.3, "right"),
                "gemini-4-argon": ("2026-11-14", 34.5, "right")}


def post_figure(rows, base, dates, out: Path, since: dt.date = dt.date(2026, 1, 1)) -> None:
    """A 16:9 image for the X post: models released since `since`, two headline panels, large type."""
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75), facecolor=SURFACE)
    panels = [("Hand position error, real Panda", PANELS[0]), ("Different arm: distance from the Panda's answer", PANELS[3])]
    xlim = (mdates.date2num(since - dt.timedelta(days=12)), mdates.date2num(dt.date(2026, 11, 20)))
    table = []
    for ai, (ax, (ptitle, (title, unit, task, kind, cond, field, stat))) in enumerate(zip(axes, panels)):
        ax.set_facecolor(SURFACE)
        for (provider, pname), (color, marker) in zip(PROVIDERS, SLOTS):
            pts = []
            for model, name in LINEUPS[provider]:
                if model not in dates or dates[model] < since:
                    continue
                mrows = [r for r in rows if r["provider"] == provider and r["model_id"] == model]
                vals, ckey = zero_reasoning_values(mrows, task, kind, cond, field)
                if vals is None:
                    continue
                c, lo, hi = bootstrap(vals, stat, rng)
                x = mdates.date2num(dates[model])
                pts.append((x, c, lo, hi))
                table.append({"panel": ptitle, "provider": pname, "model": model, "release": str(dates[model]),
                              "n": len(vals), "value": round(c, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2)})
                if ai == 0 and model in POST_LEADERS:
                    tdate, ty, ha = POST_LEADERS[model]
                    prefix = {"openai": "GPT-", "anthropic": "", "google": "Gemini "}[provider]
                    ax.annotate(prefix + name, (x, c), xytext=(mdates.date2num(dt.date.fromisoformat(tdate)), ty),
                                textcoords="data", fontsize=11, color=INK2, ha=ha, va="center",
                                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9, "shrinkA": 2, "shrinkB": 6})
            if pts:
                draw_points(ax, sorted(pts), color, marker, connect=True)
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        y0, label = base["fk"]
        style_axis(ax, ptitle, unit, y0, label, xlim)  # y-axis fits each panel's own data and intervals
        ax.title.set_fontsize(13.5)
    handles = [handle(color, marker, pname) for (_, pname), (color, marker) in zip(PROVIDERS, SLOTS)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.875), ncol=3, frameon=False, fontsize=11.5)
    fig.text(0.05, 0.965, "Only three models know where a Franka Panda's hand is", fontsize=19, weight="semibold", va="top")
    fig.text(0.05, 0.905, "Models released since January 2026 · zero reasoning tokens · 30 real DROID joint configurations",
             fontsize=12, color=INK2, va="top")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.76, bottom=0.09, wspace=0.16)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {out} and {out.with_suffix('.csv')}")


def main(argv: list[str]) -> None:
    which = argv or ["openai", "anthropic", "google", "compare", "overview", "post"]
    FIGURES.mkdir(exist_ok=True)
    rows, base, dates = load_rows(), baselines(), release_dates()
    if "openai" in which:
        lineup_figure("openai", rows, base, dates, FIGURES / "openai_trend.png")
    if "anthropic" in which:
        lineup_figure("anthropic", rows, base, dates, FIGURES / "claude_trend.png")
    if "google" in which:
        lineup_figure("google", rows, base, dates, FIGURES / "gemini_trend.png")
    if "post" in which:
        post_figure(rows, base, dates, FIGURES / "x_post.png")
    if "overview" in which:
        dotplot_figure(rows, base, dates, FIGURES / "all_models.png")
    if "compare" in which:
        compare_figure(rows, base, dates, FIGURES / "providers_compare.png")


if __name__ == "__main__":
    main(sys.argv[1:])
