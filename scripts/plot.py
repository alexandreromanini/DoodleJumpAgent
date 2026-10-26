"""Figures for the README from the evaluation CSVs and a training run.

    python scripts/plot.py --run runs/ppo_a
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference data-viz palette, light mode: categorical slots in fixed order, text and surface tokens
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AGENTS = [("random", "Aléatoire"), ("planner", "Planificateur"), ("ppo", "PPO")]
COLOR = {"random": SERIES[0], "planner": SERIES[1], "ppo": SERIES[2]}
ENDS = [("fall", "Chute"), ("monster", "Monstre"), ("blackhole", "Trou noir"), ("ufo", "OVNI"),
        ("truncated", "Bloqué (20 s sans monter)"), ("frozen", "Jeu gelé")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def thousands(x, _=None):
    return f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def scores_figure(results, out):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    rng = np.random.default_rng(0)
    names = [a for a in AGENTS if a[0] in results]
    for i, (key, label) in enumerate(names):
        scores = np.array([int(r["score"]) for r in results[key]])
        y = len(names) - 1 - i
        ax.scatter(scores, y + rng.uniform(-0.18, 0.18, scores.size), s=14, color=COLOR[key], alpha=0.55,
                   edgecolors="none", zorder=3)
        ax.boxplot(scores, positions=[y], vert=False, widths=0.5, showfliers=False, zorder=2,
                   medianprops={"color": INK, "linewidth": 2}, boxprops={"color": INK_2},
                   whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        ax.annotate(f"moyenne {scores.mean():,.0f}".replace(",", " "), (scores.max(), y),
                    xytext=(10, 0), textcoords="offset points", va="center", color=INK_2, fontsize=10)
    ax.set_yticks(range(len(names)), [label for _, label in reversed(names)])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(thousands)
    ax.set_xlabel("Score (hauteur atteinte, échelle log)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Scores sur {len(next(iter(results.values())))} parties d'évaluation (seeds jamais vues)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def ends_figure(results, out):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    names = [a for a in AGENTS if a[0] in results]
    for i, (key, label) in enumerate(names):
        rows = results[key]
        left = 0
        for j, (end, end_label) in enumerate(ENDS):
            share = sum(r["end"] == end for r in rows) / len(rows)
            if share:
                ax.barh(len(names) - 1 - i, share - 0.004, left=left, height=0.6, color=SERIES[j],
                        label=end_label if i == 0 or end_label not in ax.get_legend_handles_labels()[1] else None)
                if share >= 0.07:
                    ax.text(left + share / 2, len(names) - 1 - i, f"{share:.0%}", ha="center", va="center",
                            color=INK, fontsize=9)
            left += share
    ax.set_yticks(range(len(names)), [label for _, label in reversed(names)])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(lbl) for _, lbl in ENDS if lbl in labels]
    ax.legend([handles[k] for k in order], [labels[k] for k in order], ncol=3, frameon=False, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Comment les parties se terminent")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def learning_figure(run, results, out):
    lines = (run / "monitor.csv").read_text().splitlines()[1:]
    monitor = [r for r in csv.DictReader(lines) if r["frames"]]  # a run stopped mid-write leaves a partial row
    steps = np.cumsum([float(r["l"]) for r in monitor])
    scores = np.array([float(r["score"]) for r in monitor])
    window = max(len(scores) // 100, 20)
    smooth = np.convolve(scores, np.ones(window) / window, mode="valid")
    validation = read(run / "validation.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps[window - 1:] / 1e6, smooth, color=COLOR["ppo"], linewidth=2, alpha=0.45,
            label=f"PPO, parties d'entraînement (moyenne glissante sur {window})")
    ax.plot([int(r["timesteps"]) / 1e6 for r in validation], [float(r["mean"]) for r in validation],
            color=COLOR["ppo"], linewidth=2, marker="o", markersize=5, label="PPO, validation (politique gloutonne)")
    for key in ("random", "planner"):
        if key in results:
            mean = statistics.mean(int(r["score"]) for r in results[key])
            ax.axhline(mean, color=COLOR[key], linewidth=2, linestyle="--")
            ax.annotate(dict(AGENTS)[key] + f" ({thousands(mean)})", (0, mean), xytext=(4, 4),
                        textcoords="offset points", color=INK_2, fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(thousands)
    ax.set_xlabel("Pas d'entraînement (millions, 4 frames par pas)")
    ax.set_ylabel("Score moyen (log)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Courbe d'apprentissage de PPO")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def iterations_figure(versions, out):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    labels = [v[0] for v in versions]
    means = [statistics.mean(int(r["score"]) for r in v[1]) for v in versions]
    ax.bar(range(len(means)), means, color=COLOR["planner"], width=0.6, zorder=3)
    for i, m in enumerate(means):
        ax.annotate(thousands(m), (i, m), xytext=(0, 4), textcoords="offset points", ha="center", color=INK,
                    fontsize=10)
    ax.set_xticks(range(len(means)), labels, fontsize=9)
    ax.yaxis.set_major_formatter(thousands)
    ax.set_ylabel("Score moyen")
    ax.grid(axis="x", visible=False)
    ax.set_title("Itérations du planificateur (24 seeds de validation)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="PPO training run directory")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    res = args.results.resolve()
    results = {k: read(res / f"{k}.csv") for k, _ in AGENTS if (res / f"{k}.csv").exists()}
    if results:
        scores_figure(results, res / "scores.png")
        ends_figure(results, res / "ends.png")
    if args.run:
        learning_figure(args.run.resolve(), results, res / "learning_curve.png")
    steps = (res / "dev" / "iterations.csv")
    if steps.exists():
        versions = [(r["label"], read(res / "dev" / r["file"])) for r in read(steps)]
        iterations_figure(versions, res / "planner_iterations.png")
    print("figures written to", res)


if __name__ == "__main__":
    main()
