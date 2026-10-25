"""Evaluate an agent on fixed seeds that training never uses, one game per worker process.

    python scripts/evaluate.py --agent planner
    python scripts/evaluate.py --agent ppo --model runs/ppo/best_model.zip
"""
import argparse
import csv
import multiprocessing as mp
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _init(agent, config, model):
    global AGENT, ENV
    from doodle_agent.agents import load_config, make

    AGENT, ENV = make(agent, load_config(config), model=model)


def _play(seed):
    from doodle_agent.agents import play_episode

    return play_episode(AGENT, ENV, seed)


def summarize(rows):
    scores = [r["score"] for r in rows]
    return {"games": len(scores), "mean": statistics.mean(scores), "median": statistics.median(scores),
            "std": statistics.pstdev(scores), "min": min(scores), "max": max(scores),
            "ends": dict(Counter(r["end"] for r in rows).most_common())}


def main():
    from doodle_agent.agents import load_config
    from doodle_agent.env import EVAL_SEED_START

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=("random", "planner", "ppo"), required=True)
    parser.add_argument("--model", type=Path, help="PPO model (.zip)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--out", type=Path, help="CSV of per-game results (default results/<agent>.csv)")
    args = parser.parse_args()

    config = args.config.resolve()
    episodes = args.episodes or load_config(config)["evaluate"]["episodes"]
    out = (args.out or ROOT / "results" / f"{args.agent}.csv").resolve()
    model = args.model.resolve() if args.model else None
    seeds = range(args.seed_start, args.seed_start + episodes)

    rows = []
    with mp.get_context("spawn").Pool(min(args.workers, episodes), _init, (args.agent, config, model)) as pool:
        for row in pool.imap_unordered(_play, seeds):
            rows.append(row)
            print(f"[{len(rows):>3}/{episodes}] seed {row['seed']}: {row['score']} ({row['end']}, "
                  f"{row['frames']} frames)", flush=True)
    rows.sort(key=lambda r: r["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    s = summarize(rows)
    print(f"\n{args.agent}: mean {s['mean']:.0f}  median {s['median']:.0f}  std {s['std']:.0f}  "
          f"min {s['min']}  max {s['max']}  over {s['games']} games\nends: {s['ends']}\nwrote {out}")


if __name__ == "__main__":
    main()
