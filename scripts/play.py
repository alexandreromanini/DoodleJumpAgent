"""Watch an agent play in a window, game after game (close the window to stop).

    python scripts/play.py --agent planner --seed 1000000000
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=("random", "planner", "ppo"), required=True)
    parser.add_argument("--model", type=Path, help="PPO model (.zip)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--seed", type=int, default=1_000_000_000, help="seed of the first game")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    from doodle_agent.agents import load_config, make, play_episode

    agent, env = make(args.agent, load_config(args.config.resolve()),
                      model=args.model.resolve() if args.model else None, render_mode="human")
    for seed in range(args.seed, args.seed + args.games):
        print(play_episode(agent, env, seed))


if __name__ == "__main__":
    main()
