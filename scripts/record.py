"""Record a game as a GIF, headless.

    python scripts/record.py --agent planner --seed 1000000000 --out assets/demo.gif
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=("random", "planner", "ppo"), required=True)
    parser.add_argument("--model", type=Path, help="PPO model (.zip)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--seed", type=int, default=1_000_000_000)
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "demo.gif")
    parser.add_argument("--start", type=int, default=0, help="first game frame recorded")
    parser.add_argument("--seconds", type=float, default=15.0, help="game time recorded (60 frames/s)")
    parser.add_argument("--fps", type=int, default=30, help="GIF frame rate, the game runs at 60")
    parser.add_argument("--scale", type=float, default=0.4)
    args = parser.parse_args()

    from PIL import Image
    from doodle_agent.agents import load_config, make

    out = args.out.resolve()
    agent, env = make(args.agent, load_config(args.config.resolve()),
                      model=args.model.resolve() if args.model else None, render_mode="rgb_array")
    every, end = 60 // args.fps, args.start + int(args.seconds * 60)
    images = []

    def grab(env):
        if args.start <= env.frames < end and (env.frames - args.start) % every == 0:
            image = Image.fromarray(env.render())
            images.append(image.resize((round(image.width * args.scale), round(image.height * args.scale)),
                                       Image.LANCZOS))

    env.on_frame = grab
    obs, info = env.reset(seed=args.seed)
    agent.reset()
    done = False
    while not done and env.frames < end:
        obs, _, terminated, truncated, info = env.step(agent.act(env, obs))
        done = terminated or truncated
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=1000 // args.fps, loop=0, optimize=True)
    print(f"{len(images)} frames, score {info['score']} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
