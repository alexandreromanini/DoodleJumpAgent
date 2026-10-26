"""Train PPO (Stable-Baselines3) on the headless game, one game per subprocess.

    python scripts/train.py --out runs/ppo --seed 0                       # from scratch
    python scripts/train.py --out runs/ppo_imitation --seed 0 --imitation  # imitate the planner first, then PPO

Writes monitor.csv (every training game), validation.csv (greedy policy on validation seeds), progress.csv
(PPO losses), best_model.zip (best validation mean) and final_model.zip; with --imitation also imitation.csv
and one model per imitation round.
"""
import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_factory(cfg, max_frames):
    def make():
        from doodle_agent.env import DoodleJumpEnv

        return DoodleJumpEnv(frame_skip=cfg["ppo"]["frame_skip"], max_frames=max_frames,
                             no_progress_frames=cfg["env"]["no_progress_frames"],
                             death_penalty=cfg["ppo"]["death_penalty"])
    return make


def imitate(model, cfg, config_path, out, seed):
    """Behaviour cloning + DAgger rounds on the planner's labels (see doodle_agent.imitation)"""
    import numpy as np

    from doodle_agent.env import TRAIN_SEEDS
    from doodle_agent.imitation import collect, discounted, fit

    im, p = cfg["imitation"], cfg["ppo"]
    rng = np.random.default_rng(seed)
    games, model_path = [], None
    with open(out / "imitation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "player", "games", "player_mean_score", "samples", "accuracy"])
        for round_ in range(im["dagger_rounds"] + 1):
            seeds = rng.integers(*TRAIN_SEEDS, im["games_per_round"])
            played = collect(seeds, config_path, p["frame_skip"], p["train_max_frames"], model_path)
            games += played
            obs = np.concatenate([g[0] for g in games])
            accuracy = fit(model.policy, obs, np.concatenate([g[1] for g in games]),
                           np.concatenate([discounted(g[2], p["gamma"]) for g in games]),
                           im["epochs"], im["bc_learning_rate"], im["bc_batch_size"], seed=seed + round_)
            player_score = statistics.mean(g[3] for g in played)
            writer.writerow([round_, "planner" if model_path is None else "student", len(played), player_score,
                             len(obs), accuracy])
            f.flush()
            print(f"imitation round {round_}: {len(obs)} samples, accuracy {accuracy:.3f}, "
                  f"{'planner' if model_path is None else 'student'} played {player_score:.0f}", flush=True)
            model_path = out / f"imitation_{round_}.zip"
            model.save(model_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, help="override total_timesteps")
    parser.add_argument("--imitation", action="store_true", help="start from a policy imitating the planner")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from doodle_agent.agents import load_config
    from doodle_agent.env import VALIDATION_SEED_START

    cfg = load_config(args.config.resolve())
    p = cfg["ppo"]
    (out / "config.toml").write_bytes(args.config.resolve().read_bytes())

    venv = SubprocVecEnv([env_factory(cfg, p["train_max_frames"])] * p["n_envs"])
    venv = VecMonitor(venv, str(out / "monitor.csv"), info_keywords=("score", "frames"))
    validation_env = env_factory(cfg, p["eval_max_frames"])()  # the training games live in subprocesses

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0
            self.best = -1.0
            self.file = open(out / "validation.csv", "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(["timesteps", "mean", "median", "max"])

        def _on_step(self):
            if self.num_timesteps < self.next_eval:
                return True
            self.next_eval += p["eval_every"]
            scores = []
            for i in range(p["eval_episodes"]):
                obs, info = validation_env.reset(seed=VALIDATION_SEED_START + i)
                done = False
                while not done:
                    action = int(self.model.predict(obs, deterministic=True)[0])
                    obs, _, terminated, truncated, info = validation_env.step(action)
                    done = terminated or truncated
                scores.append(info["score"])
            mean = statistics.mean(scores)
            self.writer.writerow([self.num_timesteps, mean, statistics.median(scores), max(scores)])
            self.file.flush()
            if mean > self.best:
                self.best = mean
                self.model.save(out / "best_model")
            print(f"validation @ {self.num_timesteps}: mean {mean:.0f}, best {self.best:.0f}", flush=True)
            return True

    im = cfg["imitation"]
    tuned = {k: im[k] for k in ("learning_rate", "ent_coef", "clip_range")} if args.imitation else {}
    model = PPO("MlpPolicy", venv, n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
                gamma=p["gamma"], gae_lambda=p["gae_lambda"], policy_kwargs={"net_arch": list(p["net_arch"])},
                seed=args.seed, device="cpu", verbose=0,
                **{k: tuned.get(k, p[k]) for k in ("learning_rate", "ent_coef", "clip_range")})
    if args.imitation:
        imitate(model, cfg, args.config.resolve(), out, args.seed)
    model.set_logger(configure(str(out), ["csv"]))
    model.learn(total_timesteps=args.timesteps or (im if args.imitation else p)["total_timesteps"],
                callback=Validation())
    model.save(out / "final_model")
    venv.close()


if __name__ == "__main__":
    main()
