"""Train PPO (Stable-Baselines3) on the headless game, one game per subprocess.

    python scripts/train.py --out runs/ppo --seed 0

Writes monitor.csv (every training game), validation.csv (greedy policy on validation seeds), progress.csv
(PPO losses), best_model.zip (best validation mean) and final_model.zip.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, help="override total_timesteps")
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

    model = PPO("MlpPolicy", venv, learning_rate=p["learning_rate"], n_steps=p["n_steps"],
                batch_size=p["batch_size"], n_epochs=p["n_epochs"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
                clip_range=p["clip_range"], ent_coef=p["ent_coef"],
                policy_kwargs={"net_arch": list(p["net_arch"])}, seed=args.seed, device="cpu", verbose=0)
    model.set_logger(configure(str(out), ["csv"]))
    model.learn(total_timesteps=args.timesteps or p["total_timesteps"], callback=Validation())
    model.save(out / "final_model")
    venv.close()


if __name__ == "__main__":
    main()
