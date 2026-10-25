"""Agents behind one interface: reset() at the start of a game, act(env, obs) -> action."""
import tomllib
from pathlib import Path

import numpy as np

from .env import DoodleJumpEnv

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, "rb") as f:
        return tomllib.load(f)


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env, obs):
        return int(self.rng.integers(env.action_space.n))


class PlannerAgent:
    def __init__(self, **params):
        from .planner import Planner

        self.planner = Planner(**params)

    def reset(self):
        self.planner.reset()

    def act(self, env, obs):
        return self.planner.act(env.game)


class PPOAgent:
    def __init__(self, model_path):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device="cpu")

    def reset(self):
        pass

    def act(self, env, obs):
        return int(self.model.predict(obs, deterministic=True)[0])


def make(name, config, model=None, render_mode=None, max_frames=None):
    """Build an agent and an environment with the frame skip it was designed for"""
    section = config[name]
    if name == "random":
        agent = RandomAgent()
    elif name == "planner":
        agent = PlannerAgent(**{k: v for k, v in section.items() if k != "frame_skip"})
    elif name == "ppo":
        if model is None:
            raise ValueError("the ppo agent needs --model")
        agent = PPOAgent(model)
    else:
        raise ValueError(f"unknown agent {name}")
    env = DoodleJumpEnv(frame_skip=section["frame_skip"], death_penalty=config["ppo"]["death_penalty"],
                        max_frames=max_frames or config["env"]["max_frames"],
                        no_progress_frames=config["env"]["no_progress_frames"], render_mode=render_mode)
    return agent, env


def play_episode(agent, env, seed):
    obs, info = env.reset(seed=seed)
    agent.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(agent.act(env, obs))
        done = terminated or truncated
    return {"seed": seed, "score": info["score"], "frames": info["frames"],
            "end": info.get("death", "truncated")}
