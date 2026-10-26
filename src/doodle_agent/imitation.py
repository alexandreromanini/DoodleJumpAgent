"""Imitation of the planner (behaviour cloning with DAgger), to start PPO from a policy that already plays.

Each round plays games with the current policy (the planner itself in round 0, then the student) and labels every
visited state with the planner's action, so the student also learns how to recover from its own mistakes.
"""
import multiprocessing as mp

import numpy as np

_worker = {}


def _init(config, frame_skip, max_frames, model_path):
    from .agents import load_config
    from .env import DoodleJumpEnv
    from .planner import Planner

    cfg = load_config(config)
    params = {k: v for k, v in cfg["planner"].items() if k != "frame_skip"}
    _worker["env"] = DoodleJumpEnv(frame_skip=frame_skip, max_frames=max_frames,
                                   no_progress_frames=cfg["env"]["no_progress_frames"],
                                   death_penalty=cfg["ppo"]["death_penalty"])
    _worker["planner"] = Planner(**params)
    _worker["model"] = None
    if model_path is not None:
        from stable_baselines3 import PPO

        _worker["model"] = PPO.load(model_path, device="cpu")


def _play(seed):
    env, planner, model = _worker["env"], _worker["planner"], _worker["model"]
    obs, _ = env.reset(seed=int(seed))
    planner.reset()
    observations, labels, rewards = [], [], []
    done = False
    while not done:
        label = planner.act(env.game)
        action = label if model is None else int(model.predict(obs, deterministic=True)[0])
        observations.append(obs)
        labels.append(label)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return np.array(observations), np.array(labels), np.array(rewards, dtype=np.float32), info["score"]


def collect(seeds, config, frame_skip, max_frames, model_path=None, workers=None):
    """Play one game per seed; returns per-game (observations, planner labels, rewards, score)"""
    with mp.get_context("spawn").Pool(workers or mp.cpu_count(), _init,
                                      (config, frame_skip, max_frames, model_path)) as pool:
        return pool.map(_play, seeds)


def discounted(rewards, gamma):
    out = np.empty_like(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def fit(policy, observations, actions, returns, epochs, learning_rate, batch_size, seed=0):
    """Supervised training of an SB3 actor-critic: cross-entropy on the planner's actions, MSE on the returns"""
    import torch

    rng = np.random.default_rng(seed)
    obs = torch.as_tensor(observations)
    act = torch.as_tensor(actions)
    ret = torch.as_tensor(returns)
    optimiser = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = torch.as_tensor(rng.permutation(len(obs)))
        for start in range(0, len(obs), batch_size):
            i = order[start:start + batch_size]
            values, log_prob, _ = policy.evaluate_actions(obs[i], act[i])
            loss = -log_prob.mean() + 0.5 * torch.nn.functional.mse_loss(values.flatten(), ret[i])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    policy.set_training_mode(False)
    with torch.no_grad():
        correct = sum((policy.get_distribution(obs[i:i + 65536]).distribution.probs.argmax(1) == act[i:i + 65536])
                      .sum().item() for i in range(0, len(obs), 65536))
    return correct / len(obs)
