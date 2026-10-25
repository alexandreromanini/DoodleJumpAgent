import numpy as np
import pytest

from doodle_agent.env import OBS_SIZE, DoodleJumpEnv


@pytest.fixture(scope="module")
def env():
    return DoodleJumpEnv(frame_skip=4, max_frames=6_000)


def rollout(env, seed, n_steps, actions_seed=0):
    rng = np.random.default_rng(actions_seed)
    obs, info = env.reset(seed=seed)
    observations, rewards, infos = [obs], [], [info]
    for _ in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(6)))
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
        if terminated or truncated:
            break
    return np.array(observations), rewards, infos


def test_reset_gives_a_fresh_game(env):
    obs, info = env.reset(seed=7)
    assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info == {"score": 0, "frames": 0, "seed": 7, "leaked_tiles": 0}
    assert not env.game.player.dead and len(env.game.platforms) > 20


def test_same_seed_and_actions_replay_identically(env):
    a = rollout(env, seed=11, n_steps=400)
    b = rollout(env, seed=11, n_steps=400)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


def test_different_seeds_give_different_games(env):
    a, _, _ = rollout(env, seed=1, n_steps=0)
    b, _, _ = rollout(env, seed=2, n_steps=0)
    assert not np.array_equal(a, b)


def test_step_rewards_height_gained(env):
    _, rewards, infos = rollout(env, seed=3, n_steps=300)
    for reward, before, after in zip(rewards, infos, infos[1:]):
        if "death" not in after:
            assert reward == pytest.approx((after["score"] - before["score"]) / 100, abs=0.01)
    assert sum(rewards) > 0


def test_episode_ends_on_death():
    patient = DoodleJumpEnv(frame_skip=4, max_frames=10**7, no_progress_frames=10**7)
    _, rewards, infos = rollout(patient, seed=5, n_steps=10**6)
    assert patient.game.player.dead
    assert infos[-1]["death"] in {"fall", "monster", "ufo", "blackhole"}
    assert rewards[-1] < 0


def test_truncates_at_frame_cap():
    short = DoodleJumpEnv(frame_skip=4, max_frames=40)
    short.reset(seed=0)
    for step in range(1, 100):
        _, _, terminated, truncated, info = short.step(1)
        if terminated or truncated:
            break
    assert truncated and not terminated and info["frames"] == 40


def test_a_saturated_spawn_area_freezes_the_game_instead_of_hanging(env):
    env.reset(seed=0)
    game = env.game
    assert game.has_free_spot(60, 320, -920, -450)
    tile = game.tile_objects[0]
    for x in range(0, 640, 100):  # tiles 100 px apart: no spot is 120 px away from all of them
        for y in range(-1000, 100, 100):
            tile(game, x, y)
    assert not game.has_free_spot(60, 320, -920, -450)
    tile.total = 20  # pretend tiles fell off the screen so the game spawns new ones
    _, _, terminated, truncated, info = env.step(1)
    assert truncated and not terminated and info["frozen"]
