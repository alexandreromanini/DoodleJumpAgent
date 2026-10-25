"""Gymnasium environment around the game, headless by default."""
import math

import gymnasium as gym
import numpy as np

from .sim import ACTIONS, ENEMY_TYPES, POWER_UPS, SCREEN_WIDTH, TILE_TYPES, death_cause, load_game

N_TILES = 12
N_ENEMIES = 3
PLAYER_FEATURES = 8
TILE_FEATURES = 4 + len(TILE_TYPES) + 1 + len(POWER_UPS)  # present, dx, dy, vx, type, spent, power-up
ENEMY_FEATURES = 5 + len(ENEMY_TYPES)  # present, dx, dy, vx, vy, type
OBS_SIZE = PLAYER_FEATURES + N_TILES * TILE_FEATURES + N_ENEMIES * ENEMY_FEATURES

TRAIN_SEEDS = (0, 1_000_000_000)  # training episodes draw their game seed from this range
EVAL_SEED_START = 1_000_000_000  # final evaluation
VALIDATION_SEED_START = 1_500_000_000  # checkpoint selection during training


def wrap_dx(dx):
    """Horizontal offset across the wrap-around edges, in [-W/2, W/2)"""
    return (dx + SCREEN_WIDTH / 2) % SCREEN_WIDTH - SCREEN_WIDTH / 2


def live_power_up(tile):
    """The tile's power-up if touching it still does something"""
    pu = tile.power_up
    if pu is None or type(tile).__name__ == "ExplodingTile" or getattr(pu, "expanded", False):
        return None  # exploding tiles never update their power-up; an extended spring is spent
    return pu


class DoodleJumpEnv(gym.Env):
    """One game of Doodle Jump per environment (the game allows one per process).

    Actions: Discrete(6), {left, none, right} x {no shoot, shoot}, held for `frame_skip` frames (one shot).
    Reward: height gained / 100, `death_penalty` on death.
    Ends on death; truncated after `max_frames` or `no_progress_frames` without gaining height.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, frame_skip=4, max_frames=100_000, no_progress_frames=1_200, death_penalty=1.0,
                 render_mode=None):
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.no_progress_frames = no_progress_frames
        self.death_penalty = death_penalty
        self.render_mode = render_mode
        self.game = load_game(render=render_mode == "human")()
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_SIZE,), np.float32)
        self._clock = None
        self.on_frame = None  # called with the env after every game frame (frame recording)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game_seed = seed if seed is not None else int(self.np_random.integers(*TRAIN_SEEDS))
        self.game.reset_game(self.game_seed)
        self.frames = 0
        self.best_frame = 0
        self._enemy_pos = {}
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        left, right, shoot = ACTIONS[int(action)]
        player = self.game.player
        score = player.score
        for i in range(self.frame_skip):
            self.game.step(left, right, shoot and i == 0)
            self.frames += 1
            if self.render_mode == "human":
                self.render()
            if self.on_frame is not None:
                self.on_frame(self)
            if player.dead:
                break
        gained = player.score - score
        if gained > 0:
            self.best_frame = self.frames
        terminated = player.dead
        truncated = not terminated and (self.frames >= self.max_frames
                                        or self.frames - self.best_frame >= self.no_progress_frames)
        reward = gained / 100 - (self.death_penalty if terminated else 0.0)
        return self._obs(), reward, terminated, truncated, self._info()

    def _info(self):
        player = self.game.player
        info = {"score": int(player.score), "frames": self.frames, "seed": self.game_seed}
        if player.dead:
            info["death"] = death_cause(player)
        return info

    def _obs(self):
        game, player = self.game, self.game.player
        obs = np.zeros(OBS_SIZE, np.float32)
        angle = 2 * math.pi * player.x / SCREEN_WIDTH
        obs[:PLAYER_FEATURES] = (player.velocity_y / 20, math.sin(angle), math.cos(angle), player.falling,
                                 player.using_shield, player.using_spring_shoes, player.is_flying(),
                                 player.is_using_booster())
        bottom = player.rect.bottom

        tiles = sorted(game.platforms, key=lambda t: abs(t.rect.top - bottom))[:N_TILES]
        tiles.sort(key=lambda t: t.rect.top)
        base = PLAYER_FEATURES
        for t in tiles:
            name = type(t).__name__
            vx = 0
            if name == "MovingTile":
                vx = t.velocity * t.speed / 4
            elif name == "ShiftingTile" and t.shift:
                vx = math.copysign(1, t.target_x - t.rect.x)
            spent = getattr(t, "fall", False) or getattr(t, "collision", False)  # broken / exploding
            pu = live_power_up(t)
            row = obs[base:base + TILE_FEATURES]
            row[:4] = (1, wrap_dx(t.rect.centerx - player.x) / 320, (t.rect.top - bottom) / 450, vx)
            row[4 + TILE_TYPES.index(name)] = 1
            row[4 + len(TILE_TYPES)] = spent
            if pu is not None:
                row[5 + len(TILE_TYPES) + POWER_UPS.index(type(pu).__name__)] = 1
            base += TILE_FEATURES

        base = PLAYER_FEATURES + N_TILES * TILE_FEATURES
        enemies = sorted(game.all_enemies, key=lambda e: abs(e.rect.centery - player.rect.centery))[:N_ENEMIES]
        positions = {}
        for e in enemies:
            pos = positions[id(e)] = e.rect.center
            prev = self._enemy_pos.get(id(e), pos)
            row = obs[base:base + ENEMY_FEATURES]
            row[:5] = (1, wrap_dx(pos[0] - player.x) / 320, (pos[1] - player.rect.centery) / 450,
                       (pos[0] - prev[0]) / 20, (pos[1] - prev[1]) / 20)
            row[5 + ENEMY_TYPES.index(type(e).__name__)] = 1
            base += ENEMY_FEATURES
        self._enemy_pos = positions
        return obs

    def render(self):
        import pygame

        if self.render_mode is None:
            return None
        self.game.draw()
        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.game.screen), (1, 0, 2))
        if pygame.event.get(pygame.QUIT):
            raise SystemExit("window closed")
        pygame.event.pump()
        if self._clock is None:
            self._clock = pygame.time.Clock()
        self._clock.tick(self.metadata["render_fps"])
        return None

