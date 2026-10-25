"""Model-based planner: predicts the game with its own physics and steers to the best reachable landing.

It re-plans every frame (model predictive control). The model replays the game's arithmetic exactly:
semi-implicit Euler with gravity 0.4; a camera that pins the player's rect at y=390 and scrolls everything else
by int()-truncated steps; tile collisions checked before the player moves in a frame, enemy collisions after;
x on a circle of 129 positions 5 px apart (moving right from 640 wraps to 0).
"""
from dataclasses import dataclass

import numpy as np

from .env import live_power_up
from .sim import ACTIONS, GRAVITY, PLAYER_H, PLAYER_TOP, PLAYER_W, SCREEN_HEIGHT, SCREEN_WIDTH, SPEED

N_POS = SCREEN_WIDTH // SPEED + 1
HALF_W = PLAYER_W // 2
TILE_W, TILE_H = 57, 15
SCROLL_LIMIT = SCREEN_HEIGHT // 2 - PLAYER_H  # the camera scrolls while the player's world y is above this

JUMP = -15
BOOST = {"Spring": -23, "Trampoline": -30, "SpringShoes": -23, "Propeller": -55, "Jetpack": -65}
NEEDS_FALLING = {"Spring", "Trampoline"}  # the other power-ups trigger on any contact, even going up
ONE_USE = {"DisappearingTile", "ExplodingTile"}


@dataclass
class Trajectory:
    """Player state after frames k = 0..H if nothing is hit: `scroll[k]` is how far everything else has moved
    up the screen, `falling[k]` the flag collisions read during frame k+1, `y[k]` the world y (up is negative)."""
    scroll: np.ndarray
    falling: np.ndarray
    y: np.ndarray


def predict(y, v, prev, falling, horizon):
    scroll, falls, ys = [0], [falling], [y]
    total = 0
    for _ in range(horizon):
        v += GRAVITY
        y += v
        if v > GRAVITY and not falling:
            falling = True
        elif v < GRAVITY:
            falling = False
        if y < SCROLL_LIMIT:
            diff = int((y - 450) - prev)
            prev = int(y - 450)
            total += diff
        scroll.append(total)
        falls.append(falling)
        ys.append(y)
    return Trajectory(np.array(scroll), np.array(falls), np.array(ys))


def circular_distance(a, lo, hi):
    """Moves from position a into [lo, hi] on the x circle (vectorised over lo, hi)"""
    d_lo = np.abs(lo - a)
    d_hi = np.abs(hi - a)
    d = np.minimum(np.minimum(d_lo, N_POS - d_lo), np.minimum(d_hi, N_POS - d_hi))
    return np.where((lo <= a) & (a <= hi), 0, d)


def tile_lefts(tile, frames):
    """rect.x of a tile at the collision check of frames 0..frames (moving tiles move before the check)"""
    name = type(tile).__name__
    x = tile.rect.x
    lefts = [x]
    if name == "MovingTile":
        vel = tile.velocity
        for _ in range(frames):
            x += vel * tile.speed
            if x + TILE_W > tile.max_right:
                x, vel = tile.max_right - TILE_W, -vel
            elif x < tile.max_left:
                x, vel = tile.max_left, -vel
            lefts.append(x)
    elif name == "ShiftingTile" and tile.shift:
        target, shifting = tile.target_x, True
        for _ in range(frames):
            if shifting:
                x += SPEED if x < target else -SPEED if x > target else 0
                shifting = not (target - 5 <= x <= target + 5)
            lefts.append(x)
    else:
        lefts *= frames + 1
    return np.array(lefts)


@dataclass
class Target:
    """Something to bounce on: a tile, or a power-up sitting on one (offsets follow the tile)"""
    tile: object
    kind: str
    dx: int  # offset from the tile's rect.x
    dy: int  # offset from the tile's rect.y
    width: int
    height: int
    jump: int
    needs_falling: bool


@dataclass
class Landing:
    target: Target
    frame: int  # collision check frame (k - 1 moves are available before it)
    lo: int  # x positions that make contact at that frame
    hi: int
    height: float  # world height at contact, higher is better


def targets_of(tiles):
    out = []
    for tile in tiles:
        name = type(tile).__name__
        if name != "BrokenTile" and not (name == "ExplodingTile" and tile.collision):
            out.append(Target(tile, name, 0, 0, TILE_W, TILE_H, JUMP, True))
        pu = live_power_up(tile)
        if pu is not None and type(pu).__name__ in BOOST:
            pname = type(pu).__name__
            out.append(Target(tile, pname, pu.rect.x - tile.rect.x, pu.rect.y - tile.rect.y, pu.rect.width,
                              pu.rect.height, BOOST[pname], pname in NEEDS_FALLING))
    return out


def find_landings(traj, x_node, targets, tops, lefts, start=0):
    """Earliest reachable contact with each target along `traj`. `tops[tile]` is the tile's rect.y at frame
    `start` of the trajectory, `lefts[tile][start + k]` its rect.x at check frame k."""
    found = []
    shifted = traj.scroll[:-1]
    for t in targets:
        y = tops[id(t.tile)] + t.dy - shifted
        ok = (y > PLAYER_TOP - t.height) & (y < PLAYER_TOP + PLAYER_H)
        if t.needs_falling:
            ok &= traj.falling[:-1]
        ks = np.flatnonzero(ok) + 1
        if not ks.size:
            continue
        seq = lefts[id(t.tile)]
        left = seq[np.minimum(start + ks, len(seq) - 1)] + t.dx
        lo = np.maximum(-((HALF_W - 1 - left) // SPEED), 0)
        hi = np.minimum((left + t.width + HALF_W - 1) // SPEED, N_POS - 1)
        reach = (lo <= hi) & (circular_distance(x_node, lo, hi) <= ks - 1)
        if reach.any():
            i = int(np.argmax(reach))
            found.append(Landing(t, int(ks[i]), int(lo[i]), int(hi[i]), float(-traj.y[ks[i] - 1])))
    return found


class Planner:
    def __init__(self, horizon=160, dead_end_penalty=1000.0, exploding_penalty=50.0, boost_weight=1.0,
                 safety_frames=20):
        self.horizon = horizon
        self.dead_end_penalty = dead_end_penalty
        self.exploding_penalty = exploding_penalty
        self.boost_weight = boost_weight
        self.safety_frames = safety_frames
        self._jumps = {}
        self.reset()

    def reset(self):
        self._enemy_pos = {}
        self._last_prev = None
        self.target = None

    def _jump_shape(self, jump):
        """Trajectory of a fresh jump, relative to the contact frame (the int() truncation can differ by 1 px)"""
        if jump not in self._jumps:
            self._jumps[jump] = predict(0.0, float(jump), -450, True, self.horizon)
        return self._jumps[jump]

    def score(self, landing):
        t = landing.target
        value = landing.height + self.boost_weight * (t.jump ** 2 - JUMP ** 2) / (2 * GRAVITY)
        if t.kind == "ExplodingTile":
            value -= self.exploding_penalty
        return value

    def has_continuation(self, landing, traj, x_node, targets, tops, lefts):
        """After this bounce, is there anything to land on next? Only checked for short jumps: the tiles a big
        boost leads to do not exist yet."""
        t = landing.target
        if t.jump < BOOST["Spring"]:
            return True
        k = landing.frame
        x = min(max(x_node, landing.lo), landing.hi) if landing.lo <= landing.hi else x_node
        shift = traj.scroll[k - 1]
        nxt_tops = {key: top - shift for key, top in tops.items()}
        rest = [o for o in targets if not (o.tile is t.tile and t.kind in ONE_USE)]
        return bool(find_landings(self._jump_shape(t.jump), x, rest, nxt_tops, lefts, start=k - 1))

    def act(self, game):
        player = game.player
        x_node = round(player.x / SPEED) % N_POS
        tiles = list(game.platforms)
        traj = predict(player.y, player.velocity_y, player.previous_y_difference, player.falling, self.horizon)
        targets = targets_of(tiles)
        tops = {id(t): t.rect.y for t in tiles}
        lefts = {id(t): tile_lefts(t, 2 * self.horizon + 1) for t in tiles}

        options = find_landings(traj, x_node, targets, tops, lefts)
        options.sort(key=self.score, reverse=True)
        best = next((o for o in options if self.has_continuation(o, traj, x_node, targets, tops, lefts)),
                    options[0] if options else None)
        self.target = best

        desired = 0
        if best is not None:
            centre = (best.lo + best.hi) // 2
            if circular_distance(x_node, centre, centre) <= best.frame - 1:
                desired = self._direction(x_node, centre, centre)
            else:
                desired = self._direction(x_node, best.lo, best.hi)
        move = self._safe_move(game, traj, desired)
        return ACTIONS.index((move == -1, move == 1, self._should_shoot(game)))

    @staticmethod
    def _direction(a, lo, hi):
        if lo <= a <= hi:
            return 0
        right = min((b - a) % N_POS for b in (lo, hi))
        left = min((a - b) % N_POS for b in (lo, hi))
        return 1 if right <= left else -1

    def _enemy_tracks(self, game):
        """Enemies with their own velocity, estimated from the last frame net of the camera scroll"""
        prev_diff = game.player.previous_y_difference
        scrolled = 0 if self._last_prev is None else prev_diff - self._last_prev
        self._last_prev = prev_diff
        tracks, positions = [], {}
        for e in game.all_enemies:
            pos = positions[id(e)] = (e.rect.x, e.rect.y)
            last = self._enemy_pos.get(id(e))
            vel = (0, 0) if last is None else (pos[0] - last[0], pos[1] - last[1] + scrolled)
            tracks.append((e, pos, vel))
        self._enemy_pos = positions
        return tracks

    def _safe_move(self, game, traj, desired):
        """Veto a move that runs into a deadly enemy within `safety_frames`, holding it and bouncing nowhere"""
        player = game.player
        tracks = self._enemy_tracks(game)
        if not tracks or player.using_shield or player.is_flying():
            return desired

        def first_hit(move):
            x = player.x
            for k in range(1, self.safety_frames + 1):
                x += SPEED * move
                x = 0 if x > SCREEN_WIDTH else SCREEN_WIDTH if x < 0 else x
                px = x - HALF_W
                for e, (ex, ey), (vx, vy) in tracks:
                    ex_k, ey_k = ex + vx * k, ey + vy * k - traj.scroll[k]
                    if (px < ex_k + e.rect.width and px + PLAYER_W > ex_k
                            and PLAYER_TOP < ey_k + e.rect.height and PLAYER_TOP + PLAYER_H > ey_k
                            and (type(e).__name__ == "Blackhole" or not traj.falling[k])):
                        return k
            return None

        hits = {m: first_hit(m) for m in (desired, 0, 1, -1)}
        if hits[desired] is None:
            return desired
        safe = [m for m in (0, 1, -1) if hits[m] is None]
        return safe[0] if safe else max(hits, key=hits.get)

    @staticmethod
    def _should_shoot(game):
        player = game.player
        if player.is_flying() or game.frame % 4:
            return False
        cx = player.rect.centerx
        return any(type(e).__name__ != "Blackhole" and e.rect.left - 6 <= cx <= e.rect.right + 6
                   and e.rect.bottom < player.rect.top for e in game.all_enemies)
