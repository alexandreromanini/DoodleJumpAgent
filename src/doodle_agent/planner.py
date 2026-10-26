"""Model-based planner: predicts the game with its own physics and steers to the best reachable landing.

It re-plans every frame (model predictive control). The model replays the game's arithmetic exactly:
semi-implicit Euler with gravity 0.4; a camera that pins the player's rect at y=390 and scrolls everything else
by int()-truncated steps; tile collisions checked before the player moves in a frame, enemy collisions after;
x on a circle of 129 positions 5 px apart (moving right from 640 wraps to 0).

Each frame it lists every (target, frame) contact the predicted height allows, follows the paths that lead to
the most promising landings, and keeps the one whose first real contact is best and safe.
"""
import math
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


def tile_lefts(tiles, frames):
    """rect.x of each tile at the collision check of frames 0..frames (moving tiles move before the check)"""
    lefts = np.empty((len(tiles), frames + 1), dtype=np.int64)
    for i, tile in enumerate(tiles):
        name = type(tile).__name__
        x = tile.rect.x
        lefts[i] = x
        if name == "MovingTile":
            vel = tile.velocity
            for k in range(1, frames + 1):
                x += vel * tile.speed
                if x + TILE_W > tile.max_right:
                    x, vel = tile.max_right - TILE_W, -vel
                elif x < tile.max_left:
                    x, vel = tile.max_left, -vel
                lefts[i, k] = x
        elif name == "ShiftingTile" and tile.shift:
            target, shifting = tile.target_x, True
            for k in range(1, frames + 1):
                if shifting:
                    x += SPEED if x < target else -SPEED if x > target else 0
                    shifting = not (target - 5 <= x <= target + 5)
                lefts[i, k] = x
    return lefts


@dataclass
class Target:
    """Something to bounce on: a tile, or a power-up sitting on one (offsets follow the tile)"""
    tile: int  # index in the frame's tile list
    kind: str
    dx: int  # offset from the tile's rect.x
    dy: int  # offset from the tile's rect.y
    width: int
    height: int
    jump: int
    needs_falling: bool


class Targets:
    """Targets as parallel arrays, for vectorised contact tests"""

    def __init__(self, items):
        self.items = items
        cols = list(zip(*[(t.tile, t.dx, t.dy, t.width, t.height, t.jump, t.needs_falling) for t in items])) \
            if items else [()] * 7
        self.tile, self.dx, self.dy, self.width, self.height, self.jump, self.needs_falling = map(np.array, cols)

    def without(self, keep):
        return Targets([t for t, k in zip(self.items, keep) if k])


def targets_of(tiles):
    out = []
    for i, tile in enumerate(tiles):
        name = type(tile).__name__
        if name != "BrokenTile" and not (name == "ExplodingTile" and tile.collision):
            out.append(Target(i, name, 0, 0, TILE_W, TILE_H, JUMP, True))
        pu = live_power_up(tile)
        if pu is not None and type(pu).__name__ in BOOST:
            pname = type(pu).__name__
            out.append(Target(i, pname, pu.rect.x - tile.rect.x, pu.rect.y - tile.rect.y, pu.rect.width,
                              pu.rect.height, BOOST[pname], pname in NEEDS_FALLING))
    return Targets(out)


@dataclass
class Contacts:
    """Every (target, check frame) at which the player's height lets it touch a target, with the x positions that
    do: one entry per pair, in order of target then frame"""
    target: np.ndarray
    k: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def contacts_of(traj, targets, tops, lefts, start=0):
    """`tops[tile]` is a tile's rect.y at frame `start` of the trajectory, `lefts[tile, start + k]` its rect.x at
    check frame k."""
    if not targets.items:
        return Contacts(*[np.zeros(0, dtype=np.int64)] * 4)
    y = (tops[targets.tile] + targets.dy)[:, None] - traj.scroll[None, :-1]
    ok = (y > PLAYER_TOP - targets.height[:, None]) & (y < PLAYER_TOP + PLAYER_H)
    ok &= ~targets.needs_falling[:, None] | traj.falling[None, :-1]
    ti, kj = np.nonzero(ok)
    k = kj + 1
    left = lefts[targets.tile[ti], np.minimum(start + k, lefts.shape[1] - 1)] + targets.dx[ti]
    lo = np.maximum(-((HALF_W - 1 - left) // SPEED), 0)
    hi = np.minimum((left + targets.width[ti] + HALF_W - 1) // SPEED, N_POS - 1)
    keep = lo <= hi
    return Contacts(ti[keep], k[keep], lo[keep], hi[keep])


@dataclass
class Landing:
    target: Target
    frame: int  # collision check frame (k - 1 moves are available before it)
    lo: int  # x positions that make contact at that frame
    hi: int
    height: float  # world height at contact, higher is better


def _landing(i, contacts, targets, traj):
    k = int(contacts.k[i])
    return Landing(targets.items[contacts.target[i]], k, int(contacts.lo[i]), int(contacts.hi[i]),
                   float(-traj.y[k - 1]))


def find_landings(traj, x_node, contacts, targets):
    """The earliest contact with each target that the player can reach moving 5 px per frame"""
    reach = np.flatnonzero(circular_distance(x_node, contacts.lo, contacts.hi) <= contacts.k - 1)
    _, first = np.unique(contacts.target[reach], return_index=True)
    return [_landing(i, contacts, targets, traj) for i in reach[first]]


def node_path(a, goal, frames):
    """x positions after frames 0..frames when walking the short way from a to goal, then standing"""
    offset = (goal - a + N_POS // 2) % N_POS - N_POS // 2
    steps = np.minimum(np.arange(frames + 1), abs(offset))
    return (a + np.sign(offset) * steps) % N_POS


def contacts_on_path(xs, contacts, until):
    """Indices of the contacts that the path xs makes, up to check frame `until`"""
    at = xs[np.minimum(contacts.k - 1, len(xs) - 1)]
    return np.flatnonzero((contacts.k <= until) & (contacts.lo <= at) & (at <= contacts.hi))


def first_contact(xs, contacts, targets, traj, until):
    """What the player touches first along the path xs (the strongest jump wins a tie)"""
    hits = contacts_on_path(xs, contacts, until)
    if not hits.size:
        return None
    jumps = targets.jump[contacts.target[hits]]
    return _landing(hits[np.lexsort((jumps, contacts.k[hits]))[0]], contacts, targets, traj)


class Planner:
    def __init__(self, horizon=160, dead_end_penalty=1000.0, exploding_penalty=400.0, boost_weight=1.0,
                 candidates=8, enemy_margin=4.0, post_bounce_frames=40, min_progress=30.0):
        self.horizon = horizon
        self.dead_end_penalty = dead_end_penalty
        self.exploding_penalty = exploding_penalty
        self.boost_weight = boost_weight
        self.candidates = candidates
        self.enemy_margin = enemy_margin
        self.post_bounce_frames = post_bounce_frames
        self.min_progress = min_progress
        self._jumps = {}
        self.reset()

    def reset(self):
        self.target = None
        self._immune_until = 0

    def _jump_shape(self, jump):
        """Trajectory of a fresh jump, relative to the contact frame (the int() truncation can differ by 1 px)"""
        if jump not in self._jumps:
            self._jumps[jump] = predict(0.0, float(jump), -450, True, self.horizon)
        return self._jumps[jump]

    def score(self, landing):
        return landing.height + self.boost_weight * (landing.target.jump ** 2 - JUMP ** 2) / (2 * GRAVITY)

    def has_continuation(self, landing, x, traj, targets, tops, lefts):
        """After this bounce, can the player land somewhere higher? Bouncing in place does not count: with the
        tiles the game loses over time, that is how dead ends start. Only checked for short jumps, the tiles a
        big boost leads to do not exist yet."""
        t = landing.target
        if t.jump < BOOST["Spring"]:
            return True
        shape = self._jump_shape(t.jump)
        rest = targets.without([not (o.tile == t.tile and t.kind in ONE_USE) for o in targets.items])
        nxt = contacts_of(shape, rest, tops - traj.scroll[landing.frame - 1], lefts, start=landing.frame - 1)
        return any(o.height > self.min_progress for o in find_landings(shape, x, nxt, rest))

    def act(self, game):
        player = game.player
        x_node = round(player.x / SPEED) % N_POS
        tiles = list(game.platforms)
        traj = predict(player.y, player.velocity_y, player.previous_y_difference, player.falling, self.horizon)
        targets = targets_of(tiles)
        tops = np.array([t.rect.y for t in tiles])
        lefts = tile_lefts(tiles, 2 * self.horizon + 1)
        contacts = contacts_of(traj, targets, tops, lefts)
        # Touching an exploding tile from any side sets it off, and the game never replaces it
        fuse_targets = Targets([Target(i, "ExplodingTile", 0, 0, TILE_W, TILE_H, JUMP, False)
                                for i, t in enumerate(tiles) if type(t).__name__ == "ExplodingTile" and not t.collision])
        fuses = contacts_of(traj, fuse_targets, tops, lefts)
        # A shield only absorbs one hit (often the first of two), so enemies are avoided all the same
        tracks = self._enemy_tracks(game, traj, self.horizon + self.post_bounce_frames)
        self._immune_until = int(np.argmax(traj.falling)) if player.is_flying() else 0

        # Follow each promising landing's path and see what it really hits first
        options = sorted(find_landings(traj, x_node, contacts, targets), key=self.score, reverse=True)
        goals = []
        for o in options[:self.candidates]:
            centre = (o.lo + o.hi) // 2
            reachable = circular_distance(x_node, centre, centre) <= o.frame - 1
            goals.append(centre if reachable else min((o.lo, o.hi), key=lambda b: circular_distance(x_node, b, b)))
        goals += [x_node, (x_node + 64) % N_POS, (x_node - 64) % N_POS]  # escapes when nothing is safe
        outcomes = []
        for goal in dict.fromkeys(goals):
            xs = node_path(x_node, goal, self.horizon)
            hit = first_contact(xs, contacts, targets, traj, self.horizon)
            death, escapes = self.path_death(xs, hit, traj, tracks)
            # landing safely > nothing to land on yet > dying (later is better)
            value = (self.score(hit) if hit else -1e7) if death is None else -1e9 + death
            lit = contacts_on_path(xs, fuses, hit.frame if hit else self.horizon)
            value -= self.exploding_penalty * np.unique(fuses.target[lit]).size
            outcomes.append((value, hit, xs, escapes))
        outcomes.sort(key=lambda o: o[0], reverse=True)

        chosen = outcomes[0]
        for outcome in outcomes:  # the best one that leaves somewhere to go next
            value, hit, xs, _ = outcome
            if hit is not None and self.has_continuation(hit, int(xs[hit.frame - 1]), traj, targets, tops, lefts):
                if value >= chosen[0] - self.dead_end_penalty:
                    chosen = outcome
                break
        _, self.target, xs, escapes = chosen
        step = (int(xs[1]) - x_node + N_POS // 2) % N_POS - N_POS // 2
        step = (step > 0) - (step < 0)
        if self.target is not None and self.target.frame == 1 and escapes and step not in escapes:
            step = escapes[0]  # bouncing this frame: this move is already part of the jump
        return ACTIONS.index((step < 0, step > 0, self._should_shoot(game)))

    @staticmethod
    def _enemy_tracks(game, traj, frames):
        """Each enemy's own motion (net of the camera) at its collision check of frames 0..frames, replaying its
        movement code: monsters bounce off the screen edges, UFOs fly a figure of eight, black holes stay put"""
        tracks = []
        scroll = np.concatenate([traj.scroll, np.full(max(frames + 1 - len(traj.scroll), 0), traj.scroll[-1])])
        for e in game.all_enemies:
            name = type(e).__name__
            x0, y0 = e.rect.x, e.rect.y
            ox, oy = np.zeros(frames + 1), np.zeros(frames + 1)
            if name != "Blackhole":
                x, y, w, h = x0, y0, e.rect.width, e.rect.height
                if name == "Monster":
                    move = e.movement_function.__name__
                    sx, sy, dx, dy, speed, d = e.speed_x, e.speed_y, e.direction_x, e.direction_y, e.speed, e.direction
                else:
                    angle = e.angle
                for k in range(1, frames + 1):
                    y -= scroll[k] - scroll[k - 1]  # the player scrolls the screen before enemies move
                    if name == "UFO":
                        if not e.paused:
                            angle += 0.09
                            x += int(5 * math.sin(angle))
                            y += int(2.5 * math.sin(2 * angle))
                    elif move == "ping_pong_ball":
                        x += sx * dx
                        y += sy * dy
                        if x + w > SCREEN_WIDTH or x < 0:
                            dx = -dx
                        if y + h > SCREEN_HEIGHT or y < 0:
                            dy = -dy
                    elif move == "up_down_movement":
                        y += speed * d
                        if y + h > SCREEN_HEIGHT or y < 0:
                            d = -d
                    else:
                        x += speed * d
                        if x + w > SCREEN_WIDTH or x < 0:
                            d = -d
                    ox[k], oy[k] = x - x0, y - (y0 - scroll[k])
            tracks.append((e, x0, y0, ox, oy))
        return tracks

    def death_frame(self, xs, scroll, falling, frames, tracks):
        """First of `frames` (absolute frame numbers) at which the player, at x positions `xs`, with the camera
        at `scroll` and the falling flags given, touches a deadly enemy. Enemies check after the player moves."""
        first = None
        if not frames.size:
            return None
        px = xs * SPEED - HALF_W
        m = self.enemy_margin
        for e, x0, y0, ox, oy in tracks:
            i = np.minimum(frames, len(ox) - 1)
            ex, ey = x0 + ox[i], y0 + oy[i] - scroll
            touch = ((px < ex + e.rect.width + m) & (px + PLAYER_W > ex - m)
                     & (PLAYER_TOP < ey + e.rect.height + m) & (PLAYER_TOP + PLAYER_H > ey - m))
            if type(e).__name__ != "Blackhole":
                touch &= ~falling  # landing on monsters and UFOs kills them
            touch &= frames >= self._immune_until  # a jetpack or propeller protects until the way down
            if touch.any():
                k = int(frames[np.argmax(touch)])
                first = k if first is None else min(first, k)
        return first

    def path_death(self, xs, hit, traj, tracks):
        """Does following xs kill the player, before the contact or during the jump it gives? Returns the frame
        of death (None if safe) and the moves that survive the jump, from the contact frame on."""
        if not tracks:
            return None, [-1, 0, 1]
        until = hit.frame if hit else self.horizon
        j = np.arange(1, until)
        death = self.death_frame(xs[1:until], traj.scroll[1:until], traj.falling[1:until], j, tracks)
        if death is not None or hit is None:
            return death, [-1, 0, 1]
        # After the bounce the player can still steer: deadly only if staying, left and right all die
        shape = self._jump_shape(hit.target.jump)
        post = np.arange(1, min(self.post_bounce_frames, len(shape.scroll) - 1) + 1)
        frames = hit.frame - 1 + post
        x = int(xs[hit.frame - 1])
        deaths = {move: self.death_frame((x + move * post) % N_POS, traj.scroll[hit.frame - 1] + shape.scroll[post],
                                         shape.falling[post], frames, tracks) for move in (0, -1, 1)}
        safe = [move for move, d in deaths.items() if d is None]
        return (None if safe else max(deaths.values())), safe

    @staticmethod
    def _should_shoot(game):
        player = game.player
        if player.is_flying() or game.frame % 4:
            return False
        cx = player.rect.centerx
        return any(type(e).__name__ != "Blackhole" and e.rect.left - 6 <= cx <= e.rect.right + 6
                   and e.rect.bottom < player.rect.top for e in game.all_enemies)
