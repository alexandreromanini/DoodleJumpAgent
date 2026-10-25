"""Model-based planner: predicts the game with its own physics and steers to the best reachable landing.

It re-plans every frame (model predictive control). The model replays the game's arithmetic exactly:
semi-implicit Euler with gravity 0.4; a camera that pins the player's rect at y=390 and scrolls everything else
by int()-truncated steps; tile collisions checked before the player moves in a frame, enemy collisions after;
x on a circle of 129 positions 5 px apart (moving right from 640 wraps to 0).
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


@dataclass
class Contacts:
    """The frames at which the player's height lets it touch a target, and the x positions that do at each"""
    target: Target
    ks: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


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


def contacts_of(traj, targets, tops, lefts, start=0):
    """`tops[id(tile)]` is a tile's rect.y at frame `start` of the trajectory, `lefts[id(tile)][start + k]` its
    rect.x at check frame k."""
    out = []
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
        keep = lo <= hi
        if keep.any():
            out.append(Contacts(t, ks[keep], lo[keep], hi[keep]))
    return out


def find_landings(traj, x_node, contacts):
    """The earliest contact with each target that the player can reach moving 5 px per frame"""
    found = []
    for c in contacts:
        reach = circular_distance(x_node, c.lo, c.hi) <= c.ks - 1
        if reach.any():
            i = int(np.argmax(reach))
            found.append(Landing(c.target, int(c.ks[i]), int(c.lo[i]), int(c.hi[i]), float(-traj.y[c.ks[i] - 1])))
    return found


def node_path(a, goal, frames):
    """x positions after frames 0..frames when walking the short way from a to goal, then standing"""
    offset = (goal - a + N_POS // 2) % N_POS - N_POS // 2
    steps = np.minimum(np.arange(frames + 1), abs(offset))
    return (a + np.sign(offset) * steps) % N_POS


def first_contact(xs, contacts, traj, until):
    """What the player touches first along the path xs, up to check frame `until` (the strongest jump wins a tie)"""
    best = None
    for c in contacts:
        m = c.ks <= until
        if not m.any():
            continue
        ks = c.ks[m]
        at = xs[ks - 1]
        hit = (c.lo[m] <= at) & (at <= c.hi[m])
        if hit.any():
            i = int(np.argmax(hit))
            k = int(ks[i])
            if best is None or (k, c.target.jump) < (best.frame, best.target.jump):
                best = Landing(c.target, k, int(c.lo[m][i]), int(c.hi[m][i]), float(-traj.y[k - 1]))
    return best


class Planner:
    def __init__(self, horizon=160, dead_end_penalty=1000.0, exploding_penalty=50.0, boost_weight=1.0,
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

    def _jump_shape(self, jump):
        """Trajectory of a fresh jump, relative to the contact frame (the int() truncation can differ by 1 px)"""
        if jump not in self._jumps:
            self._jumps[jump] = predict(0.0, float(jump), -450, True, self.horizon)
        return self._jumps[jump]

    def score(self, landing):
        t = landing.target
        return landing.height + self.boost_weight * (t.jump ** 2 - JUMP ** 2) / (2 * GRAVITY)

    def has_continuation(self, landing, x, traj, targets, tops, lefts):
        """After this bounce, can the player land somewhere higher? Bouncing in place does not count: with the
        tiles the game loses over time, that is how dead ends start. Only checked for short jumps, the tiles a
        big boost leads to do not exist yet."""
        t = landing.target
        if t.jump < BOOST["Spring"]:
            return True
        k = landing.frame
        shift = traj.scroll[k - 1]
        nxt_tops = {key: top - shift for key, top in tops.items()}
        rest = [o for o in targets if not (o.tile is t.tile and t.kind in ONE_USE)]
        nxt = contacts_of(self._jump_shape(t.jump), rest, nxt_tops, lefts, start=k - 1)
        return any(o.height > self.min_progress for o in find_landings(self._jump_shape(t.jump), x, nxt))

    def act(self, game):
        player = game.player
        x_node = round(player.x / SPEED) % N_POS
        tiles = list(game.platforms)
        traj = predict(player.y, player.velocity_y, player.previous_y_difference, player.falling, self.horizon)
        targets = targets_of(tiles)
        tops = {id(t): t.rect.y for t in tiles}
        lefts = {id(t): tile_lefts(t, 2 * self.horizon + 1) for t in tiles}
        contacts = contacts_of(traj, targets, tops, lefts)
        # Touching an exploding tile from any side sets it off, and the game never replaces it
        fuses = contacts_of(traj, [Target(t, "ExplodingTile", 0, 0, TILE_W, TILE_H, JUMP, False) for t in tiles
                                   if type(t).__name__ == "ExplodingTile" and not t.collision], tops, lefts)
        tracks = [] if player.using_shield or player.is_flying() else \
            self._enemy_tracks(game, traj, self.horizon + self.post_bounce_frames)

        # Follow each promising landing's path and see what it really hits first
        options = sorted(find_landings(traj, x_node, contacts), key=self.score, reverse=True)[:self.candidates]
        goals = []
        for o in options:
            centre = (o.lo + o.hi) // 2
            reachable = circular_distance(x_node, centre, centre) <= o.frame - 1
            goals.append(centre if reachable else min((o.lo, o.hi), key=lambda b: circular_distance(x_node, b, b)))
        goals += [x_node, (x_node + 64) % N_POS, (x_node - 64) % N_POS]  # escapes when nothing is safe
        outcomes = []
        for goal in dict.fromkeys(goals):
            xs = node_path(x_node, goal, self.horizon)
            hit = first_contact(xs, contacts, traj, self.horizon)
            death = self.path_death(xs, hit, traj, tracks)
            # landing safely > nothing to land on yet > dying (later is better)
            value = (self.score(hit) if hit else -1e7) if death is None else -1e9 + death
            until = hit.frame if hit else self.horizon
            value -= self.exploding_penalty * sum(first_contact(xs, [f], traj, until) is not None for f in fuses)
            outcomes.append((value, hit, xs))
        outcomes.sort(key=lambda o: o[0], reverse=True)

        chosen = outcomes[0]
        for value, hit, xs in outcomes:  # the best one that leaves somewhere to go next
            if hit is not None and self.has_continuation(hit, int(xs[hit.frame - 1]), traj, targets, tops, lefts):
                if value >= chosen[0] - self.dead_end_penalty:
                    chosen = (value, hit, xs)
                break
        _, self.target, xs = chosen
        step = (int(xs[1]) - x_node + N_POS // 2) % N_POS - N_POS // 2
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
            if touch.any():
                k = int(frames[np.argmax(touch)])
                first = k if first is None else min(first, k)
        return first

    def path_death(self, xs, hit, traj, tracks):
        """Does following xs kill the player, before the contact or during the jump it gives?"""
        if not tracks:
            return None
        until = hit.frame if hit else self.horizon
        j = np.arange(1, until)
        death = self.death_frame(xs[1:until], traj.scroll[1:until], traj.falling[1:until], j, tracks)
        if death is not None or hit is None:
            return death
        # After the bounce the player can still steer: deadly only if staying, left and right all die
        shape = self._jump_shape(hit.target.jump)
        post = np.arange(1, min(self.post_bounce_frames, len(shape.scroll) - 1) + 1)
        frames = hit.frame - 1 + post
        x = int(xs[hit.frame - 1])
        deaths = [self.death_frame((x + move * post) % N_POS, traj.scroll[hit.frame - 1] + shape.scroll[post],
                                   shape.falling[post], frames, tracks) for move in (0, -1, 1)]
        return None if None in deaths else max(deaths)

    @staticmethod
    def _should_shoot(game):
        player = game.player
        if player.is_flying() or game.frame % 4:
            return False
        cx = player.rect.centerx
        return any(type(e).__name__ != "Blackhole" and e.rect.left - 6 <= cx <= e.rect.right + 6
                   and e.rect.bottom < player.rect.top for e in game.all_enemies)
