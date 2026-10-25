"""Play one seeded game and dump a per-frame state trace, to check that the patched game behaves like the original.

    python tests/game_trace.py patched  GAME_DIR SEED FRAMES OUT.json            # closed-loop policy, records inputs
    python tests/game_trace.py original GAME_DIR SEED FRAMES OUT.json INPUTS.json  # replays them through the real loop

`original` drives the unmodified main loop (menu click, handle_events, update, draw) with scripted keyboard and
mouse; `original-nodraw` does the same but skips draw(). Standalone on purpose: it must run against the original
game, which knows nothing of this project.
"""
import hashlib
import json
import os
import random
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

mode, game_dir, seed, n_frames, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.chdir(game_dir)
sys.path.insert(0, game_dir)

import numpy as np  # noqa: E402
import pygame  # noqa: E402

held = {pygame.K_LEFT: False, pygame.K_RIGHT: False}
mouse_pos = (0, 0)
if mode != "patched":
    pygame.key.get_pressed = lambda: {**held, pygame.K_SPACE: False, pygame.K_UP: False}
    pygame.mouse.get_pos = lambda: mouse_pos

from game import Game  # noqa: E402


def snapshot(g):
    p = g.player
    tiles = []
    for t in g.platforms:
        pu = t.power_up
        tiles.append((type(t).__name__, tuple(t.rect),
                      type(pu).__name__ if pu else None, tuple(pu.rect) if pu else None,
                      getattr(pu, "expanded", None)))
    enemies = [(type(e).__name__, tuple(e.rect)) for e in g.all_enemies]
    state = (p.x, p.y, p.velocity_y, p.score, p.falling, p.dead, p.using_shield, p.using_spring_shoes,
             p.is_flying(), tuple(p.rect), tiles, enemies, [tuple(b.rect) for b in g.bullets])
    return hashlib.sha1(repr(state).encode()).hexdigest()


def greedy_inputs(g, rng):
    """Steer under the closest tile below, shoot now and then: enough to reach special tiles and enemies"""
    p = g.player
    below = [t for t in g.platforms if t.rect.top > p.rect.bottom - 10]
    left = right = False
    if below:
        target = min(below, key=lambda t: t.rect.top - p.rect.bottom).rect.centerx
        left, right = target < p.x - 5, target > p.x + 5
    return [bool(left), bool(right), bool(rng.random() < 0.05)]


trace = []
g = Game()
if mode == "patched":
    rng = np.random.default_rng(seed)
    inputs = []
    g.reset_game(seed)
    while len(trace) < n_frames and not g.player.dead:
        inputs.append(greedy_inputs(g, rng))
        g.step(*inputs[-1])
        trace.append(snapshot(g))
    json.dump({"inputs": inputs, "trace": trace}, open(out, "w"))
    sys.exit()

inputs = json.load(open(sys.argv[6]))["inputs"]
random.seed(seed)  # menu frames draw no random numbers, so seeding before the click is equivalent
play = g.play_button.rect.center
menu_events = [[(pygame.MOUSEBUTTONDOWN, play)], [(pygame.MOUSEBUTTONUP, play)]]
frame = 0
while len(trace) < len(inputs):
    # The game starts within the frame that handles the button release, so that frame gets the first inputs
    left, right, shoot = inputs[len(trace)]
    held[pygame.K_LEFT], held[pygame.K_RIGHT] = left, right
    if menu_events:
        for kind, pos in menu_events.pop(0):
            mouse_pos = pos
            pygame.event.post(pygame.event.Event(kind, button=1, pos=pos))
    if shoot and not menu_events:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    g.handle_events()
    g.update()
    if mode != "original-nodraw":
        g.draw()
    g.frame += 1
    if g.play_game:
        trace.append(snapshot(g))
    frame += 1
    assert frame < len(inputs) + 100, "game never started"
json.dump({"trace": trace}, open(out, "w"))
