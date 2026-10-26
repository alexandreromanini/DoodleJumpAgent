"""Loading the game and reading its state.

The game opens its window, loads its images and starts the mixer when `game.py` is imported, and loads assets
from paths relative to the working directory. So it is imported once per process, in the mode chosen first.
"""
import os
import sys
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[2] / "third_party" / "doodle-jump"

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 900
GRAVITY = 0.4
SPEED = 5  # horizontal px/frame
PLAYER_W, PLAYER_H = 62, 60
PLAYER_TOP = SCREEN_HEIGHT // 2 - PLAYER_H  # the camera pins the player's rect here while climbing

TILE_TYPES = ("Tile", "MovingTile", "ShiftingTile", "MoveableTile", "DisappearingTile", "BrokenTile", "ExplodingTile")
POWER_UPS = ("Spring", "Trampoline", "Jetpack", "Propeller", "Shield", "SpringShoes")
ENEMY_TYPES = ("Monster", "Blackhole", "UFO")

# (left, right, shoot) for each discrete action
ACTIONS = tuple((move == -1, move == 1, shoot) for shoot in (False, True) for move in (-1, 0, 1))
ACTION_NAMES = tuple(f"{m}{'+shoot' if s else ''}" for s in (False, True) for m in ("left", "none", "right"))

_mode = None


def load_game(render=False):
    """Return the game's `Game` class, importing it in headless (dummy SDL drivers) or windowed mode."""
    global _mode
    if _mode is not None:
        if _mode != render:
            raise RuntimeError("the game was already imported in the other render mode in this process")
        return sys.modules["game"].Game
    if not render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # The game loads ~75 assets through cwd-relative paths; chdir instead of rewriting them all.
    # Callers resolve their own paths before this runs.
    os.chdir(GAME_DIR)
    sys.path.insert(0, str(GAME_DIR))
    import game

    _mode = render
    return game.Game


def death_cause(player):
    """Why a dead player died: fall, monster, ufo or blackhole"""
    if player.suction_object_collided_with is not None:
        return type(player.suction_object_collided_with).__name__.lower()
    if player.knocked_out:
        return "monster"
    return "fall"
