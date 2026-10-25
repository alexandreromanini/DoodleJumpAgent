"""The patched game, stepped headless through the new API, must replay the original game exactly.

The original is the upstream commit, run through its own main loop with rendering; both games get the same seed
and the same inputs, and every frame's state (player, tiles, power-ups, enemies, bullets) must match.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "third_party" / "doodle-jump"
UPSTREAM = "3740388"
TRACE = Path(__file__).with_name("game_trace.py")


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    path = tmp_path_factory.mktemp("original")
    archive = subprocess.run(["git", "-C", str(GAME), "archive", UPSTREAM], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(path)], input=archive, check=True)
    return path


def trace(mode, game_dir, seed, frames, out, inputs=None):
    cmd = [sys.executable, str(TRACE), mode, str(game_dir), str(seed), str(frames), str(out)]
    subprocess.run(cmd + ([str(inputs)] if inputs else []), check=True, capture_output=True)
    return json.loads(out.read_text())["trace"]


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


@pytest.mark.parametrize("seed", [1, 2])
def test_headless_api_replays_the_original_game(original, tmp_path, seed):
    patched = trace("patched", GAME, seed, 2500, tmp_path / "patched.json")
    replayed = trace("original", original, seed, 2500, tmp_path / "original.json", tmp_path / "patched.json")
    assert len(patched) > 500
    assert first_difference(patched, replayed) is None


def test_skipping_draw_in_the_original_changes_the_game(original, tmp_path):
    """Why the spring fix exists: the original game updates used springs in draw()"""
    trace("patched", GAME, 2, 2500, tmp_path / "patched.json")
    rendered = trace("original", original, 2, 2500, tmp_path / "a.json", tmp_path / "patched.json")
    headless = trace("original-nodraw", original, 2, 2500, tmp_path / "b.json", tmp_path / "patched.json")
    assert first_difference(rendered, headless) is not None
