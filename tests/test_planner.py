"""The planner's model must predict the game exactly."""
import pytest

from doodle_agent.env import DoodleJumpEnv
from doodle_agent.planner import N_POS, Planner, find_landings, predict, targets_of, tile_lefts


@pytest.fixture(scope="module")
def env():
    return DoodleJumpEnv(frame_skip=1, max_frames=100_000)


def static_tile(game):
    """A plain tile near the top of the screen, so it stays on screen while the camera moves"""
    return min((t for t in game.platforms if type(t).__name__ == "Tile"), key=lambda t: abs(t.rect.y - 150))


@pytest.mark.parametrize("seed", range(5))
def test_predicted_scroll_and_height_match_the_game(env, seed):
    env.reset(seed=seed)
    game, player = env.game, env.game.player
    for _ in range(200):  # get into the climbing regime
        env.step(4)
    planner = Planner()
    for _ in range(30):
        env.step(planner.act(game))
    traj = predict(player.y, player.velocity_y, player.previous_y_difference, player.falling, 25)
    tile = static_tile(game)
    y0 = tile.rect.y
    v_before = player.velocity_y
    for k in range(1, 26):
        env.step(1)
        if player.velocity_y < v_before:  # bounced: the no-bounce prediction stops being valid
            break
        v_before = player.velocity_y
        assert player.y == traj.y[k]
        assert player.falling == traj.falling[k]
        assert tile.rect.y == y0 - traj.scroll[k]


@pytest.mark.parametrize("seed", range(8))
def test_predicted_first_contact_matches_the_game(env, seed):
    """Standing still from the start, the first bounce happens exactly on the predicted frame"""
    env.reset(seed=seed)
    game, player = env.game, env.game.player
    x_node = round(player.x / 5) % N_POS
    traj = predict(player.y, player.velocity_y, player.previous_y_difference, player.falling, 200)
    tiles = list(game.platforms)
    landings = find_landings(traj, x_node, targets_of(tiles), {id(t): t.rect.y for t in tiles},
                             {id(t): tile_lefts(t, 401) for t in tiles})
    still = [ld.frame for ld in landings if ld.lo <= x_node <= ld.hi]
    expected = min(still) if still else None
    bounced = None
    for frame in range(1, 201):
        v = player.velocity_y
        env.step(1)
        if player.velocity_y < v:
            bounced = frame
            break
        if player.dead:
            break
    assert bounced == expected
