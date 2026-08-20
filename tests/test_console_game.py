import random

from app.console_game import play_console_game
from app.core.dice import DiceRoller
from app.core.game_state import PlayerId


def test_console_runner_completes_a_full_game() -> None:
    lines: list[str] = []

    engine = play_console_game(dice_roller=DiceRoller(random.Random(7)), output=lines.append)

    assert engine.is_game_over() is True
    assert len(engine.state.players[PlayerId.PLAYER].used_categories) == 12
    assert len(engine.state.players[PlayerId.AI].used_categories) == 12
    assert any(line.startswith("Game over") for line in lines)
    assert any(line.startswith("Winner:") for line in lines)
