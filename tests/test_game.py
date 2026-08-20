import pytest

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT, DiceRoller
from app.core.game_engine import GameEngine, MAX_ROLLS_PER_TURN
from app.core.game_state import PlayerId


class ScriptedDiceRoller(DiceRoller):
    """Deterministic die source used to keep game-engine tests reproducible."""

    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = iter(values)

    def roll(self, count: int = DICE_COUNT) -> tuple[int, ...]:
        return tuple(next(self._values) for _ in range(count))


def make_engine(
    values: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 6, 6, 1, 1, 1, 2, 2, 2, 2)
) -> GameEngine:
    return GameEngine(ScriptedDiceRoller(values))


def test_initial_state_is_not_a_started_game() -> None:
    engine = make_engine()

    assert engine.state.game_started is False
    assert engine.state.turn == 0
    assert engine.state.current_dice is None
    assert engine.state.player_score == 0
    assert engine.state.ai_score == 0


def test_start_game_initializes_first_player_turn() -> None:
    engine = make_engine()
    state = engine.start_game()

    assert state.game_started is True
    assert state.current_player is PlayerId.PLAYER
    assert state.turn == 1
    assert state.roll_count == 0
    assert engine.get_available_categories() == ALL_CATEGORIES


def test_roll_requires_a_started_game() -> None:
    with pytest.raises(RuntimeError, match="Start a game"):
        make_engine().roll_dice()


def test_first_roll_rolls_five_dice() -> None:
    engine = make_engine()
    engine.start_game()

    assert engine.roll_dice() == (1, 2, 3, 4, 5)
    assert engine.state.roll_count == 1


def test_held_dice_are_not_rerolled() -> None:
    engine = make_engine()
    engine.start_game()
    engine.roll_dice()
    engine.hold_dice(1)
    engine.hold_dice(3)

    assert engine.roll_dice() == (6, 2, 6, 4, 6)
    assert engine.state.held_dice == (2, 4)
    assert engine.state.roll_count == 2


def test_hold_and_unhold_validate_indices_and_state() -> None:
    engine = make_engine()
    engine.start_game()
    engine.roll_dice()

    with pytest.raises(ValueError):
        engine.hold_dice(5)
    engine.hold_dice(0)
    with pytest.raises(ValueError, match="already held"):
        engine.hold_dice(0)
    engine.unhold_dice(0)
    with pytest.raises(ValueError, match="not held"):
        engine.unhold_dice(0)


def test_turn_cannot_exceed_three_rolls() -> None:
    engine = make_engine((1,) * 20)
    engine.start_game()
    for _ in range(MAX_ROLLS_PER_TURN):
        engine.roll_dice()

    with pytest.raises(RuntimeError, match="more than three"):
        engine.roll_dice()


def test_current_scores_require_dice_and_only_include_unused_categories() -> None:
    engine = make_engine()
    engine.start_game()
    with pytest.raises(RuntimeError, match="Roll dice"):
        engine.get_current_scores()

    engine.roll_dice()
    assert engine.get_current_scores()[Category.CHOICE] == 15


def test_scoring_applies_score_and_prevents_duplicate_category() -> None:
    engine = make_engine()
    engine.start_game()
    engine.roll_dice()

    assert engine.score_category(Category.CHOICE) == 15
    assert engine.state.player_category_scores == {Category.CHOICE: 15}
    assert Category.CHOICE in engine.state.player_used_categories
    with pytest.raises(RuntimeError, match="End the scored turn"):
        engine.roll_dice()

    engine.end_turn()
    engine.roll_dice()
    engine.score_category(Category.ONES)
    engine.end_turn()
    engine.roll_dice()
    with pytest.raises(ValueError, match="already been used"):
        engine.score_category(Category.CHOICE)


def test_end_turn_requires_score_then_starts_next_player_turn() -> None:
    engine = make_engine()
    engine.start_game()
    with pytest.raises(RuntimeError, match="must be scored"):
        engine.end_turn()

    engine.roll_dice()
    engine.score_category(Category.CHOICE)
    next_state = engine.end_turn()

    assert next_state.current_player is PlayerId.AI
    assert next_state.turn == 2
    assert next_state.current_dice is None
    assert next_state.roll_count == 0
    assert next_state.turn_scored is False


def test_game_ends_after_both_players_use_every_category() -> None:
    engine = make_engine((1, 1, 1, 1, 1))
    engine.start_game()
    engine.state.players[PlayerId.PLAYER].category_scores.update(
        {category: 0 for category in ALL_CATEGORIES if category is not Category.ONES}
    )
    engine.state.players[PlayerId.AI].category_scores.update(
        {category: 0 for category in ALL_CATEGORIES}
    )

    engine.roll_dice()
    engine.score_category(Category.ONES)

    assert engine.is_game_over() is True
    with pytest.raises(RuntimeError, match="already over"):
        engine.roll_dice()
