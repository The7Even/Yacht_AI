from app.ai.action_generator import ActionType
from app.ai.fast_expected_value_strategy import FastExpectedValueStrategy
from app.core.categories import Category
from app.core.game_state import GameState, PlayerId


def state_with_roll() -> GameState:
    state = GameState()
    state.start_game()
    state.current_dice = (5, 5, 5, 2, 6)
    state.roll_count = 2
    return state


def test_fast_strategy_returns_legal_action() -> None:
    result = FastExpectedValueStrategy().decide(state_with_roll())
    assert result.action.type in {ActionType.SCORE, ActionType.REROLL}


def test_fast_strategy_can_score_completed_yacht() -> None:
    state = state_with_roll()
    state.current_dice = (6, 6, 6, 6, 6)
    result = FastExpectedValueStrategy().decide(state)
    assert result.action.type is ActionType.SCORE
    assert result.action.selected_category is Category.YACHT


def test_fast_strategy_respects_used_categories() -> None:
    state = state_with_roll()
    state.players[PlayerId.PLAYER].used_categories.add(Category.FIVES)
    result = FastExpectedValueStrategy().decide(state)
    assert result.action.selected_category is not Category.FIVES


def test_fast_strategy_does_not_reroll_after_third_roll() -> None:
    state = state_with_roll()
    state.roll_count = 3
    result = FastExpectedValueStrategy().decide(state)
    assert result.action.type is ActionType.SCORE
