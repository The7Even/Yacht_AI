import math

from app.ai.action_generator import ActionType
from app.ai.ai_player import AIPlayer
from app.ai.bonus_evaluator import BonusEvaluator
from app.ai.expected_value_strategy import ExpectedValueStrategy
from app.ai.strategic_expected_value_strategy import StrategicExpectedValueStrategy
from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_state import GameState, PlayerId


def state_with_upper_total(
    dice: tuple[int, ...], *, upper_total: int, available: set[Category], roll_count: int = 3
) -> GameState:
    state = GameState(game_started=True, turn=1, current_dice=dice, roll_count=roll_count)
    scores = state.players[PlayerId.PLAYER].category_scores
    scores.update({category: 0 for category in set(ALL_CATEGORIES) - available})
    used_upper = [category for category in ALL_CATEGORIES if category.is_upper and category not in available]
    remaining_total = upper_total
    for category in reversed(used_upper):
        value = min(category.upper_face * 5, remaining_total)
        scores[category] = value
        remaining_total -= value
    assert remaining_total == 0
    return state


def test_near_bonus_upper_score_receives_actual_bonus_value() -> None:
    strategy = StrategicExpectedValueStrategy()
    state = state_with_upper_total(
        (5, 5, 5, 2, 6), upper_total=62, available={Category.FIVES, Category.CHOICE}
    )

    value = strategy.evaluate_category_value(
        state.current_dice, Category.FIVES, (Category.FIVES, Category.CHOICE), 62, False  # type: ignore[arg-type]
    )

    assert value == 50.0  # 15 Fives + the now-earned 35-point bonus


def test_low_upper_total_has_different_bonus_value_for_same_dice_and_categories() -> None:
    strategy = StrategicExpectedValueStrategy()
    high_value = strategy.evaluate_category_value(
        (5, 5, 5, 2, 6), Category.FIVES, (Category.FIVES, Category.CHOICE), 62, False
    )
    low_value = strategy.evaluate_category_value(
        (5, 5, 5, 2, 6), Category.FIVES, (Category.FIVES, Category.CHOICE), 20, False
    )

    assert high_value == 50.0
    assert low_value == 15.0


def test_no_extra_bonus_value_exists_after_bonus_is_earned() -> None:
    evaluation = BonusEvaluator().evaluate(
        upper_total=63,
        remaining_upper_categories=(Category.FIVES,),
        selected_category=Category.FIVES,
        category_score=15,
        has_bonus=True,
    )

    assert evaluation.total_value == 0.0


def test_no_bonus_value_exists_when_no_upper_category_remains() -> None:
    evaluation = BonusEvaluator().evaluate(
        upper_total=62,
        remaining_upper_categories=(),
        selected_category=Category.CHOICE,
        category_score=23,
    )

    assert evaluation.total_value == 0.0


def test_same_dice_have_different_strategic_terminal_values_at_different_upper_totals() -> None:
    strategy = StrategicExpectedValueStrategy()
    high = strategy.evaluate_strategic_terminal_state(
        (5, 5, 5, 2, 6), (Category.FIVES, Category.CHOICE), 62, False
    )
    low = strategy.evaluate_strategic_terminal_state(
        (5, 5, 5, 2, 6), (Category.FIVES, Category.CHOICE), 20, False
    )

    assert high == 50.0
    assert low == 23.0
    assert high != low


def test_bonus_heuristic_does_not_discard_materially_higher_immediate_score() -> None:
    state = state_with_upper_total(
        (5, 5, 5, 5, 6), upper_total=20, available={Category.FIVES, Category.CHOICE}
    )
    decision = AIPlayer(StrategicExpectedValueStrategy()).decide(state)

    assert decision.action.type is ActionType.SCORE
    assert decision.selected_category is Category.CHOICE


def test_strategic_and_pure_ev_can_be_selected_interchangeably() -> None:
    state = state_with_upper_total(
        (5, 5, 5, 2, 6), upper_total=62, available={Category.FIVES, Category.CHOICE}
    )

    pure = AIPlayer(ExpectedValueStrategy()).decide(state)
    strategic = AIPlayer(StrategicExpectedValueStrategy()).decide(state)

    assert pure.selected_category is Category.CHOICE
    assert strategic.selected_category is Category.FIVES
    assert math.isclose(strategic.expected_value or 0.0, 50.0)
