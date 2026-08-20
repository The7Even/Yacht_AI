import math

from app.ai.game_aware_strategy import GameAwareStrategicExpectedValueStrategy
from app.ai.game_state_evaluator import GameStateEvaluator
from app.ai.strategic_expected_value_strategy import StrategicExpectedValueStrategy
from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_state import GameState, PlayerId


def make_state(
    *,
    player_score: int = 0,
    ai_score: int = 0,
    player_used_count: int = 1,
    ai_used_count: int = 1,
) -> GameState:
    state = GameState(game_started=True, turn=1, current_dice=(5, 5, 5, 2, 6), roll_count=3)
    for player_id, score, used_count in (
        (PlayerId.PLAYER, player_score, player_used_count),
        (PlayerId.AI, ai_score, ai_used_count),
    ):
        used = ALL_CATEGORIES[:used_count]
        scores = state.players[player_id].category_scores
        for category in used:
            scores[category] = 0
        if used:
            scores[used[0]] = score
    return state


def test_leading_and_trailing_positions_are_evaluated_differently() -> None:
    evaluator = GameStateEvaluator()
    leading = evaluator.evaluate(make_state(player_score=150, ai_score=100))
    trailing = evaluator.evaluate(make_state(player_score=100, ai_score=150))

    assert leading.position_value > trailing.position_value
    assert leading.score_difference == -trailing.score_difference


def test_same_score_difference_has_more_weight_near_end_of_game() -> None:
    evaluator = GameStateEvaluator()
    early = evaluator.evaluate(
        make_state(player_score=110, ai_score=100, player_used_count=1, ai_used_count=1)
    )
    late = evaluator.evaluate(
        make_state(player_score=110, ai_score=100, player_used_count=11, ai_used_count=11)
    )

    assert late.remaining_turns < early.remaining_turns
    assert abs(late.position_value) > abs(early.position_value)


def test_remaining_category_counts_change_future_score_value() -> None:
    evaluator = GameStateEvaluator()
    many_remaining = evaluator.evaluate(make_state(player_used_count=1, ai_used_count=1))
    few_remaining = evaluator.evaluate(make_state(player_used_count=11, ai_used_count=11))

    assert many_remaining.my_future_score > few_remaining.my_future_score


def test_bonus_state_is_distinguished_for_each_participant() -> None:
    evaluator = GameStateEvaluator()
    mine = make_state(player_used_count=3, ai_used_count=3)
    mine.players[PlayerId.PLAYER].category_scores.update(
        {Category.SIXES: 30, Category.FIVES: 25, Category.FOURS: 10}
    )
    theirs = make_state(player_used_count=3, ai_used_count=3)
    theirs.players[PlayerId.AI].category_scores.update(
        {Category.SIXES: 30, Category.FIVES: 25, Category.FOURS: 10}
    )

    assert mine.players[PlayerId.PLAYER].has_upper_bonus is True
    assert theirs.players[PlayerId.AI].has_upper_bonus is True
    assert evaluator.evaluate(mine).position_value > evaluator.evaluate(theirs).position_value


def test_opponent_high_and_low_remaining_upper_categories_have_different_value() -> None:
    evaluator = GameStateEvaluator()
    high = make_state(player_used_count=12, ai_used_count=10)
    low = make_state(player_used_count=12, ai_used_count=10)
    high.players[PlayerId.AI].category_scores = {
        category: 0 for category in ALL_CATEGORIES if category not in {Category.FIVES, Category.SIXES}
    }
    low.players[PlayerId.AI].category_scores = {
        category: 0 for category in ALL_CATEGORIES if category not in {Category.ONES, Category.TWOS}
    }

    assert evaluator.evaluate(high).opponent_future_score > evaluator.evaluate(low).opponent_future_score


def test_game_context_does_not_readd_the_selected_category_score() -> None:
    state = make_state(player_score=100, ai_score=100, player_used_count=10, ai_used_count=10)
    state.players[PlayerId.PLAYER].category_scores = {
        category: 0 for category in ALL_CATEGORIES if category not in {Category.FIVES, Category.CHOICE}
    }
    baseline = StrategicExpectedValueStrategy().decide(state)
    aware = GameAwareStrategicExpectedValueStrategy().decide(state)
    context = GameStateEvaluator().evaluate(state).position_value

    assert aware.action == baseline.action
    assert math.isclose((aware.expected_value or 0.0) - (baseline.expected_value or 0.0), context)
    assert all(
        math.isclose(aware_alt.expected_value - base_alt.expected_value, context)
        for aware_alt, base_alt in zip(aware.alternatives, baseline.alternatives, strict=True)
    )
