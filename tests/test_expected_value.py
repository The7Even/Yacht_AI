import math

from app.ai.action_generator import ActionType
from app.ai.ai_player import AIPlayer
from app.ai.expected_value_strategy import ExpectedValueStrategy
from app.ai.strategy import RuleBasedStrategy
from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId


def state_with_hand(
    dice: tuple[int, ...], *, roll_count: int, available: set[Category]
) -> GameState:
    state = GameState(game_started=True, turn=1, current_dice=dice, roll_count=roll_count)
    used = set(ALL_CATEGORIES) - available
    state.players[PlayerId.PLAYER].category_scores.update({category: 0 for category in used})
    return state


def test_outcome_enumeration_counts_one_and_two_rerolled_dice() -> None:
    strategy = ExpectedValueStrategy()
    dice = (1, 2, 3, 4, 5)

    assert len(strategy.enumerate_reroll_outcomes(dice, (0, 1, 2, 3))) == 6
    assert len(strategy.enumerate_reroll_outcomes(dice, (0, 1, 2))) == 36


def test_rerolling_all_dice_enumerates_7776_outcomes_with_total_probability_one() -> None:
    strategy = ExpectedValueStrategy()
    outcomes = strategy.outcome_probabilities((1, 2, 3, 4, 5), ())

    assert len(outcomes) == 7776
    assert math.isclose(sum(probability for _, probability in outcomes), 1.0)


def test_single_reroll_ev_matches_manual_choice_calculation() -> None:
    strategy = ExpectedValueStrategy()
    ev = strategy.expected_value_for_hold(
        (5, 5, 5, 2, 6), (0, 1, 2), remaining_rolls=1, available_categories=(Category.CHOICE,)
    )

    assert math.isclose(ev, 22.0)


def test_terminal_evaluation_recognizes_available_yacht() -> None:
    assert ExpectedValueStrategy().evaluate_terminal_state(
        (5, 5, 5, 5, 5), (Category.YACHT, Category.CHOICE)
    ) == 50.0


def test_terminal_evaluation_excludes_used_categories() -> None:
    assert ExpectedValueStrategy().evaluate_terminal_state(
        (5, 5, 5, 5, 5), (Category.CHOICE,)
    ) == 25.0


def test_ev_selects_three_fives_when_only_yacht_remains() -> None:
    state = state_with_hand((5, 5, 5, 2, 6), roll_count=2, available={Category.YACHT})
    decision = AIPlayer(ExpectedValueStrategy()).decide(state)

    assert decision.action.type is ActionType.REROLL
    assert decision.held_indices == (0, 1, 2)
    assert decision.expected_value is not None
    assert decision.alternatives


def test_completed_four_of_a_kind_and_full_house_are_compared_with_rerolls() -> None:
    strategy = ExpectedValueStrategy()
    four_kind = strategy.decide(
        state_with_hand((4, 4, 4, 4, 6), roll_count=2, available={Category.FOUR_OF_A_KIND})
    )
    full_house = strategy.decide(
        state_with_hand((4, 4, 4, 5, 5), roll_count=2, available={Category.FULL_HOUSE})
    )

    assert four_kind.selected_category is Category.FOUR_OF_A_KIND
    assert full_house.selected_category is Category.FULL_HOUSE


def test_ev_pursues_large_straight_when_it_is_the_only_available_category() -> None:
    state = state_with_hand(
        (1, 2, 3, 4, 6), roll_count=2, available={Category.LARGE_STRAIGHT}
    )
    decision = ExpectedValueStrategy().decide(state)

    assert decision.action.type is ActionType.REROLL
    assert decision.held_indices == (0, 1, 2, 3)


def test_no_reroll_is_selected_when_no_rolls_remain() -> None:
    state = state_with_hand((5, 5, 5, 2, 6), roll_count=3, available={Category.CHOICE})
    decision = ExpectedValueStrategy().decide(state)

    assert decision.action.type is ActionType.SCORE
    assert decision.selected_category is Category.CHOICE


def test_rule_based_and_expected_value_strategies_work_with_the_same_engine_state() -> None:
    engine = GameEngine()
    engine.start_game()
    engine.roll_dice()

    rule_decision = AIPlayer(RuleBasedStrategy()).decide(engine.state)
    ev_decision = AIPlayer(ExpectedValueStrategy()).decide(engine.state)

    assert rule_decision.action.type in ActionType
    assert ev_decision.action.type in ActionType
    assert all(index in range(5) for index in ev_decision.held_indices)
