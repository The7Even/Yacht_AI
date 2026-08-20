from copy import deepcopy
import random

from app.ai.action_generator import Action, ActionType
from app.ai.monte_carlo_evaluator import MonteCarloWinProbabilityEvaluator
from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId


def near_final_state() -> GameState:
    state = GameState(game_started=True, turn=23, current_dice=(5, 5, 5, 2, 6), roll_count=2)
    state.players[PlayerId.PLAYER].category_scores = {
        category: 0 for category in ALL_CATEGORIES if category is not Category.CHOICE
    }
    state.players[PlayerId.PLAYER].category_scores[Category.FOUR_OF_A_KIND] = 100
    state.players[PlayerId.AI].category_scores = {category: 0 for category in ALL_CATEGORIES}
    state.players[PlayerId.AI].category_scores[Category.CHOICE] = 123
    return state


def test_probability_is_bounded_and_runs_a_complete_game() -> None:
    engine = GameEngine(DiceRoller(random.Random(3)))
    engine.start_game()
    engine.roll_dice()
    evaluator = MonteCarloWinProbabilityEvaluator(seed=11)

    probability = evaluator.estimate_win_probability(
        engine.state, Action(ActionType.SCORE, selected_category=Category.CHOICE), simulation_count=2
    )

    assert 0.0 <= probability <= 1.0


def test_seed_makes_results_reproducible() -> None:
    state = near_final_state()
    action = Action(ActionType.REROLL, held_indices=(0, 1, 2))

    first = MonteCarloWinProbabilityEvaluator(seed=42).estimate_win_probability(state, action, 60)
    second = MonteCarloWinProbabilityEvaluator(seed=42).estimate_win_probability(state, action, 60)

    assert first == second


def test_different_candidate_actions_can_have_different_win_probabilities() -> None:
    state = near_final_state()
    evaluator = MonteCarloWinProbabilityEvaluator(seed=7)
    probabilities = evaluator.estimate_actions_win_probability(
        state,
        (
            Action(ActionType.SCORE, selected_category=Category.CHOICE),
            Action(ActionType.REROLL, held_indices=(0, 1, 2)),
        ),
        simulation_count=80,
    )

    assert len(probabilities) == 2
    assert len(set(probabilities.values())) > 1


def test_simulation_does_not_mutate_source_game_state() -> None:
    state = near_final_state()
    original = deepcopy(state)

    MonteCarloWinProbabilityEvaluator(seed=4).estimate_win_probability(
        state, Action(ActionType.REROLL, held_indices=(0, 1, 2)), simulation_count=10
    )

    assert state == original


def test_multiple_simulation_counts_are_supported() -> None:
    state = near_final_state()
    action = Action(ActionType.SCORE, selected_category=Category.CHOICE)
    evaluator = MonteCarloWinProbabilityEvaluator(seed=1)

    assert evaluator.estimate_win_probability(state, action, simulation_count=1) == 0.5
    assert evaluator.estimate_win_probability(state, action, simulation_count=25) == 0.5


def test_finished_game_is_evaluated_without_running_simulations() -> None:
    state = near_final_state()
    state.game_over = True
    state.players[PlayerId.PLAYER].category_scores[Category.CHOICE] = 30

    probability = MonteCarloWinProbabilityEvaluator(seed=99).estimate_win_probability(
        state, Action(ActionType.SCORE, selected_category=Category.CHOICE), simulation_count=1
    )

    assert probability == 1.0
