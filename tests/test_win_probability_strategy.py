from copy import deepcopy

from app.ai.action_generator import Action, ActionType
from app.ai.fast_expected_value_strategy import FastExpectedValueStrategy
from app.ai.monte_carlo_rollout_strategy import MonteCarloRolloutStrategy
from app.ai.win_probability_strategy import WinProbabilityStrategy
from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_state import GameState, PlayerId


class FakeEvaluator:
    def __init__(self, probabilities: dict[Action, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[tuple[GameState, tuple[Action, ...], int]] = []

    def estimate_actions_win_probability(
        self, state: GameState, actions: tuple[Action, ...], simulation_count: int
    ) -> dict[Action, float]:
        self.calls.append((deepcopy(state), actions, simulation_count))
        return {action: self.probabilities[action] for action in actions}


def state_with_roll() -> GameState:
    state = GameState(
        game_started=True,
        current_dice=(5, 5, 5, 2, 6),
        roll_count=2,
    )
    state.players[PlayerId.PLAYER].category_scores = {
        category: 0 for category in ALL_CATEGORIES if category is not Category.CHOICE
    }
    return state


def test_selects_highest_probability_action() -> None:
    state = state_with_roll()
    reroll = Action(ActionType.REROLL, held_indices=(0, 1, 2))
    score = Action(ActionType.SCORE, selected_category=Category.CHOICE)

    probabilities = {
        action: 0.10 for action in WinProbabilityStrategy._candidate_actions(state)
    }
    probabilities[reroll] = 0.30
    probabilities[score] = 0.80
    evaluator = FakeEvaluator(probabilities)

    result = WinProbabilityStrategy(evaluator=evaluator, simulation_count=123).decide(state)

    assert result.action == score
    assert result.win_probability == 0.80
    assert result.alternatives[0].win_probability == 0.80
    assert result.alternatives[1].win_probability == 0.30
    assert evaluator.calls[0][2] == 123


def test_returns_at_most_three_ranked_alternatives() -> None:
    state = state_with_roll()
    probabilities = {}
    for action in WinProbabilityStrategy._candidate_actions(state):
        probabilities[action] = 0.5
    evaluator = FakeEvaluator(probabilities)

    result = WinProbabilityStrategy(evaluator=evaluator, simulation_count=10).decide(state)

    assert len(result.alternatives) == 3
    assert all(0.0 <= alternative.win_probability <= 1.0 for alternative in result.alternatives)


def test_ties_are_deterministic() -> None:
    state = state_with_roll()
    actions = WinProbabilityStrategy._candidate_actions(state)
    evaluator = FakeEvaluator({action: 0.5 for action in actions})
    strategy = WinProbabilityStrategy(evaluator=evaluator, simulation_count=10)

    first = strategy.decide(state)
    second = strategy.decide(state)

    assert first.action == second.action
    assert first.alternatives == second.alternatives


def test_strategy_does_not_mutate_game_state() -> None:
    state = state_with_roll()
    original = deepcopy(state)
    actions = WinProbabilityStrategy._candidate_actions(state)
    evaluator = FakeEvaluator({action: 0.4 for action in actions})

    WinProbabilityStrategy(evaluator=evaluator, simulation_count=10).decide(state)

    assert state == original


def test_default_continuation_policy_is_monte_carlo_rollout() -> None:
    strategy = WinProbabilityStrategy(simulation_count=1)

    assert isinstance(strategy._evaluator._player_strategy, MonteCarloRolloutStrategy)


def test_continuation_and_opponent_policies_are_injectable() -> None:
    continuation = FastExpectedValueStrategy()
    opponent = FastExpectedValueStrategy()
    strategy = WinProbabilityStrategy(
        simulation_count=1,
        continuation_strategy=continuation,
        opponent_strategy=opponent,
    )

    assert strategy._evaluator._player_strategy is continuation
    assert strategy._evaluator._opponent_strategy is opponent


def test_requires_positive_simulation_count() -> None:
    try:
        WinProbabilityStrategy(simulation_count=0)
    except ValueError as exc:
        assert "simulation_count" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
