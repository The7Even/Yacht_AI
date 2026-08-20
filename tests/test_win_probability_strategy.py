import pytest
from copy import deepcopy

from app.ai.fast_expected_value_strategy import FastExpectedValueStrategy
from app.ai.monte_carlo_rollout_strategy import MonteCarloRolloutStrategy
from app.ai.win_probability_strategy import WinProbabilityStrategy

# ... existing tests ...


def test_default_continuation_policy_is_fast_ev() -> None:
    strategy = WinProbabilityStrategy(simulation_count=1)

    assert isinstance(strategy._evaluator._player_strategy, FastExpectedValueStrategy)


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
