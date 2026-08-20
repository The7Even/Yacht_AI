from app.ai.fast_expected_value_strategy import FastExpectedValueStrategy
from app.ai.win_probability_strategy import WinProbabilityStrategy


def test_default_continuation_policy_is_fast_ev() -> None:
    strategy = WinProbabilityStrategy(simulation_count=1)
    assert isinstance(strategy._evaluator._player_strategy, FastExpectedValueStrategy)
