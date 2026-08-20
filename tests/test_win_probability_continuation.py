from app.ai.monte_carlo_rollout_strategy import MonteCarloRolloutStrategy
from app.ai.win_probability_strategy import WinProbabilityStrategy


def test_default_continuation_policy_is_monte_carlo_rollout() -> None:
    strategy = WinProbabilityStrategy(simulation_count=1)
    assert isinstance(strategy._evaluator._player_strategy, MonteCarloRolloutStrategy)
