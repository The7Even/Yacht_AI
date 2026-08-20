import pytest

from app.ai.benchmark import StrategyBenchmark
from app.ai.strategy import RuleBasedStrategy


def test_benchmark_is_reproducible() -> None:
    benchmark = StrategyBenchmark(seed=42)

    first = benchmark.run(RuleBasedStrategy, RuleBasedStrategy, games=3)
    second = benchmark.run(RuleBasedStrategy, RuleBasedStrategy, games=3)

    assert first == second
    assert first.games == 3
    assert first.player_one_wins + first.player_two_wins + first.draws == 3


def test_match_scores_are_valid() -> None:
    result = StrategyBenchmark(seed=7).play_match(
        RuleBasedStrategy,
        RuleBasedStrategy,
        seed=7,
    )

    assert result.player_score >= 0
    assert result.ai_score >= 0
    assert result.player_score != result.ai_score or result.draw


def test_round_robin_runs_both_player_orders() -> None:
    results = StrategyBenchmark(seed=100).round_robin(
        {"rule_a": RuleBasedStrategy, "rule_b": RuleBasedStrategy},
        games_per_matchup=2,
    )

    assert len(results) == 2
    assert all(result.games == 2 for result in results)
    assert {result.strategy_one for result in results} == {"RuleBasedStrategy"}
    assert {result.strategy_two for result in results} == {"RuleBasedStrategy"}


def test_invalid_game_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        StrategyBenchmark().run(RuleBasedStrategy, RuleBasedStrategy, games=0)


def test_round_robin_requires_two_strategies() -> None:
    with pytest.raises(ValueError):
        StrategyBenchmark().round_robin({"only": RuleBasedStrategy}, games_per_matchup=1)
