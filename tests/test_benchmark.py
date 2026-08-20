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
