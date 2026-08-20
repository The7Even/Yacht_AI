import pytest

from app.ai.benchmark import BenchmarkResult
from app.ai.benchmark_analysis import BenchmarkReport


def sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        strategy_one="Alpha",
        strategy_two="Beta",
        games=4,
        player_one_wins=3,
        player_two_wins=1,
        draws=0,
        player_one_total_score=420,
        player_two_total_score=360,
        strategy_one_started_games=2,
        strategy_two_started_games=2,
        strategy_one_started_wins=1,
        strategy_one_second_player_wins=2,
        strategy_two_started_wins=1,
        strategy_two_second_player_wins=0,
    )


def test_report_aggregates_both_strategies() -> None:
    report = BenchmarkReport.from_results((sample_result(),))

    alpha = report.get("Alpha")
    beta = report.get("Beta")

    assert alpha.games == 4
    assert (alpha.wins, alpha.losses, alpha.draws) == (3, 1, 0)
    assert alpha.points_rate == pytest.approx(0.75)
    assert alpha.average_score == pytest.approx(105.0)
    assert alpha.average_margin == pytest.approx(15.0)
    assert alpha.first_player_win_rate == pytest.approx(0.5)
    assert alpha.second_player_win_rate == pytest.approx(1.0)
    assert alpha.first_player_advantage == pytest.approx(-0.5)

    assert beta.wins == 1
    assert beta.losses == 3
    assert beta.points_rate == pytest.approx(0.25)
    assert beta.first_player_win_rate == pytest.approx(0.5)
    assert beta.second_player_win_rate == pytest.approx(0.0)


def test_leaderboard_orders_by_points_rate_then_margin() -> None:
    result = sample_result()
    report = BenchmarkReport.from_results((result,))

    assert [item.strategy for item in report.leaderboard] == ["Alpha", "Beta"]


def test_wilson_interval_is_bounded() -> None:
    stats = BenchmarkReport.from_results((sample_result(),)).get("Alpha")

    low, high = stats.decisive_win_rate_ci95
    assert 0.0 <= low <= stats.decisive_win_rate <= high <= 1.0


def test_empty_report_has_no_leaderboard() -> None:
    report = BenchmarkReport.from_results(())

    assert report.stats == ()
    assert report.leaderboard == ()


def test_unknown_strategy_raises_key_error() -> None:
    report = BenchmarkReport.from_results((sample_result(),))

    with pytest.raises(KeyError):
        report.get("Missing")
