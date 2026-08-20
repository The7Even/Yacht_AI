"""Command-line entry point for comparing Yacht AI strategies."""

from __future__ import annotations

import argparse

from app.ai.benchmark import StrategyBenchmark
from app.ai.benchmark_analysis import BenchmarkReport
from app.ai.expected_value_strategy import ExpectedValueStrategy
from app.ai.game_aware_strategy import GameAwareStrategicExpectedValueStrategy
from app.ai.strategy import RuleBasedStrategy
from app.ai.strategic_expected_value_strategy import StrategicExpectedValueStrategy
from app.ai.win_probability_strategy import WinProbabilityStrategy


def strategy_specs(win_probability_simulations: int):
    """Return the built-in strategies used by the benchmark."""
    return (
        ("RuleBased", RuleBasedStrategy),
        ("ExpectedValue", ExpectedValueStrategy),
        ("StrategicEV", StrategicExpectedValueStrategy),
        ("GameAwareEV", GameAwareStrategicExpectedValueStrategy),
        (
            f"WinProbability({win_probability_simulations})",
            lambda: WinProbabilityStrategy(simulation_count=win_probability_simulations),
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Yacht AI strategy benchmarks.")
    parser.add_argument("--games", type=int, default=10, help="Games per strategy pair.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--win-probability-simulations",
        type=int,
        default=25,
        help="Monte Carlo simulations per candidate action for WinProbabilityStrategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.win_probability_simulations <= 0:
        raise SystemExit("--win-probability-simulations must be positive")

    benchmark = StrategyBenchmark(seed=args.seed)
    results = benchmark.round_robin(
        strategy_specs(args.win_probability_simulations),
        games_per_pair=args.games,
    )

    print("Yacht AI strategy benchmark")
    print(f"games per matchup: {args.games}")
    print(f"seed: {args.seed}")
    print()
    print(
        f"{'Strategy A':<22} {'Strategy B':<22} "
        f"{'Win A':>7} {'Win B':>7} {'Draw':>7} {'A rate':>8} {'Avg Δ':>9}"
    )
    print("-" * 90)
    for result in results:
        print(
            f"{result.strategy_one:<22} {result.strategy_two:<22} "
            f"{result.player_one_wins:>7} {result.player_two_wins:>7} {result.draws:>7} "
            f"{result.player_one_win_rate:>7.1%} {result.average_margin:>9.2f}"
        )

    report = BenchmarkReport.from_results(results)
    print()
    print("Leaderboard")
    print(
        f"{'#':>2} {'Strategy':<22} {'Pts%':>7} {'W-L-D':>11} "
        f"{'Avg':>8} {'Avg Δ':>8} {'1st':>8} {'2nd':>8}"
    )
    print("-" * 82)
    for rank, stats in enumerate(report.leaderboard, start=1):
        print(
            f"{rank:>2} {stats.strategy:<22} {stats.points_rate:>6.1%} "
            f"{stats.wins:>3}-{stats.losses:<3}-{stats.draws:<3} "
            f"{stats.average_score:>8.2f} {stats.average_margin:>8.2f} "
            f"{stats.first_player_win_rate:>7.1%} {stats.second_player_win_rate:>7.1%}"
        )


if __name__ == "__main__":
    main()
