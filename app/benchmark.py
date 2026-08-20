"""Command-line entry point for comparing Yacht AI strategies."""

from __future__ import annotations

import argparse

from app.ai.benchmark import StrategyBenchmark
from app.ai.benchmark_analysis import BenchmarkReport
from app.ai.expected_value_strategy import ExpectedValueStrategy
from app.ai.fast_expected_value_strategy import FastExpectedValueStrategy
from app.ai.game_aware_strategy import GameAwareStrategicExpectedValueStrategy
from app.ai.strategy import RuleBasedStrategy
from app.ai.strategic_expected_value_strategy import StrategicExpectedValueStrategy
from app.ai.win_probability_strategy import WinProbabilityStrategy


def strategy_specs(
    win_probability_simulations: int,
    win_probability_candidates: int | None = None,
    *,
    quick: bool = False,
):
    """Return the built-in strategies used by the benchmark.

    Quick mode deliberately uses a separate one-step EV implementation so the
    exact research strategies remain untouched and available for precise runs.
    """
    if quick:
        return (
            ("RuleBased", RuleBasedStrategy),
            ("FastEV", FastExpectedValueStrategy),
            (
                f"WinProbability({win_probability_simulations})",
                lambda: WinProbabilityStrategy(
                    simulation_count=win_probability_simulations,
                    max_candidates=win_probability_candidates,
                ),
            ),
        )
    return (
        ("RuleBased", RuleBasedStrategy),
        ("ExpectedValue", ExpectedValueStrategy),
        ("StrategicEV", StrategicExpectedValueStrategy),
        ("GameAwareEV", GameAwareStrategicExpectedValueStrategy),
        (
            f"WinProbability({win_probability_simulations})",
            lambda: WinProbabilityStrategy(
                simulation_count=win_probability_simulations,
                max_candidates=win_probability_candidates,
            ),
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Yacht AI strategy benchmarks.")
    parser.add_argument("--games", type=int, default=10, help="Games per strategy pair.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--quick", action="store_true", help="Use fast benchmark strategies for development checks."
    )
    parser.add_argument(
        "--win-probability-simulations", type=int, default=5,
        help="Monte Carlo simulations per candidate action for WinProbabilityStrategy.",
    )
    parser.add_argument(
        "--win-probability-candidates", type=int, default=12,
        help="Maximum candidate actions evaluated by WinProbabilityStrategy; 0 means all.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.win_probability_simulations <= 0:
        raise SystemExit("--win-probability-simulations must be positive")
    if args.win_probability_candidates < 0:
        raise SystemExit("--win-probability-candidates must be non-negative")

    candidates = args.win_probability_candidates or None
    specs = strategy_specs(args.win_probability_simulations, candidates, quick=args.quick)
    benchmark = StrategyBenchmark(seed=args.seed)

    print("Yacht AI strategy benchmark", flush=True)
    print(f"mode: {'QUICK' if args.quick else 'EXACT'}", flush=True)
    print(f"games per matchup: {args.games}", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        f"win probability: {args.win_probability_simulations} simulations, "
        f"{candidates or 'all'} candidates", flush=True,
    )
    if args.quick:
        print("quick mode: FastEV uses one-step expected value; exact strategies are unchanged.", flush=True)
    print()
    print(
        f"{'Strategy A':<22} {'Strategy B':<22} {'Win A':>7} {'Win B':>7} "
        f"{'Draw':>7} {'A rate':>8} {'Avg Δ':>9}", flush=True,
    )
    print("-" * 90, flush=True)

    results = []
    pair_count = len(specs) * (len(specs) - 1) // 2
    completed = 0
    for index, (name_one, factory_one) in enumerate(specs):
        for name_two, factory_two in specs[index + 1 :]:
            completed += 1
            print(f"[{completed}/{pair_count}] {name_one} vs {name_two} ...", flush=True)
            result = benchmark.run(
                factory_one, factory_two, games=args.games, alternate_first_player=True
            )
            results.append(result)
            print(
                f"    {result.player_one_wins:>3} - {result.player_two_wins:<3} "
                f"({result.draws} draw), {result.player_one_win_rate:.1%} / "
                f"{result.player_two_win_rate:.1%}", flush=True,
            )

    report = BenchmarkReport.from_results(results)
    print()
    print("Leaderboard", flush=True)
    print(
        f"{'#':>2} {'Strategy':<22} {'Pts%':>7} {'W-L-D':>11} {'Avg':>8} "
        f"{'Avg Δ':>8} {'1st':>8} {'2nd':>8}", flush=True,
    )
    print("-" * 82, flush=True)
    for rank, stats in enumerate(report.leaderboard, start=1):
        print(
            f"{rank:>2} {stats.strategy:<22} {stats.points_rate:>6.1%} "
            f"{stats.wins:>3}-{stats.losses:<3}-{stats.draws:<3} "
            f"{stats.average_score:>8.2f} {stats.average_margin:>8.2f} "
            f"{stats.first_player_win_rate:>7.1%} {stats.second_player_win_rate:>7.1%}", flush=True,
        )


if __name__ == "__main__":
    main()
