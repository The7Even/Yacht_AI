"""Aggregate and rank strategy benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from .benchmark import BenchmarkResult


@dataclass(frozen=True)
class StrategyStats:
    """Aggregate performance statistics for one strategy."""

    strategy: str
    games: int
    wins: int
    losses: int
    draws: int
    total_score: int
    total_opponent_score: int
    started_games: int
    started_wins: int
    second_games: int
    second_wins: int

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def points_rate(self) -> float:
        return self.points / self.games if self.games else 0.0

    @property
    def decisive_games(self) -> int:
        return self.wins + self.losses

    @property
    def decisive_win_rate(self) -> float:
        return self.wins / self.decisive_games if self.decisive_games else 0.0

    @property
    def average_score(self) -> float:
        return self.total_score / self.games if self.games else 0.0

    @property
    def average_margin(self) -> float:
        return (self.total_score - self.total_opponent_score) / self.games if self.games else 0.0

    @property
    def first_player_win_rate(self) -> float:
        return self.started_wins / self.started_games if self.started_games else 0.0

    @property
    def second_player_win_rate(self) -> float:
        return self.second_wins / self.second_games if self.second_games else 0.0

    @property
    def first_player_advantage(self) -> float:
        """Difference between first- and second-player win rates."""
        if not self.started_games or not self.second_games:
            return 0.0
        return self.first_player_win_rate - self.second_player_win_rate

    @property
    def decisive_win_rate_ci95(self) -> tuple[float, float]:
        """Wilson 95% interval for wins among decisive games."""
        n = self.decisive_games
        if n == 0:
            return (0.0, 0.0)
        z = 1.959963984540054
        p = self.decisive_win_rate
        denominator = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denominator
        margin = z * sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class BenchmarkReport:
    """A complete analysis of one round-robin benchmark."""

    stats: tuple[StrategyStats, ...]

    @classmethod
    def from_results(cls, results: Iterable[BenchmarkResult]) -> "BenchmarkReport":
        aggregates: dict[str, dict[str, int]] = {}

        def bucket(name: str) -> dict[str, int]:
            return aggregates.setdefault(
                name,
                {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "score": 0,
                    "opponent_score": 0,
                    "started_games": 0,
                    "started_wins": 0,
                    "second_games": 0,
                    "second_wins": 0,
                },
            )

        for result in results:
            if result.games <= 0:
                raise ValueError("Benchmark results must contain at least one game")

            one = bucket(result.strategy_one)
            two = bucket(result.strategy_two)

            one["games"] += result.games
            one["wins"] += result.player_one_wins
            one["losses"] += result.player_two_wins
            one["draws"] += result.draws
            one["score"] += result.player_one_total_score
            one["opponent_score"] += result.player_two_total_score
            one["started_games"] += result.strategy_one_started_games
            one["started_wins"] += result.strategy_one_started_wins
            one["second_games"] += result.strategy_two_started_games
            one["second_wins"] += result.strategy_one_second_player_wins

            two["games"] += result.games
            two["wins"] += result.player_two_wins
            two["losses"] += result.player_one_wins
            two["draws"] += result.draws
            two["score"] += result.player_two_total_score
            two["opponent_score"] += result.player_one_total_score
            two["started_games"] += result.strategy_two_started_games
            two["started_wins"] += result.strategy_two_started_wins
            two["second_games"] += result.strategy_one_started_games
            two["second_wins"] += result.strategy_two_second_player_wins

        stats = tuple(
            StrategyStats(
                strategy=name,
                games=data["games"],
                wins=data["wins"],
                losses=data["losses"],
                draws=data["draws"],
                total_score=data["score"],
                total_opponent_score=data["opponent_score"],
                started_games=data["started_games"],
                started_wins=data["started_wins"],
                second_games=data["second_games"],
                second_wins=data["second_wins"],
            )
            for name, data in aggregates.items()
        )
        return cls(stats=stats)

    @property
    def leaderboard(self) -> tuple[StrategyStats, ...]:
        """Return strategies ordered by benchmark points rate and score margin."""
        return tuple(
            sorted(
                self.stats,
                key=lambda item: (-item.points_rate, -item.average_margin, item.strategy),
            )
        )

    def get(self, strategy: str) -> StrategyStats:
        for item in self.stats:
            if item.strategy == strategy:
                return item
        raise KeyError(strategy)
