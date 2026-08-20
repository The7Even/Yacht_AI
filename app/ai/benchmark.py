"""Reproducible AI-vs-AI matchup and round-robin benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable

from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import PlayerId

from .action_generator import ActionType, DecisionResult
from .strategy import Strategy


StrategyFactory = Callable[[], Strategy]


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one completed game from strategy one's perspective."""

    strategy_one_score: int
    strategy_two_score: int
    strategy_one_started: bool

    @property
    def winner(self) -> int:
        """Return 1 for strategy one, -1 for strategy two, or 0 for a draw."""
        if self.strategy_one_score > self.strategy_two_score:
            return 1
        if self.strategy_one_score < self.strategy_two_score:
            return -1
        return 0

    @property
    def draw(self) -> bool:
        return self.winner == 0

    # Backward-compatible seat-oriented aliases for simple callers.
    @property
    def player_score(self) -> int:
        return self.strategy_one_score if self.strategy_one_started else self.strategy_two_score

    @property
    def ai_score(self) -> int:
        return self.strategy_two_score if self.strategy_one_started else self.strategy_one_score


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated results for one strategy matchup."""

    strategy_one: str
    strategy_two: str
    games: int
    player_one_wins: int
    player_two_wins: int
    draws: int
    player_one_total_score: int
    player_two_total_score: int
    strategy_one_started_games: int
    strategy_two_started_games: int
    strategy_one_started_wins: int
    strategy_one_second_player_wins: int

    @property
    def player_one_win_rate(self) -> float:
        return self.player_one_wins / self.games if self.games else 0.0

    @property
    def player_two_win_rate(self) -> float:
        return self.player_two_wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def average_score_one(self) -> float:
        return self.player_one_total_score / self.games if self.games else 0.0

    @property
    def average_score_two(self) -> float:
        return self.player_two_total_score / self.games if self.games else 0.0

    @property
    def strategy_one_win_rate(self) -> float:
        return self.player_one_win_rate

    @property
    def average_margin(self) -> float:
        return self.average_score_one - self.average_score_two


class StrategyBenchmark:
    """Run reproducible AI-vs-AI matches using the real GameEngine."""

    def __init__(self, seed: int | None = 0) -> None:
        self.seed = seed

    def play_match(
        self,
        strategy_one: StrategyFactory,
        strategy_two: StrategyFactory,
        *,
        seed: int | None = None,
        strategy_one_starts: bool = True,
    ) -> MatchResult:
        """Play one complete game and report it from strategy one's perspective."""
        rng = random.Random(seed)
        engine = GameEngine(dice_roller=DiceRoller(rng))
        engine.start_game()

        strategies = {
            PlayerId.PLAYER: strategy_one() if strategy_one_starts else strategy_two(),
            PlayerId.AI: strategy_two() if strategy_one_starts else strategy_one(),
        }

        while not engine.is_game_over():
            active_strategy = strategies[engine.state.current_player]
            self._play_turn(engine, active_strategy)
            if not engine.is_game_over():
                engine.end_turn()

        strategy_one_player = PlayerId.PLAYER if strategy_one_starts else PlayerId.AI
        strategy_two_player = PlayerId.AI if strategy_one_starts else PlayerId.PLAYER
        return MatchResult(
            strategy_one_score=engine.state.players[strategy_one_player].total_score,
            strategy_two_score=engine.state.players[strategy_two_player].total_score,
            strategy_one_started=strategy_one_starts,
        )

    def run(
        self,
        strategy_one: StrategyFactory,
        strategy_two: StrategyFactory,
        games: int = 100,
        *,
        alternate_first_player: bool = True,
    ) -> BenchmarkResult:
        """Run a matchup, balancing first-player advantage by default."""
        if games <= 0:
            raise ValueError("games must be positive")

        one_wins = two_wins = draws = 0
        one_score = two_score = 0
        one_starts = one_second_wins = one_started_wins = 0

        name_one = self._factory_name(strategy_one)
        name_two = self._factory_name(strategy_two)

        for index in range(games):
            strategy_one_starts = not alternate_first_player or index % 2 == 0
            game_seed = None if self.seed is None else self.seed + index
            result = self.play_match(
                strategy_one,
                strategy_two,
                seed=game_seed,
                strategy_one_starts=strategy_one_starts,
            )

            one_score += result.strategy_one_score
            two_score += result.strategy_two_score
            if result.winner > 0:
                one_wins += 1
                if result.strategy_one_started:
                    one_started_wins += 1
                else:
                    one_second_wins += 1
            elif result.winner < 0:
                two_wins += 1
            else:
                draws += 1

            if result.strategy_one_started:
                one_starts += 1

        return BenchmarkResult(
            strategy_one=name_one,
            strategy_two=name_two,
            games=games,
            player_one_wins=one_wins,
            player_two_wins=two_wins,
            draws=draws,
            player_one_total_score=one_score,
            player_two_total_score=two_score,
            strategy_one_started_games=one_starts,
            strategy_two_started_games=games - one_starts,
            strategy_one_started_wins=one_started_wins,
            strategy_one_second_player_wins=one_second_wins,
        )

    def round_robin(
        self,
        strategies: Iterable[tuple[str, StrategyFactory]],
        games_per_pair: int = 20,
        *,
        alternate_first_player: bool = True,
    ) -> tuple[BenchmarkResult, ...]:
        """Run every unordered strategy pair exactly once."""
        specs = tuple(strategies)
        if len(specs) < 2:
            raise ValueError("At least two strategies are required")
        if games_per_pair <= 0:
            raise ValueError("games_per_pair must be positive")

        results: list[BenchmarkResult] = []
        pair_index = 0
        for index, (name_one, factory_one) in enumerate(specs):
            for name_two, factory_two in specs[index + 1 :]:
                pair_seed = None if self.seed is None else self.seed + pair_index * games_per_pair
                benchmark = StrategyBenchmark(pair_seed)
                result = benchmark.run(
                    _named_factory(name_one, factory_one),
                    _named_factory(name_two, factory_two),
                    games_per_pair,
                    alternate_first_player=alternate_first_player,
                )
                results.append(result)
                pair_index += 1
        return tuple(results)

    @staticmethod
    def _play_turn(engine: GameEngine, strategy: Strategy) -> None:
        """Apply strategy decisions until the active participant scores."""
        while not engine.state.turn_scored:
            if engine.state.current_dice is None:
                engine.roll_dice()

            decision: DecisionResult = strategy.decide(engine.state)
            action = decision.action
            if action.type is ActionType.SCORE:
                if action.selected_category is None:
                    raise RuntimeError("Strategy returned an incomplete score action.")
                engine.score_category(action.selected_category)
            elif action.type is ActionType.REROLL:
                StrategyBenchmark._set_held_indices(engine, frozenset(action.held_indices))
                engine.roll_dice()
            else:
                raise RuntimeError("Benchmark strategies must return SCORE or REROLL actions.")

    @staticmethod
    def _set_held_indices(engine: GameEngine, desired: frozenset[int]) -> None:
        for index in engine.state.held_indices - desired:
            engine.unhold_dice(index)
        for index in desired - engine.state.held_indices:
            engine.hold_dice(index)

    @staticmethod
    def _factory_name(factory: StrategyFactory) -> str:
        return getattr(factory, "__benchmark_name__", getattr(factory, "__name__", factory.__class__.__name__))


def _named_factory(name: str, factory: StrategyFactory) -> StrategyFactory:
    """Attach a stable display name without changing the factory behavior."""
    def create() -> Strategy:
        return factory()

    setattr(create, "__benchmark_name__", name)
    return create
