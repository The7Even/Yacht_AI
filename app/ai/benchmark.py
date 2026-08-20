"""AI-vs-AI benchmark utilities for comparing Yacht strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.ai.ai_player import AIPlayer
from app.ai.strategy import Strategy
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one completed AI-vs-AI game."""

    player_score: int
    ai_score: int
    winner: PlayerId | None

    @property
    def draw(self) -> bool:
        return self.winner is None


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


StrategyFactory = Callable[[], Strategy]


class StrategyBenchmark:
    """Run reproducible AI-vs-AI matches."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def play_match(self, strategy_one: StrategyFactory, strategy_two: StrategyFactory, seed: int) -> MatchResult:
        engine = GameEngine(dice_roller=DiceRoller(seed=seed))
        state = engine.start_game()
        strategies = {
            PlayerId.PLAYER: strategy_one(),
            PlayerId.AI: strategy_two(),
        }

        while not state.game_over:
            player = state.current_player
            ai = AIPlayer(strategy=strategies[player])
            engine.roll_dice()

            while not state.turn_scored:
                decision = ai.decide(state)
                action = decision.action
                if action.type.value == "score":
                    engine.score_category(action.selected_category)
                    break
                engine.roll_dice()

            if not state.game_over:
                engine.end_turn()

        player_score = state.player_score
        ai_score = state.ai_score
        winner = (
            PlayerId.PLAYER
            if player_score > ai_score
            else PlayerId.AI
            if ai_score > player_score
            else None
        )
        return MatchResult(player_score, ai_score, winner)

    def run(self, strategy_one: StrategyFactory, strategy_two: StrategyFactory, games: int = 100) -> BenchmarkResult:
        if games <= 0:
            raise ValueError("games must be positive")

        one_wins = two_wins = draws = 0
        one_score = two_score = 0
        for index in range(games):
            result = self.play_match(strategy_one, strategy_two, self.seed + index)
            one_score += result.player_score
            two_score += result.ai_score
            if result.winner is PlayerId.PLAYER:
                one_wins += 1
            elif result.winner is PlayerId.AI:
                two_wins += 1
            else:
                draws += 1

        return BenchmarkResult(
            strategy_one=strategy_one().__class__.__name__,
            strategy_two=strategy_two().__class__.__name__,
            games=games,
            player_one_wins=one_wins,
            player_two_wins=two_wins,
            draws=draws,
            player_one_total_score=one_score,
            player_two_total_score=two_score,
        )
