"""Monte Carlo win-probability estimation using the real game engine."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
import os
import random
import sys
import time

from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId

from .action_generator import Action, ActionType
from .ai_player import AIPlayer
from .strategy import RuleBasedStrategy, Strategy


@dataclass(frozen=True)
class ActionStatistics:
    win_probability: float
    average_score: float
    average_opponent_score: float


class MonteCarloWinProbabilityEvaluator:
    """Estimate final win probability with a bounded strategic look-ahead."""

    def __init__(
        self,
        *,
        player_strategy: Strategy | None = None,
        opponent_strategy: Strategy | None = None,
        seed: int | None = None,
        strong_continuation_turns: int = 2,
        strong_decisions_per_turn: int = 1,
        show_progress: bool = False,
    ) -> None:
        if strong_continuation_turns < 0:
            raise ValueError("strong_continuation_turns must be non-negative.")
        if strong_decisions_per_turn < 0:
            raise ValueError("strong_decisions_per_turn must be non-negative.")
        self._player_strategy = player_strategy or RuleBasedStrategy()
        self._opponent_strategy = opponent_strategy or RuleBasedStrategy()
        self._fallback_strategy = RuleBasedStrategy()
        self._strong_continuation_turns = strong_continuation_turns
        self._strong_decisions_per_turn = strong_decisions_per_turn
        self._seed = seed
        self._show_progress = show_progress
        # Opt-in profiling keeps normal benchmark behaviour and timing unchanged.
        self._profile = os.getenv("WP_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}

    def estimate_win_probability(
        self,
        game_state: GameState,
        action: Action,
        simulation_count: int = 10_000,
        perspective: PlayerId | None = None,
    ) -> float:
        statistics = self._estimate_statistics_with_scenario_seeds(
            game_state,
            action,
            simulation_count,
            perspective,
            self._scenario_seeds(simulation_count),
            progress_label="WP decision · candidate 1/1",
        )
        return statistics.win_probability

    def estimate_actions_win_probability(
        self,
        game_state: GameState,
        actions: Iterable[Action],
        simulation_count: int = 10_000,
        perspective: PlayerId | None = None,
    ) -> dict[Action, float]:
        return {
            action: statistics.win_probability
            for action, statistics in self.estimate_actions_statistics(
                game_state, actions, simulation_count, perspective
            ).items()
        }

    def estimate_actions_statistics(
        self,
        game_state: GameState,
        actions: Iterable[Action],
        simulation_count: int = 10_000,
        perspective: PlayerId | None = None,
    ) -> dict[Action, ActionStatistics]:
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive.")
        candidates = tuple(actions)
        if not candidates:
            return {}

        scenario_seeds = self._scenario_seeds(simulation_count)
        total_work = len(candidates) * simulation_count
        completed = 0
        started = time.perf_counter()
        results: dict[Action, ActionStatistics] = {}

        if self._show_progress:
            self._print_progress(0, total_work, started, detail=f"WP decision · candidate 0/{len(candidates)}")

        for candidate_index, action in enumerate(candidates, start=1):
            candidate_started = time.perf_counter()
            result = self._estimate_statistics_with_scenario_seeds(
                game_state,
                action,
                simulation_count,
                perspective,
                scenario_seeds,
                progress_label=f"WP decision · candidate {candidate_index}/{len(candidates)}",
                progress_base=completed,
                progress_total=total_work,
                progress_started=started,
            )
            candidate_elapsed = time.perf_counter() - candidate_started
            results[action] = result
            completed += simulation_count
            if self._profile:
                print(
                    f"WP PROFILE | candidate {candidate_index}/{len(candidates)} | "
                    f"{candidate_elapsed:.3f}s | {candidate_elapsed / simulation_count:.3f}s/sim | "
                    f"{action}"
                )
            if self._show_progress:
                self._print_progress(
                    completed,
                    total_work,
                    started,
                    detail=f"WP decision · candidate {candidate_index}/{len(candidates)} complete",
                )

        if self._profile:
            elapsed = time.perf_counter() - started
            print(
                f"WP PROFILE | decision total {elapsed:.3f}s | "
                f"{len(candidates)} candidates × {simulation_count} simulations"
            )
        if self._show_progress:
            self._finish_progress_line()
        return results

    def _scenario_seeds(self, simulation_count: int) -> tuple[int, ...]:
        rng = random.Random(self._seed)
        return tuple(rng.randrange(0, 2**63) for _ in range(simulation_count))

    def _estimate_statistics_with_scenario_seeds(
        self,
        game_state: GameState,
        action: Action,
        simulation_count: int,
        perspective: PlayerId | None,
        scenario_seeds: tuple[int, ...],
        progress_label: str = "",
        progress_base: int = 0,
        progress_total: int | None = None,
        progress_started: float | None = None,
    ) -> ActionStatistics:
        player_id = perspective or game_state.current_player
        if game_state.game_over:
            opponent = PlayerId.AI if player_id is PlayerId.PLAYER else PlayerId.PLAYER
            return ActionStatistics(
                self._final_result(game_state, player_id),
                float(game_state.players[player_id].total_score),
                float(game_state.players[opponent].total_score),
            )
        if not game_state.game_started:
            raise ValueError("Monte Carlo evaluation requires a started game.")

        opponent_id = PlayerId.AI if player_id is PlayerId.PLAYER else PlayerId.PLAYER
        players = self._make_players(player_id)
        wins = 0.0
        total_score = 0.0
        total_opponent_score = 0.0
        progress_step = max(1, simulation_count // 20)

        for simulation_index, seed in enumerate(scenario_seeds, start=1):
            engine = GameEngine(DiceRoller(random.Random(seed)))
            engine.state = deepcopy(game_state)
            self._apply_candidate_action(engine, action)
            self._finish_game(engine, players, player_id)
            wins += self._final_result(engine.state, player_id)
            total_score += engine.state.players[player_id].total_score
            total_opponent_score += engine.state.players[opponent_id].total_score

            if (
                self._show_progress
                and progress_total is not None
                and progress_started is not None
                and (simulation_index % progress_step == 0 or simulation_index == simulation_count)
            ):
                completed = progress_base + simulation_index
                self._print_progress(
                    completed,
                    progress_total,
                    progress_started,
                    detail=f"{progress_label} · sim {simulation_index}/{simulation_count}",
                )

        return ActionStatistics(
            win_probability=wins / simulation_count,
            average_score=total_score / simulation_count,
            average_opponent_score=total_opponent_score / simulation_count,
        )

    def _print_progress(
        self,
        completed: int,
        total: int,
        started: float,
        *,
        detail: str | None = None,
    ) -> None:
        elapsed = time.perf_counter() - started
        fraction = completed / total if total else 1.0
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (total - completed) / rate if rate > 0 else 0.0
        width = 32
        filled = int(width * fraction)
        bar = "#" * filled + "." * (width - filled)
        text = (
            f"WP decision [{bar}] {fraction * 100:5.1f}% | "
            f"{self._format_duration(elapsed)} elapsed | "
            f"ETA {self._format_duration(remaining)} | {rate:5.1f}/s"
        )
        if detail:
            text += f" | {detail}"
        self._write_progress_line(text)

    @staticmethod
    def _write_progress_line(text: str) -> None:
        sys.stdout.write("\x1b[2K\r" + text[:220])
        sys.stdout.flush()

    @staticmethod
    def _finish_progress_line() -> None:
        sys.stdout.write("\x1b[2K\r")
        sys.stdout.flush()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _make_players(self, perspective: PlayerId) -> dict[PlayerId, AIPlayer]:
        return {
            PlayerId.PLAYER: AIPlayer(
                self._player_strategy if PlayerId.PLAYER is perspective else self._opponent_strategy
            ),
            PlayerId.AI: AIPlayer(
                self._player_strategy if PlayerId.AI is perspective else self._opponent_strategy
            ),
        }

    def _apply_candidate_action(self, engine: GameEngine, action: Action) -> None:
        if action.type is ActionType.SCORE:
            if action.selected_category is None:
                raise ValueError("Score action requires a category.")
            engine.score_category(action.selected_category)
            return
        if action.type is ActionType.REROLL:
            self._set_held_indices(engine, frozenset(action.held_indices))
            engine.roll_dice()
            return
        raise ValueError("Monte Carlo candidate actions must be SCORE or REROLL.")

    @staticmethod
    def _set_held_indices(engine: GameEngine, desired: frozenset[int]) -> None:
        for index in engine.state.held_indices - desired:
            engine.unhold_dice(index)
        for index in desired - engine.state.held_indices:
            engine.hold_dice(index)

    def _finish_game(self, engine: GameEngine, players: dict[PlayerId, AIPlayer], perspective: PlayerId) -> None:
        strong_turns_remaining = self._strong_continuation_turns
        while not engine.is_game_over():
            active_player = engine.state.current_player
            player = players[active_player]
            strong_decisions_remaining = 0
            if active_player is perspective and strong_turns_remaining > 0:
                strong_decisions_remaining = self._strong_decisions_per_turn
            else:
                player = AIPlayer(self._fallback_strategy)
            self._finish_turn(engine, player, strong_decisions_remaining=strong_decisions_remaining)
            if active_player is perspective and strong_turns_remaining > 0:
                strong_turns_remaining -= 1
            if not engine.is_game_over():
                engine.end_turn()

    def _finish_turn(self, engine: GameEngine, player: AIPlayer, *, strong_decisions_remaining: int = 0) -> None:
        fallback_player = AIPlayer(self._fallback_strategy)
        while not engine.state.turn_scored:
            if engine.state.current_dice is None:
                engine.roll_dice()
            active_player = player if strong_decisions_remaining > 0 else fallback_player
            decision = active_player.decide(engine.state)
            if strong_decisions_remaining > 0:
                strong_decisions_remaining -= 1
            if decision.action.type is ActionType.SCORE:
                if decision.selected_category is None:
                    raise RuntimeError("Strategy returned an incomplete score action.")
                engine.score_category(decision.action.selected_category)
            elif decision.action.type is ActionType.REROLL:
                MonteCarloWinProbabilityEvaluator._set_held_indices(engine, frozenset(decision.action.held_indices))
                engine.roll_dice()
            else:
                raise RuntimeError("Simulation strategies must return SCORE or REROLL actions.")

    @staticmethod
    def _final_result(state: GameState, perspective: PlayerId) -> float:
        opponent = PlayerId.AI if perspective is PlayerId.PLAYER else PlayerId.PLAYER
        my_score = state.players[perspective].total_score
        opponent_score = state.players[opponent].total_score
        if my_score > opponent_score:
            return 1.0
        if my_score < opponent_score:
            return 0.0
        return 0.5
