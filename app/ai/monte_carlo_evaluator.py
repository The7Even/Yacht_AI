"""Monte Carlo win-probability estimation using the real game engine."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
import random

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
    """Estimate final win probability with a bounded strategic look-ahead.

    The evaluated player's expensive continuation policy is used for only the
    next few turns. The remainder of each rollout uses the cheap fallback
    policy, preventing candidate evaluation from recursively paying the full
    cost of FastEV for the entire future game.
    """

    def __init__(
        self,
        *,
        player_strategy: Strategy | None = None,
        opponent_strategy: Strategy | None = None,
        seed: int | None = None,
        strong_continuation_turns: int = 2,
    ) -> None:
        if strong_continuation_turns < 0:
            raise ValueError("strong_continuation_turns must be non-negative.")
        self._player_strategy = player_strategy or RuleBasedStrategy()
        self._opponent_strategy = opponent_strategy or RuleBasedStrategy()
        self._fallback_strategy = RuleBasedStrategy()
        self._strong_continuation_turns = strong_continuation_turns
        self._seed = seed

    def estimate_win_probability(
        self,
        game_state: GameState,
        action: Action,
        simulation_count: int = 10_000,
        perspective: PlayerId | None = None,
    ) -> float:
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive.")
        player_id = perspective or game_state.current_player
        if game_state.game_over:
            return self._final_result(game_state, player_id)
        if not game_state.game_started:
            raise ValueError("Monte Carlo evaluation requires a started game.")

        rng = random.Random(self._seed)
        players = self._make_players(player_id)
        total = 0.0
        for _ in range(simulation_count):
            engine = GameEngine(DiceRoller(rng))
            engine.state = deepcopy(game_state)
            self._apply_candidate_action(engine, action)
            self._finish_game(engine, players, player_id)
            total += self._final_result(engine.state, player_id)
        return total / simulation_count

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
        return {
            action: self._estimate_statistics_with_scenario_seeds(
                game_state, action, simulation_count, perspective, scenario_seeds
            )
            for action in candidates
        }

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
        for seed in scenario_seeds:
            engine = GameEngine(DiceRoller(random.Random(seed)))
            engine.state = deepcopy(game_state)
            self._apply_candidate_action(engine, action)
            self._finish_game(engine, players, player_id)
            wins += self._final_result(engine.state, player_id)
            total_score += engine.state.players[player_id].total_score
            total_opponent_score += engine.state.players[opponent_id].total_score

        return ActionStatistics(
            win_probability=wins / simulation_count,
            average_score=total_score / simulation_count,
            average_opponent_score=total_opponent_score / simulation_count,
        )

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

    def _finish_game(
        self,
        engine: GameEngine,
        players: dict[PlayerId, AIPlayer],
        perspective: PlayerId,
    ) -> None:
        strong_turns_remaining = self._strong_continuation_turns
        while not engine.is_game_over():
            active_player = engine.state.current_player
            player = players[active_player]

            if active_player is perspective and strong_turns_remaining <= 0:
                player = AIPlayer(self._fallback_strategy)

            self._finish_turn(engine, player)
            if active_player is perspective and strong_turns_remaining > 0:
                strong_turns_remaining -= 1
            if not engine.is_game_over():
                engine.end_turn()

    @staticmethod
    def _finish_turn(engine: GameEngine, player: AIPlayer) -> None:
        while not engine.state.turn_scored:
            if engine.state.current_dice is None:
                engine.roll_dice()
            decision = player.decide(engine.state)
            if decision.action.type is ActionType.SCORE:
                if decision.selected_category is None:
                    raise RuntimeError("Strategy returned an incomplete score action.")
                engine.score_category(decision.selected_category)
            elif decision.action.type is ActionType.REROLL:
                MonteCarloWinProbabilityEvaluator._set_held_indices(engine, frozenset(decision.held_indices))
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
