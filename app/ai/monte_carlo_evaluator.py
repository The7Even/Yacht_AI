"""Monte Carlo win-probability estimation using the real game engine."""

from collections.abc import Iterable
from copy import deepcopy
import random

from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId

from .action_generator import Action, ActionType
from .ai_player import AIPlayer
from .strategy import RuleBasedStrategy, Strategy


class MonteCarloWinProbabilityEvaluator:
    """Estimates final win probability without mutating the supplied game state."""

    def __init__(
        self,
        *,
        player_strategy: Strategy | None = None,
        opponent_strategy: Strategy | None = None,
        seed: int | None = None,
    ) -> None:
        self._player_strategy = player_strategy or RuleBasedStrategy()
        self._opponent_strategy = opponent_strategy or RuleBasedStrategy()
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
            self._finish_game(engine, players)
            total += self._final_result(engine.state, player_id)
        return total / simulation_count

    def estimate_actions_win_probability(
        self,
        game_state: GameState,
        actions: Iterable[Action],
        simulation_count: int = 10_000,
        perspective: PlayerId | None = None,
    ) -> dict[Action, float]:
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive.")
        candidates = tuple(actions)
        if not candidates:
            return {}

        # Common-random-number comparison: each candidate starts each rollout
        # from the same scenario seed. This reduces variance caused purely by
        # different lucky/unlucky future dice streams.
        scenario_seeds = self._scenario_seeds(simulation_count)
        return {
            action: self._estimate_with_scenario_seeds(
                game_state, action, simulation_count, perspective, scenario_seeds
            )
            for action in candidates
        }

    def _scenario_seeds(self, simulation_count: int) -> tuple[int, ...]:
        rng = random.Random(self._seed)
        return tuple(rng.randrange(0, 2**63) for _ in range(simulation_count))

    def _estimate_with_scenario_seeds(
        self,
        game_state: GameState,
        action: Action,
        simulation_count: int,
        perspective: PlayerId | None,
        scenario_seeds: tuple[int, ...],
    ) -> float:
        player_id = perspective or game_state.current_player
        if game_state.game_over:
            return self._final_result(game_state, player_id)
        if not game_state.game_started:
            raise ValueError("Monte Carlo evaluation requires a started game.")

        players = self._make_players(player_id)
        total = 0.0
        for seed in scenario_seeds:
            engine = GameEngine(DiceRoller(random.Random(seed)))
            engine.state = deepcopy(game_state)
            self._apply_candidate_action(engine, action)
            self._finish_game(engine, players)
            total += self._final_result(engine.state, player_id)
        return total / simulation_count

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

    def _finish_game(self, engine: GameEngine, players: dict[PlayerId, AIPlayer]) -> None:
        while not engine.is_game_over():
            active_player = engine.state.current_player
            self._finish_turn(engine, players[active_player])
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
