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
        """Estimate the chance that *perspective* wins after applying *action*.

        Ties count as 0.5 win credit.  A completed input game is evaluated
        directly and requires no random simulations.
        """
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive.")
        player_id = perspective or game_state.current_player
        if game_state.game_over:
            return self._final_result(game_state, player_id)
        if not game_state.game_started:
            raise ValueError("Monte Carlo evaluation requires a started game.")

        rng = random.Random(self._seed)
        total = 0.0
        for _ in range(simulation_count):
            engine = GameEngine(DiceRoller(rng))
            engine.state = deepcopy(game_state)
            self._apply_candidate_action(engine, action)
            self._finish_game(engine, player_id)
            total += self._final_result(engine.state, player_id)
        return total / simulation_count

    def estimate_actions_win_probability(
        self,
        game_state: GameState,
        actions: Iterable[Action],
        simulation_count: int = 10_000,
        perspective: PlayerId | None = None,
    ) -> dict[Action, float]:
        """Evaluate several independent candidate actions with the same seed policy."""
        return {
            action: self.estimate_win_probability(game_state, action, simulation_count, perspective)
            for action in actions
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

    def _finish_game(self, engine: GameEngine, perspective: PlayerId) -> None:
        """Continue from the candidate action until both scorecards are complete."""
        while not engine.is_game_over():
            strategy = self._player_strategy if engine.state.current_player is perspective else self._opponent_strategy
            self._finish_turn(engine, strategy)
            if not engine.is_game_over():
                engine.end_turn()

    @staticmethod
    def _finish_turn(engine: GameEngine, strategy: Strategy) -> None:
        player = AIPlayer(strategy)
        while not engine.state.turn_scored:
            if engine.state.current_dice is None:
                engine.roll_dice()
            decision = player.decide(engine.state)
            if decision.action.type is ActionType.SCORE:
                if decision.selected_category is None:
                    raise RuntimeError("Strategy returned an incomplete score action.")
                engine.score_category(decision.selected_category)
            elif decision.action.type is ActionType.REROLL:
                MonteCarloWinProbabilityEvaluator._set_held_indices(
                    engine, frozenset(decision.held_indices)
                )
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
