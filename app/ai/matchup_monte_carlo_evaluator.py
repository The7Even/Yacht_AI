"""Matchup-aware Monte Carlo evaluator for WinProbabilityStrategy."""

from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId

from .ai_player import AIPlayer
from .monte_carlo_evaluator import MonteCarloWinProbabilityEvaluator


class MatchupAwareMonteCarloWinProbabilityEvaluator(MonteCarloWinProbabilityEvaluator):
    """Use the supplied opponent strategy throughout the simulated game.

    The base evaluator intentionally falls back to RuleBasedStrategy after the
    perspective player's bounded strong continuation. That fallback is useful
    for keeping rollouts cheap, but it also accidentally replaced the opponent
    strategy. This subclass keeps that bounded fallback for our own future
    turns while allowing the opponent model to remain the actual matchup
    strategy for the whole simulated game.
    """

    _OPPONENT_DECISION_BUDGET = 1_000_000

    def _finish_game(
        self,
        engine: GameEngine,
        players: dict[PlayerId, AIPlayer],
        perspective: PlayerId,
        *,
        profile_decisions: bool = False,
    ) -> tuple[int, int, float]:
        strong_turns_remaining = self._strong_continuation_turns
        turn_count = 0
        decision_count = 0
        decision_elapsed = 0.0

        while not engine.is_game_over():
            active_player = engine.state.current_player
            player = players[active_player]

            if active_player is perspective:
                if strong_turns_remaining > 0:
                    strong_decisions_remaining = self._strong_decisions_per_turn
                else:
                    player = AIPlayer(self._fallback_strategy)
                    strong_decisions_remaining = 0
            else:
                # The opponent must use the actual matchup strategy, not the
                # RuleBased fallback used for our own cheap continuation.
                strong_decisions_remaining = self._OPPONENT_DECISION_BUDGET

            turn_decisions, turn_elapsed = self._finish_turn(
                engine,
                player,
                strong_decisions_remaining=strong_decisions_remaining,
                profile_decisions=profile_decisions,
            )
            decision_count += turn_decisions
            decision_elapsed += turn_elapsed
            turn_count += 1

            if active_player is perspective and strong_turns_remaining > 0:
                strong_turns_remaining -= 1
            if not engine.is_game_over():
                engine.end_turn()

        return turn_count, decision_count, decision_elapsed
