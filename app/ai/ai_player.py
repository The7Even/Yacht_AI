"""Adapter that exposes a strategy as a game participant."""

from app.core.game_state import GameState

from .action_generator import DecisionResult
from .strategy import RuleBasedStrategy, Strategy


class AIPlayer:
    """Reads a game state and returns an action without modifying that state."""

    def __init__(self, strategy: Strategy | None = None) -> None:
        self._strategy = strategy or RuleBasedStrategy()

    def decide(self, state: GameState) -> DecisionResult:
        return self._strategy.decide(state)
