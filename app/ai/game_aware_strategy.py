"""Strategic EV augmented with a non-predictive game-position evaluation."""

from app.core.game_state import GameState

from .action_generator import ActionAlternative, DecisionResult
from .game_state_evaluator import GameStateEvaluator
from .strategic_expected_value_strategy import StrategicExpectedValueStrategy


class GameAwareStrategicExpectedValueStrategy(StrategicExpectedValueStrategy):
    """Adds score-lead and opponent-future context while preserving dice EV choices.

    The position value is intentionally action-independent in this phase.  It
    changes the evaluation shown to callers but does not fabricate a risk model
    before win probability or action variance is available.
    """

    def __init__(self, game_state_evaluator: GameStateEvaluator | None = None) -> None:
        super().__init__()
        self._game_state_evaluator = game_state_evaluator or GameStateEvaluator()

    def decide(self, state: GameState) -> DecisionResult:
        decision = super().decide(state)
        context = self._game_state_evaluator.evaluate(state)
        alternatives = tuple(
            ActionAlternative(action=alternative.action, expected_value=alternative.expected_value + context.position_value)
            for alternative in decision.alternatives
        )
        expected_value = (decision.expected_value or 0.0) + context.position_value
        return DecisionResult(
            action=decision.action,
            expected_value=expected_value,
            alternatives=alternatives,
            reasoning=(
                f"{decision.reasoning} Position context: {context.position_value:+.2f} "
                f"with {context.remaining_turns} categories remaining."
            ),
        )
