"""Fast Yacht-aware continuation policy for Monte Carlo game rollouts."""

from app.core.categories import Category
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState, PlayerId
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionType, DecisionResult
from .fast_expected_value_strategy import FastExpectedValueStrategy


class MonteCarloRolloutStrategy:
    """Strong, deterministic continuation policy used inside Monte Carlo rollouts.

    The previous rollout policy used a hand-written reroll heuristic. That made
    the Monte Carlo evaluator simulate a consistently weak player, so the win
    probability estimate was optimizing against a distorted continuation model.
    We keep this class as the rollout-policy seam, but use the cached exact
    one-step EV policy for the actual continuation decision.
    """

    def __init__(self) -> None:
        self._fast_ev = FastExpectedValueStrategy()

    def decide(self, state: GameState) -> DecisionResult:
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("MonteCarloRolloutStrategy requires a rolled hand.")

        result = self._fast_ev.decide(state)
        action = result.action

        if action.type is ActionType.SCORE:
            category = action.selected_category
            assert category is not None
            score = ScoreCalculator.calculate(category, state.current_dice)
            return DecisionResult(
                action,
                expected_value=result.expected_value,
                alternatives=result.alternatives,
                reasoning=f"Rollout/EV: bank {category.display_name} for {score}.",
            )

        held = ", ".join(str(index) for index in action.held_indices) or "none"
        return DecisionResult(
            action,
            expected_value=result.expected_value,
            alternatives=result.alternatives,
            reasoning=f"Rollout/EV: keep dice at indices [{held}] based on exact one-step EV.",
        )
