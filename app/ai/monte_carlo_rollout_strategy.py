"""Fast Yacht-aware continuation policy for Monte Carlo game rollouts."""

from app.core.categories import Category, UPPER_BONUS_THRESHOLD
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionType, DecisionResult
from .fast_expected_value_strategy import FastExpectedValueStrategy


class MonteCarloRolloutStrategy:
    """Strong continuation policy used inside Monte Carlo game rollouts.

    The rollout uses the exact one-step EV policy for ordinary decisions, while
    preserving a small set of game-state invariants that must dominate a noisy
    one-step EV estimate. In particular, taking the upper-section bonus on the
    final roll is treated as a deterministic scoring opportunity.
    """

    def __init__(self) -> None:
        self._fast_ev = FastExpectedValueStrategy()

    def decide(self, state: GameState) -> DecisionResult:
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("MonteCarloRolloutStrategy requires a rolled hand.")

        bonus_action = self._bonus_scoring_action(state)
        if bonus_action is not None:
            return bonus_action

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

    @staticmethod
    def _bonus_scoring_action(state: GameState) -> DecisionResult | None:
        """Bank an upper category immediately when it crosses the bonus threshold."""
        if state.roll_count < MAX_ROLLS_PER_TURN or state.current_dice is None:
            return None

        player = state.players[state.current_player]
        if player.has_upper_bonus:
            return None

        candidates: list[tuple[int, Category]] = []
        for category in Category:
            if not category.is_upper or category in player.used_categories:
                continue
            score = ScoreCalculator.calculate(category, state.current_dice)
            if score <= 0:
                continue
            if player.upper_total + score >= UPPER_BONUS_THRESHOLD:
                candidates.append((score, category))

        if not candidates:
            return None

        score, category = max(candidates, key=lambda item: (item[0], item[1].value))
        action = Action(
            type=ActionType.SCORE,
            held_indices=(),
            selected_category=category,
        )
        return DecisionResult(
            action,
            expected_value=float(score),
            alternatives=(),
            reasoning=(
                f"Rollout/EV: bank {category.display_name} for {score}; "
                "this reaches the upper bonus threshold."
            ),
        )
