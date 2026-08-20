"""Fast, one-step expected-value strategy for benchmark runs.

This strategy is intentionally separate from ExpectedValueStrategy.  It evaluates
all legal hold choices, but only one reroll ahead, making benchmark runs practical
without changing the exact strategy used by the application.
"""

from itertools import product

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT, MAX_FACE, MIN_FACE, validate_dice
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionAlternative, ActionGenerator, ActionType, DecisionResult


class FastExpectedValueStrategy:
    """Choose the best immediate score or one-step reroll EV."""

    def decide(self, state: GameState) -> DecisionResult:
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("FastExpectedValueStrategy requires a rolled hand.")

        dice = validate_dice(state.current_dice)
        available = tuple(
            category
            for category in ALL_CATEGORIES
            if category not in state.players[state.current_player].used_categories
        )
        if not available:
            raise ValueError("The active player has no categories available.")

        score_values = {
            category: ScoreCalculator.calculate(category, dice) for category in available
        }
        best_category = max(available, key=lambda category: (score_values[category], -ALL_CATEGORIES.index(category)))
        candidates = [
            ActionAlternative(
                Action(ActionType.SCORE, selected_category=best_category),
                float(score_values[best_category]),
            )
        ]

        if state.roll_count < MAX_ROLLS_PER_TURN:
            for action in ActionGenerator.reroll_actions(state.held_indices):
                if len(action.held_indices) == DICE_COUNT:
                    continue
                value = self._one_step_value(dice, action.held_indices, available)
                candidates.append(ActionAlternative(action, value))

        candidates.sort(key=lambda candidate: (
            candidate.expected_value,
            1 if candidate.action.type is ActionType.SCORE else 0,
            len(candidate.action.held_indices),
        ), reverse=True)
        best = candidates[0]
        return DecisionResult(
            action=best.action,
            expected_value=best.expected_value,
            alternatives=tuple(candidates[:3]),
            reasoning=f"Fast one-step EV: {best.expected_value:.2f}",
        )

    @staticmethod
    def _one_step_value(
        dice: tuple[int, ...], held_indices: tuple[int, ...], available: tuple[Category, ...]
    ) -> float:
        rerolled = tuple(index for index in range(DICE_COUNT) if index not in held_indices)
        if not rerolled:
            return float(max(ScoreCalculator.calculate(category, dice) for category in available))

        total = 0.0
        outcomes = 6 ** len(rerolled)
        for faces in product(range(MIN_FACE, MAX_FACE + 1), repeat=len(rerolled)):
            result = list(dice)
            for index, face in zip(rerolled, faces, strict=True):
                result[index] = face
            total += max(ScoreCalculator.calculate(category, result) for category in available)
        return total / outcomes
