"""Expected-value strategy augmented with upper-section bonus value."""

from collections.abc import Iterable
from functools import lru_cache

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import validate_dice
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionAlternative, ActionType, DecisionResult
from .bonus_evaluator import BonusEvaluator
from .expected_value_strategy import ExpectedValueStrategy


class StrategicExpectedValueStrategy(ExpectedValueStrategy):
    """Exact reroll EV plus a bounded, state-dependent upper-bonus value."""

    def __init__(self, bonus_evaluator: BonusEvaluator | None = None) -> None:
        self._bonus_evaluator = bonus_evaluator or BonusEvaluator()

    def decide(self, state: GameState) -> DecisionResult:
        """Choose a score or reroll action using the active player's bonus state."""
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("StrategicExpectedValueStrategy requires a rolled hand.")
        available = self._available_categories(state)
        if not available:
            raise ValueError("The active player has no categories available.")

        player = state.players[state.current_player]
        dice = validate_dice(state.current_dice)
        remaining_rolls = MAX_ROLLS_PER_TURN - state.roll_count
        terminal_category = self._best_strategic_category(
            dice, available, player.upper_total, player.has_upper_bonus
        )
        candidates = [
            ActionAlternative(
                Action(ActionType.SCORE, selected_category=terminal_category),
                self.evaluate_strategic_terminal_state(
                    dice, available, player.upper_total, player.has_upper_bonus
                ),
            )
        ]
        if remaining_rolls > 0:
            for action in self._meaningful_reroll_actions(state.held_indices):
                candidates.append(
                    ActionAlternative(
                        action,
                        self._strategic_expected_value_for_hold(
                            dice,
                            action.held_indices,
                            remaining_rolls,
                            available,
                            player.upper_total,
                            player.has_upper_bonus,
                        ),
                    )
                )

        candidates.sort(key=self._candidate_sort_key, reverse=True)
        best = candidates[0]
        return DecisionResult(
            action=best.action,
            expected_value=best.expected_value,
            alternatives=tuple(candidates[:3]),
            reasoning=self._strategic_reasoning(best.action, best.expected_value, player.upper_total),
        )

    def evaluate_category_value(
        self,
        dice: Iterable[int],
        category: Category,
        available_categories: Iterable[Category],
        upper_total: int,
        has_bonus: bool,
    ) -> float:
        """Return the immediate score plus this category's bonus contribution."""
        values = validate_dice(dice)
        available = self._category_key(available_categories)
        if category not in available:
            raise ValueError("Cannot evaluate a category that is not available.")
        immediate_score = ScoreCalculator.calculate(category, values)
        bonus = self._bonus_evaluator.evaluate(
            upper_total=upper_total,
            remaining_upper_categories=available,
            selected_category=category,
            category_score=immediate_score,
            has_bonus=has_bonus,
        )
        return float(immediate_score) + bonus.total_value

    def evaluate_strategic_terminal_state(
        self,
        dice: Iterable[int],
        available_categories: Iterable[Category],
        upper_total: int,
        has_bonus: bool,
    ) -> float:
        """Return the greatest immediate-plus-bonus value among available categories."""
        available = self._category_key(available_categories)
        if not available:
            raise ValueError("At least one category must be available.")
        return max(
            self.evaluate_category_value(dice, category, available, upper_total, has_bonus)
            for category in available
        )

    def _strategic_expected_value_for_hold(
        self,
        dice: tuple[int, ...],
        held_indices: tuple[int, ...],
        remaining_rolls: int,
        available: tuple[Category, ...],
        upper_total: int,
        has_bonus: bool,
    ) -> float:
        outcomes = self.enumerate_reroll_outcomes(dice, held_indices)
        return sum(
            self._strategic_state_value(
                tuple(sorted(outcome)),
                remaining_rolls - 1,
                available,
                upper_total,
                has_bonus,
            )
            for outcome in outcomes
        ) / len(outcomes)

    @lru_cache(maxsize=None)
    def _strategic_state_value(
        self,
        sorted_dice: tuple[int, ...],
        remaining_rolls: int,
        available: tuple[Category, ...],
        upper_total: int,
        has_bonus: bool,
    ) -> float:
        terminal_value = self.evaluate_strategic_terminal_state(
            sorted_dice, available, upper_total, has_bonus
        )
        if remaining_rolls == 0:
            return terminal_value
        reroll_value = max(
            self._strategic_expected_value_for_hold(
                sorted_dice,
                action.held_indices,
                remaining_rolls,
                available,
                upper_total,
                has_bonus,
            )
            for action in self._meaningful_reroll_actions()
        )
        return max(terminal_value, reroll_value)

    def _best_strategic_category(
        self,
        dice: tuple[int, ...],
        available: tuple[Category, ...],
        upper_total: int,
        has_bonus: bool,
    ) -> Category:
        category_order = {category: index for index, category in enumerate(ALL_CATEGORIES)}
        return max(
            available,
            key=lambda category: (
                self.evaluate_category_value(dice, category, available, upper_total, has_bonus),
                -category_order[category],
            ),
        )

    @staticmethod
    def _strategic_reasoning(action: Action, expected_value: float, upper_total: int) -> str:
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            return (
                f"Record {action.selected_category.display_name}; value {expected_value:.2f} "
                f"with upper total {upper_total}."
            )
        held = ", ".join(str(index) for index in action.held_indices) or "none"
        return f"Keep dice at indices [{held}] for strategic value {expected_value:.2f}."
