"""Exact future-score estimates for still-unused upper-section categories."""

from collections.abc import Iterable
from functools import lru_cache
from itertools import product

from app.core.categories import Category
from app.core.dice import MAX_FACE, MIN_FACE
from app.core.game_engine import MAX_ROLLS_PER_TURN


class UpperCategoryExpectedValueEvaluator:
    """Calculates a category-targeted, exact EV for future upper turns.

    Each remaining upper category is evaluated as an independent future turn.
    The target category may be scored early or after up to two rerolls, matching
    the GameEngine's normal turn rules.
    """

    def expected_category_score(
        self, category: Category, remaining_rolls: int = MAX_ROLLS_PER_TURN - 1
    ) -> float:
        """Return the exact expected score for a fresh turn targeting *category*."""
        if not category.is_upper:
            raise ValueError("Only upper-section categories have an upper EV.")
        if remaining_rolls < 0:
            raise ValueError("remaining_rolls cannot be negative.")
        return self._expected_category_score(category, remaining_rolls)

    def expected_remaining_upper_score(
        self,
        categories: Iterable[Category],
        remaining_rolls: int = MAX_ROLLS_PER_TURN - 1,
    ) -> float:
        """Sum future targeted EVs for the provided, still-unused upper categories."""
        upper_categories = tuple(category for category in categories if category.is_upper)
        return sum(self.expected_category_score(category, remaining_rolls) for category in upper_categories)

    @staticmethod
    @lru_cache(maxsize=None)
    def _expected_category_score(category: Category, remaining_rolls: int) -> float:
        """Average exact category values across all 7,776 initial hands."""
        target_face = category.upper_face
        assert target_face is not None
        outcomes = product(range(MIN_FACE, MAX_FACE + 1), repeat=5)
        total = sum(
            UpperCategoryExpectedValueEvaluator._target_count_value(
                target_face, outcome.count(target_face), remaining_rolls
            )
            for outcome in outcomes
        )
        return total / (MAX_FACE**5)

    @staticmethod
    @lru_cache(maxsize=None)
    def _target_count_value(target_face: int, target_count: int, remaining_rolls: int) -> float:
        """Optimal exact value when only one upper face is being targeted.

        Holding every target face and rerolling every other die dominates all
        other holds: a non-target cannot contribute to this category, while a
        target can only be lost by rerolling it.  This reduces the full dice
        tree to six count states without approximating any probabilities.
        """
        terminal_value = float(target_face * target_count)
        if remaining_rolls == 0 or target_count == 5:
            return terminal_value

        rerolled_count = 5 - target_count
        outcomes = product(range(MIN_FACE, MAX_FACE + 1), repeat=rerolled_count)
        reroll_value = sum(
            UpperCategoryExpectedValueEvaluator._target_count_value(
                target_face,
                target_count + outcome.count(target_face),
                remaining_rolls - 1,
            )
            for outcome in outcomes
        ) / (MAX_FACE**rerolled_count)
        return max(terminal_value, reroll_value)
