"""Category-selection value adjustments for FastEV.

The base FastEV evaluator is intentionally exact for immediate scoring and
reroll EV.  This module adds a small, explicit opportunity-cost layer only at
category selection time.  It is deliberately conservative: special-category
bonuses apply only when the category is completed and is close to Choice, so
obvious immediate-score decisions such as Large Straight or Yacht are not
changed.
"""

from collections.abc import Iterable

from app.core.categories import Category
from app.core.scoring import ScoreCalculator


# These are calibration values rather than claims of exact mathematical EV.
# The earlier 10k-branch experiments found positive future-value differences
# for 4K and Full House versus Choice, so the first implementation uses a
# smaller, conservative net preference to avoid overfitting to one experiment.
CHOICE_OPPORTUNITY_COST = 3.0
FOUR_OF_A_KIND_PRESERVATION_VALUE = 4.0
FULL_HOUSE_PRESERVATION_VALUE = 3.0
NEAR_CHOICE_GAP = 3


class CategoryValueEvaluator:
    """Apply a bounded opportunity-cost adjustment to score categories."""

    def adjusted_score(
        self,
        dice: Iterable[int],
        category: Category,
        available_categories: Iterable[Category],
    ) -> float:
        """Return immediate score plus a conservative preservation adjustment.

        Choice is treated as a flexible fallback resource and receives a small
        opportunity-cost penalty while it remains available.  Four of a Kind
        and Full House receive preservation value only when they are actually
        completed and their immediate score is close to the currently available
        Choice score.
        """
        values = tuple(dice)
        available = tuple(available_categories)
        immediate = ScoreCalculator.calculate(category, values)

        if category not in available:
            raise ValueError("Cannot evaluate a category that is not available.")

        if category is Category.CHOICE:
            return float(immediate) - CHOICE_OPPORTUNITY_COST

        if Category.CHOICE not in available:
            return float(immediate)

        choice_score = ScoreCalculator.calculate(Category.CHOICE, values)
        score_gap = choice_score - immediate

        if immediate > 0 and score_gap <= NEAR_CHOICE_GAP:
            if category is Category.FOUR_OF_A_KIND:
                return float(immediate) + FOUR_OF_A_KIND_PRESERVATION_VALUE
            if category is Category.FULL_HOUSE:
                return float(immediate) + FULL_HOUSE_PRESERVATION_VALUE

        return float(immediate)

    def best_category(
        self,
        dice: Iterable[int],
        available_categories: Iterable[Category],
    ) -> Category:
        """Return the category with the highest adjusted selection value."""
        available = tuple(available_categories)
        if not available:
            raise ValueError("At least one category must be available.")

        values = tuple(dice)
        priority = {category: index for index, category in enumerate(available)}
        return max(
            available,
            key=lambda category: (
                self.adjusted_score(values, category, available),
                -priority[category],
            ),
        )

    def selection_values(
        self,
        dice: Iterable[int],
        available_categories: Iterable[Category],
    ) -> dict[Category, float]:
        """Return adjusted values for every available category."""
        available = tuple(available_categories)
        return {
            category: self.adjusted_score(dice, category, available)
            for category in available
        }
