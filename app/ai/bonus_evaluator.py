"""Heuristic, state-aware evaluation of the Yacht upper-section bonus."""

from dataclasses import dataclass
from collections.abc import Iterable

from app.core.categories import Category, UPPER_BONUS_SCORE, UPPER_BONUS_THRESHOLD

from .upper_expected_value import UpperCategoryExpectedValueEvaluator


@dataclass(frozen=True)
class BonusEvaluation:
    """Breakdown of actual and prospective upper-bonus value."""

    awarded_value: float = 0.0
    progress_value: float = 0.0
    expected_future_upper_score: float = 0.0

    @property
    def total_value(self) -> float:
        return self.awarded_value + self.progress_value


class BonusEvaluator:
    """Assigns a small, explainable value to bonus-progressing upper scores."""

    _MAX_PROGRESS_FRACTION = 0.25

    def __init__(
        self, upper_expected_value_evaluator: UpperCategoryExpectedValueEvaluator | None = None
    ) -> None:
        self._upper_expected_value_evaluator = (
            upper_expected_value_evaluator or UpperCategoryExpectedValueEvaluator()
        )

    def evaluate(
        self,
        *,
        upper_total: int,
        remaining_upper_categories: Iterable[Category],
        selected_category: Category,
        category_score: int,
        has_bonus: bool = False,
    ) -> BonusEvaluation:
        """Evaluate the bonus impact of committing one category score.

        A score that crosses 63 receives the actual +35.  Before that point,
        only an attainable, bounded progress value is returned; it cannot
        dominate a materially better immediate lower-section score.
        """
        remaining = tuple(category for category in remaining_upper_categories if category.is_upper)
        if has_bonus or upper_total >= UPPER_BONUS_THRESHOLD or not remaining:
            return BonusEvaluation()
        if selected_category not in remaining or not selected_category.is_upper:
            return BonusEvaluation()

        projected_total = upper_total + category_score
        if projected_total >= UPPER_BONUS_THRESHOLD:
            return BonusEvaluation(awarded_value=float(UPPER_BONUS_SCORE))

        remaining_after = tuple(category for category in remaining if category is not selected_category)
        maximum_possible_total = projected_total + sum(category.upper_face * 5 for category in remaining_after)
        if maximum_possible_total < UPPER_BONUS_THRESHOLD:
            return BonusEvaluation()

        expected_future_upper_score = self._upper_expected_value_evaluator.expected_remaining_upper_score(
            remaining_after
        )
        points_still_needed = UPPER_BONUS_THRESHOLD - projected_total
        progress_fraction = min(expected_future_upper_score / points_still_needed, 1.0)
        return BonusEvaluation(
            progress_value=(
                UPPER_BONUS_SCORE
                * self._MAX_PROGRESS_FRACTION
                * progress_fraction
            ),
            expected_future_upper_score=expected_future_upper_score,
        )
