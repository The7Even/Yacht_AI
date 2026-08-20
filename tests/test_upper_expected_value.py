import math

from app.ai.bonus_evaluator import BonusEvaluator
from app.ai.strategic_expected_value_strategy import StrategicExpectedValueStrategy
from app.ai.upper_expected_value import UpperCategoryExpectedValueEvaluator
from app.core.categories import Category


def test_remaining_upper_categories_produce_different_future_expectations() -> None:
    evaluator = UpperCategoryExpectedValueEvaluator()

    high_faces = evaluator.expected_remaining_upper_score((Category.FIVES, Category.SIXES))
    low_faces = evaluator.expected_remaining_upper_score((Category.ONES, Category.TWOS))

    assert high_faces > low_faces


def test_bonus_value_changes_with_upper_total_when_future_categories_are_same() -> None:
    evaluator = BonusEvaluator()
    near = evaluator.evaluate(
        upper_total=50,
        remaining_upper_categories=(Category.FIVES, Category.SIXES),
        selected_category=Category.FIVES,
        category_score=15,
    )
    far = evaluator.evaluate(
        upper_total=20,
        remaining_upper_categories=(Category.FIVES, Category.SIXES),
        selected_category=Category.FIVES,
        category_score=15,
    )

    assert near.total_value > far.total_value
    assert far.expected_future_upper_score > 0.0


def test_current_selection_that_reaches_bonus_still_awards_exactly_35() -> None:
    evaluation = BonusEvaluator().evaluate(
        upper_total=62,
        remaining_upper_categories=(Category.FIVES, Category.SIXES),
        selected_category=Category.FIVES,
        category_score=15,
    )

    assert evaluation.awarded_value == 35.0
    assert evaluation.progress_value == 0.0


def test_already_earned_bonus_is_not_recalculated() -> None:
    evaluation = BonusEvaluator().evaluate(
        upper_total=63,
        remaining_upper_categories=(Category.FIVES, Category.SIXES),
        selected_category=Category.FIVES,
        category_score=15,
        has_bonus=True,
    )

    assert evaluation.total_value == 0.0


def test_current_upper_score_is_not_counted_again_as_future_upper_value() -> None:
    bonus_evaluator = BonusEvaluator()
    bonus = bonus_evaluator.evaluate(
        upper_total=20,
        remaining_upper_categories=(Category.FIVES, Category.SIXES),
        selected_category=Category.FIVES,
        category_score=15,
    )
    value = StrategicExpectedValueStrategy(bonus_evaluator).evaluate_category_value(
        (5, 5, 5, 2, 6),
        Category.FIVES,
        (Category.FIVES, Category.SIXES),
        20,
        False,
    )

    assert bonus.expected_future_upper_score > 0.0
    assert math.isclose(value, 15.0 + bonus.total_value)
    assert value < 15.0 + bonus.expected_future_upper_score
