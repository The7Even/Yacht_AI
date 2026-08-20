"""Pure score calculation for Yacht hands."""

from collections import Counter
from collections.abc import Iterable

from .categories import Category
from .dice import validate_dice


SMALL_STRAIGHT_SCORE = 15
LARGE_STRAIGHT_SCORE = 30
YACHT_SCORE = 50


class ScoreCalculator:
    """Calculates Yacht category scores without changing game state."""

    @classmethod
    def calculate(cls, category: Category, dice: Iterable[int]) -> int:
        """Return the score for *category* and a complete five-die hand."""
        values = validate_dice(dice)
        counts = Counter(values)
        total = sum(values)

        upper_face = category.upper_face
        if upper_face is not None:
            return upper_face * counts[upper_face]
        if category is Category.CHOICE:
            return total
        if category is Category.FOUR_OF_A_KIND:
            return total if max(counts.values()) >= 4 else 0
        if category is Category.FULL_HOUSE:
            return total if cls._is_full_house(counts) else 0
        if category is Category.SMALL_STRAIGHT:
            return SMALL_STRAIGHT_SCORE if cls._has_small_straight(counts) else 0
        if category is Category.LARGE_STRAIGHT:
            return LARGE_STRAIGHT_SCORE if cls._has_large_straight(counts) else 0
        if category is Category.YACHT:
            return YACHT_SCORE if len(counts) == 1 else 0
        raise ValueError(f"Unsupported category: {category!r}")

    @classmethod
    def calculate_all(cls, dice: Iterable[int]) -> dict[Category, int]:
        """Return scores for every category for a single hand."""
        values = validate_dice(dice)
        return {category: cls.calculate(category, values) for category in Category}

    @staticmethod
    def _is_full_house(counts: Counter[int]) -> bool:
        frequencies = sorted(counts.values())
        return frequencies == [2, 3] or frequencies == [5]

    @staticmethod
    def _has_small_straight(counts: Counter[int]) -> bool:
        faces = set(counts)
        return any(set(run).issubset(faces) for run in ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)))

    @staticmethod
    def _has_large_straight(counts: Counter[int]) -> bool:
        faces = set(counts)
        return faces == {1, 2, 3, 4, 5} or faces == {2, 3, 4, 5, 6}
