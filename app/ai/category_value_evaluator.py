"""Category-selection evaluator with optional trained neural inference.

The supplied neural network is a category-selection model: five dice are
encoded into 30 one-hot features and the network emits 12 values, one for each
Yacht score category.  FastEV still decides rerolls; this module lets the
trained model take over the final category choice when its model file exists.
"""

from collections.abc import Iterable
from pathlib import Path

from app.core.categories import ALL_CATEGORIES, Category
from app.core.scoring import ScoreCalculator

from .neural_category_selector import DEFAULT_MODEL_PATH, NeuralCategorySelector


# These are calibration values rather than claims of exact mathematical EV.
CHOICE_OPPORTUNITY_COST = 3.0
FOUR_OF_A_KIND_PRESERVATION_VALUE = 4.0
FULL_HOUSE_PRESERVATION_VALUE = 3.0
NEAR_CHOICE_GAP = 3


class CategoryValueEvaluator:
    """Choose score categories using the trained model when available.

    ``use_neural_model`` is enabled by default. If the model file is absent,
    behavior falls back to the previous conservative opportunity-cost logic.
    This makes the GitHub code runnable before the local ``.pth`` is copied in.
    """

    def __init__(
        self,
        use_neural_model: bool = True,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self._neural_selector: NeuralCategorySelector | None = None
        if use_neural_model and Path(model_path).exists():
            self._neural_selector = NeuralCategorySelector(model_path)

    @property
    def neural_model_enabled(self) -> bool:
        """Whether the trained network is currently available for inference."""
        return self._neural_selector is not None

    def adjusted_score(
        self,
        dice: Iterable[int],
        category: Category,
        available_categories: Iterable[Category],
    ) -> float:
        """Return the value used by FastEV for a score-now candidate.

        When the neural model is enabled, its raw output is used for the
        selected category so the model's preference is preserved. Otherwise
        the conservative opportunity-cost evaluator is used.
        """
        values = tuple(dice)
        available = tuple(available_categories)
        if category not in available:
            raise ValueError("Cannot evaluate a category that is not available.")

        if self._neural_selector is not None:
            _, _, predictions = self._neural_selector.select(values, available)
            return predictions[category]

        immediate = ScoreCalculator.calculate(category, values)
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
        """Return the highest-valued legal category."""
        available = tuple(available_categories)
        if not available:
            raise ValueError("At least one category must be available.")

        values = tuple(dice)
        if self._neural_selector is not None:
            category, _, _ = self._neural_selector.select(values, available)
            return category

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
        """Return model or opportunity-cost values for every legal category."""
        available = tuple(available_categories)
        if self._neural_selector is not None:
            values = tuple(dice)
            _, _, predictions = self._neural_selector.select(values, available)
            return {category: predictions[category] for category in available}
        return {
            category: self.adjusted_score(dice, category, available)
            for category in available
        }
