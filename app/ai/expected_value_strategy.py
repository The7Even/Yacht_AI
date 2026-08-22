"""Exact, exhaustive expected-value strategy for Yacht rerolls.

The search is mathematically identical to the original exhaustive evaluator, but
uses symmetry reduction internally: dice order is irrelevant to Yacht scoring,
so reroll outcomes are evaluated as unique sorted hands with multinomial
probabilities instead of every ordered dice permutation.
"""

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from math import factorial

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT, MAX_FACE, MIN_FACE, validate_dice
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionAlternative, ActionGenerator, ActionType, DecisionResult


_ALL_HANDS = tuple(combinations_with_replacement(range(MIN_FACE, MAX_FACE + 1), DICE_COUNT))
_CATEGORY_INDEX = {category: index for index, category in enumerate(ALL_CATEGORIES)}

# There are only C(9,5)=252 unordered five-die hands. Precompute every
# category score once instead of constructing Counters for every recursive node.
_SCORE_TABLE = {
    hand: tuple(ScoreCalculator.calculate(category, hand) for category in ALL_CATEGORIES)
    for hand in _ALL_HANDS
}


class ExpectedValueStrategy:
    """Chooses between scoring now and exact EV-maximizing reroll actions.

    Terminal evaluation intentionally contains only immediate category score.  It is
    kept separate so a later strategy can add bonus, risk, or opponent value.
    """

    def decide(self, state: GameState) -> DecisionResult:
        """Return the highest-value valid action for a rolled game state."""
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("ExpectedValueStrategy requires a rolled hand.")
        available = self._available_categories(state)
        if not available:
            raise ValueError("The active player has no categories available.")

        dice = validate_dice(state.current_dice)
        sorted_dice = tuple(sorted(dice))
        remaining_rolls = MAX_ROLLS_PER_TURN - state.roll_count
        terminal_action = Action(
            ActionType.SCORE, selected_category=self._best_category(sorted_dice, available)
        )
        candidates = [
            ActionAlternative(terminal_action, self.evaluate_terminal_state(sorted_dice, available))
        ]

        if remaining_rolls > 0:
            available_key = self._category_key(available)
            # Walk the original 31 index-based actions in stable order, but
            # collapse actions that hold the same multiset of die values.
            # This preserves the old tie-breaking behavior while removing
            # duplicate EV calculations for identical dice.
            seen_held_values: set[tuple[int, ...]] = set()
            for action in self._meaningful_reroll_actions():
                held_values = tuple(sorted(dice[index] for index in action.held_indices))
                if held_values in seen_held_values:
                    continue
                seen_held_values.add(held_values)
                expected_value = self._expected_value_for_held_values(
                    sorted_dice, held_values, remaining_rolls, available_key
                )
                candidates.append(ActionAlternative(action, expected_value))

        candidates.sort(key=self._candidate_sort_key, reverse=True)
        best = candidates[0]
        return DecisionResult(
            action=best.action,
            expected_value=best.expected_value,
            alternatives=tuple(candidates[:3]),
            reasoning=self._reasoning(best.action, best.expected_value),
        )

    @staticmethod
    def enumerate_reroll_outcomes(
        dice: Iterable[int], held_indices: Iterable[int]
    ) -> tuple[tuple[int, ...], ...]:
        """Enumerate every equally likely ordered result of rerolling unheld dice.

        This public method intentionally retains the original ordered-outcome
        behavior because it is useful for tests and external callers. The AI's
        internal evaluator uses symmetry-reduced outcomes instead.
        """
        values = validate_dice(dice)
        held = frozenset(held_indices)
        if any(type(index) is not int or not 0 <= index < DICE_COUNT for index in held):
            raise ValueError("Held indices must be integers from 0 through 4.")
        rerolled_indices = tuple(index for index in range(DICE_COUNT) if index not in held)
        outcomes: list[tuple[int, ...]] = []
        for faces in product(range(MIN_FACE, MAX_FACE + 1), repeat=len(rerolled_indices)):
            result = list(values)
            for index, face in zip(rerolled_indices, faces, strict=True):
                result[index] = face
            outcomes.append(tuple(result))
        return tuple(outcomes)

    @classmethod
    def outcome_probabilities(
        cls, dice: Iterable[int], held_indices: Iterable[int]
    ) -> tuple[tuple[tuple[int, ...], float], ...]:
        """Return exact enumerated ordered outcomes with uniform probabilities."""
        outcomes = cls.enumerate_reroll_outcomes(dice, held_indices)
        probability = 1.0 / len(outcomes)
        return tuple((outcome, probability) for outcome in outcomes)

    def expected_value_for_hold(
        self,
        dice: Iterable[int],
        held_indices: Iterable[int],
        remaining_rolls: int,
        available_categories: Iterable[Category],
    ) -> float:
        """Evaluate one reroll action exactly using symmetry-reduced outcomes."""
        if remaining_rolls <= 0:
            raise ValueError("A reroll requires at least one remaining roll.")
        values = validate_dice(dice)
        held = frozenset(held_indices)
        if any(type(index) is not int or not 0 <= index < DICE_COUNT for index in held):
            raise ValueError("Held indices must be integers from 0 through 4.")
        held_values = tuple(sorted(values[index] for index in held))
        available = self._category_key(available_categories)
        return self._expected_value_for_held_values(
            tuple(sorted(values)), held_values, remaining_rolls, available
        )

    def evaluate_terminal_state(
        self, dice: Iterable[int], available_categories: Iterable[Category]
    ) -> float:
        """Return immediate value of the best still-available score category."""
        values = tuple(sorted(validate_dice(dice)))
        available = self._category_key(available_categories)
        if not available:
            raise ValueError("At least one category must be available.")
        scores = _SCORE_TABLE[values]
        return float(max(scores[_CATEGORY_INDEX[category]] for category in available))

    @classmethod
    @lru_cache(maxsize=100_000)
    def _reroll_distribution(
        cls, held_values: tuple[int, ...]
    ) -> tuple[tuple[tuple[int, ...], float], ...]:
        """Return unique final hands and exact probabilities for one hold set."""
        reroll_count = DICE_COUNT - len(held_values)
        if reroll_count <= 0:
            raise ValueError("At least one die must be rerolled.")

        denominator = 6 ** reroll_count
        factorial_n = factorial(reroll_count)
        outcomes: list[tuple[tuple[int, ...], float]] = []

        for faces in combinations_with_replacement(
            range(MIN_FACE, MAX_FACE + 1), reroll_count
        ):
            multiplicity = factorial_n
            for count in Counter(faces).values():
                multiplicity //= factorial(count)
            result = tuple(sorted(held_values + faces))
            outcomes.append((result, multiplicity / denominator))

        return tuple(outcomes)

    @classmethod
    @lru_cache(maxsize=100_000)
    def _state_value(
        cls, sorted_dice: tuple[int, ...], remaining_rolls: int, available: tuple[Category, ...]
    ) -> float:
        """Return the exact optimal value from a canonical hand."""
        terminal_value = cls._terminal_value_cached(sorted_dice, available)
        if remaining_rolls == 0:
            return terminal_value

        reroll_value = max(
            cls._expected_value_for_held_values_static(
                sorted_dice, held_values, remaining_rolls, available
            )
            for held_values in cls._unique_held_value_sets(sorted_dice)
        )
        return max(terminal_value, reroll_value)

    @classmethod
    @lru_cache(maxsize=100_000)
    def _terminal_value_cached(
        cls, sorted_dice: tuple[int, ...], available: tuple[Category, ...]
    ) -> float:
        scores = _SCORE_TABLE[sorted_dice]
        return float(max(scores[_CATEGORY_INDEX[category]] for category in available))

    @classmethod
    @lru_cache(maxsize=100_000)
    def _expected_value_for_held_values_static(
        cls,
        sorted_dice: tuple[int, ...],
        held_values: tuple[int, ...],
        remaining_rolls: int,
        available: tuple[Category, ...],
    ) -> float:
        outcomes = cls._reroll_distribution(held_values)
        return sum(
            probability
            * cls._state_value(outcome, remaining_rolls - 1, available)
            for outcome, probability in outcomes
        )

    def _expected_value_for_held_values(
        self,
        sorted_dice: tuple[int, ...],
        held_values: tuple[int, ...],
        remaining_rolls: int,
        available: tuple[Category, ...],
    ) -> float:
        return self._expected_value_for_held_values_static(
            sorted_dice, held_values, remaining_rolls, available
        )

    @staticmethod
    @lru_cache(maxsize=100_000)
    def _unique_held_value_sets(sorted_dice: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """Return every distinct sub-multiset of up to four held dice."""
        values: set[tuple[int, ...]] = set()
        for count in range(DICE_COUNT):
            for indices in combinations(range(DICE_COUNT), count):
                values.add(tuple(sorted(sorted_dice[index] for index in indices)))
        return tuple(sorted(values, key=lambda item: (len(item), item)))

    @staticmethod
    def _available_categories(state: GameState) -> tuple[Category, ...]:
        used = state.players[state.current_player].used_categories
        return tuple(category for category in ALL_CATEGORIES if category not in used)

    @staticmethod
    @lru_cache(maxsize=1)
    def _meaningful_reroll_actions() -> tuple[Action, ...]:
        return tuple(
            action
            for action in ActionGenerator.reroll_actions()
            if len(action.held_indices) < DICE_COUNT
        )

    @staticmethod
    def _category_key(categories: Iterable[Category]) -> tuple[Category, ...]:
        category_set = set(categories)
        return tuple(category for category in ALL_CATEGORIES if category in category_set)

    @staticmethod
    def _best_category(dice: tuple[int, ...], available: tuple[Category, ...]) -> Category:
        scores = _SCORE_TABLE[dice]
        return max(
            available,
            key=lambda category: (scores[_CATEGORY_INDEX[category]], -_CATEGORY_INDEX[category]),
        )

    @staticmethod
    def _candidate_sort_key(candidate: ActionAlternative) -> tuple[float, int, int]:
        """Prefer score-now on exact ties, then preserve a stable hold order."""
        return (
            candidate.expected_value,
            1 if candidate.action.type is ActionType.SCORE else 0,
            len(candidate.action.held_indices),
        )

    @staticmethod
    def _reasoning(action: Action, expected_value: float) -> str:
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            return f"Record {action.selected_category.display_name}; expected value {expected_value:.2f}."
        held = ", ".join(str(index) for index in action.held_indices) or "none"
        return f"Keep dice at indices [{held}] for expected value {expected_value:.2f}."
