"""Exact, exhaustive expected-value strategy for Yacht rerolls."""

from collections.abc import Iterable
from functools import lru_cache
from itertools import product

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT, MAX_FACE, MIN_FACE, validate_dice
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionAlternative, ActionGenerator, ActionType, DecisionResult


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
        remaining_rolls = MAX_ROLLS_PER_TURN - state.roll_count
        terminal_action = Action(
            ActionType.SCORE, selected_category=self._best_category(dice, available)
        )
        candidates = [
            ActionAlternative(terminal_action, self.evaluate_terminal_state(dice, available))
        ]

        if remaining_rolls > 0:
            used = self._used_key(available)
            for action in self._meaningful_reroll_actions(state.held_indices):
                expected_value = self.expected_value_for_hold(
                    dice, action.held_indices, remaining_rolls, used
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
        """Enumerate every equally likely result of rerolling unheld dice."""
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
        """Return exact enumerated outcomes with their uniform probabilities."""
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
        """Evaluate one reroll action and all optimal later choices exactly."""
        if remaining_rolls <= 0:
            raise ValueError("A reroll requires at least one remaining roll.")
        available = self._category_key(available_categories)
        outcomes = self.enumerate_reroll_outcomes(dice, held_indices)
        return sum(
            self._state_value(tuple(sorted(outcome)), remaining_rolls - 1, available)
            for outcome in outcomes
        ) / len(outcomes)

    def evaluate_terminal_state(
        self, dice: Iterable[int], available_categories: Iterable[Category]
    ) -> float:
        """Return immediate value of the best still-available score category."""
        values = validate_dice(dice)
        available = self._category_key(available_categories)
        if not available:
            raise ValueError("At least one category must be available.")
        return float(max(ScoreCalculator.calculate(category, values) for category in available))

    @lru_cache(maxsize=None)
    def _state_value(
        self, sorted_dice: tuple[int, ...], remaining_rolls: int, available: tuple[Category, ...]
    ) -> float:
        terminal_value = self.evaluate_terminal_state(sorted_dice, available)
        if remaining_rolls == 0:
            return terminal_value

        reroll_value = max(
            self.expected_value_for_hold(sorted_dice, action.held_indices, remaining_rolls, available)
            for action in self._meaningful_reroll_actions()
        )
        return max(terminal_value, reroll_value)

    @staticmethod
    def _available_categories(state: GameState) -> tuple[Category, ...]:
        used = state.players[state.current_player].used_categories
        return tuple(category for category in ALL_CATEGORIES if category not in used)

    @staticmethod
    def _meaningful_reroll_actions(required_held_indices: frozenset[int] = frozenset()) -> tuple[Action, ...]:
        return tuple(
            action
            for action in ActionGenerator.reroll_actions(required_held_indices)
            if len(action.held_indices) < DICE_COUNT
        )

    @staticmethod
    def _category_key(categories: Iterable[Category]) -> tuple[Category, ...]:
        category_set = set(categories)
        return tuple(category for category in ALL_CATEGORIES if category in category_set)

    @staticmethod
    def _used_key(available: tuple[Category, ...]) -> tuple[Category, ...]:
        return available

    @staticmethod
    def _best_category(dice: tuple[int, ...], available: tuple[Category, ...]) -> Category:
        category_order = {category: index for index, category in enumerate(ALL_CATEGORIES)}
        return max(
            available,
            key=lambda category: (ScoreCalculator.calculate(category, dice), -category_order[category]),
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
