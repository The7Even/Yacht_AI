"""Rule-based Yacht strategy intended as a simple, extensible baseline."""

from collections import Counter
from typing import Protocol

from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionType, DecisionResult


class Strategy(Protocol):
    """A pluggable policy that reads state but never mutates it."""

    def decide(self, state: GameState) -> DecisionResult: ...


class RuleBasedStrategy:
    """A playable heuristic policy, deliberately without probability search."""

    _COMPLETED_CATEGORY_PRIORITY = (
        Category.YACHT,
        Category.FOUR_OF_A_KIND,
        Category.FULL_HOUSE,
        Category.LARGE_STRAIGHT,
        Category.SMALL_STRAIGHT,
    )

    def decide(self, state: GameState) -> DecisionResult:
        """Recommend a reroll or scoring action for the active rolled hand."""
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("RuleBasedStrategy requires a rolled hand.")

        available = tuple(
            category
            for category in ALL_CATEGORIES
            if category not in state.players[state.current_player].used_categories
        )
        if not available:
            raise ValueError("The active player has no categories available.")

        completed = self._completed_category(state.current_dice, available)
        if completed is not None:
            return self._score(completed, f"Completed {completed.display_name} is secured.")

        held = self._hold_indices(state.current_dice)
        if state.roll_count < MAX_ROLLS_PER_TURN and len(held) < len(state.current_dice):
            return DecisionResult(
                Action(ActionType.REROLL, held),
                self._reroll_reason(state.current_dice, held),
            )

        category = self._best_scoring_category(state.current_dice, available)
        return self._score(category, f"No rolls remain; record {category.display_name}.")

    def _completed_category(
        self, dice: tuple[int, ...], available: tuple[Category, ...]
    ) -> Category | None:
        for category in self._COMPLETED_CATEGORY_PRIORITY:
            if category in available and ScoreCalculator.calculate(category, dice) > 0:
                return category
        return None

    @staticmethod
    def _best_scoring_category(dice: tuple[int, ...], available: tuple[Category, ...]) -> Category:
        priority = {category: index for index, category in enumerate(ALL_CATEGORIES)}
        return max(
            available,
            key=lambda category: (ScoreCalculator.calculate(category, dice), -priority[category]),
        )

    @staticmethod
    def _hold_indices(dice: tuple[int, ...]) -> tuple[int, ...]:
        counts = Counter(dice)
        repeated_faces = [face for face, count in counts.items() if count >= 2]
        if repeated_faces:
            target = max(repeated_faces, key=lambda face: (counts[face], face))
            return tuple(index for index, face in enumerate(dice) if face == target)

        straight_faces = RuleBasedStrategy._longest_straight_faces(dice)
        if len(straight_faces) >= 3:
            held: list[int] = []
            seen: set[int] = set()
            for index, face in enumerate(dice):
                if face in straight_faces and face not in seen:
                    held.append(index)
                    seen.add(face)
            return tuple(held)

        highest_index = max(range(len(dice)), key=dice.__getitem__)
        return (highest_index,)

    @staticmethod
    def _longest_straight_faces(dice: tuple[int, ...]) -> tuple[int, ...]:
        faces = sorted(set(dice))
        best: list[int] = []
        current: list[int] = []
        for face in faces:
            if current and face != current[-1] + 1:
                current = []
            current.append(face)
            if len(current) > len(best):
                best = current[:]
        return tuple(best)

    @staticmethod
    def _reroll_reason(dice: tuple[int, ...], held: tuple[int, ...]) -> str:
        kept = [dice[index] for index in held]
        if len(set(kept)) == 1 and len(kept) >= 2:
            return f"Keep {len(kept)} matching {kept[0]}s and reroll the remaining dice."
        if len(kept) >= 3:
            return "Keep the consecutive dice to pursue a straight."
        return f"Keep {kept[0]} and improve the remaining dice."

    @staticmethod
    def _score(category: Category, reasoning: str) -> DecisionResult:
        return DecisionResult(Action(ActionType.SCORE, selected_category=category), reasoning)
