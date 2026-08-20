"""Structured AI actions and basic legal-action generation."""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT
from app.core.game_state import GameState


class ActionType(str, Enum):
    """Kinds of decisions an AI may return."""

    HOLD = "hold"
    REROLL = "reroll"
    SCORE = "score"


@dataclass(frozen=True)
class Action:
    """A proposed action; applying it remains the GameEngine's responsibility."""

    type: ActionType
    held_indices: tuple[int, ...] = ()
    selected_category: Category | None = None

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.held_indices))) != self.held_indices:
            raise ValueError("Held indices must be unique and sorted.")
        if any(type(index) is not int or not 0 <= index < DICE_COUNT for index in self.held_indices):
            raise ValueError("Held indices must be integers from 0 through 4.")
        if self.type is ActionType.SCORE and self.selected_category is None:
            raise ValueError("A score action requires a category.")
        if self.type is not ActionType.SCORE and self.selected_category is not None:
            raise ValueError("Only a score action may select a category.")


@dataclass(frozen=True)
class DecisionResult:
    """A UI-safe explanation of an AI's recommended action."""

    action: Action
    reasoning: str
    expected_value: float | None = None
    alternatives: tuple["ActionAlternative", ...] = ()

    @property
    def held_indices(self) -> tuple[int, ...]:
        return self.action.held_indices

    @property
    def selected_category(self) -> Category | None:
        return self.action.selected_category

    @property
    def win_probability(self) -> float | None:
        """Estimated win probability when alternatives contain probability scores."""
        return self.alternatives[0].expected_value if self.alternatives else None


@dataclass(frozen=True)
class ActionAlternative:
    """A candidate action and its evaluated value.

    The existing ``expected_value`` field is retained for compatibility with
    the EV strategies; for Monte Carlo alternatives it represents probability.
    ``win_probability`` is the explicit alias used by win-probability clients.
    """

    action: Action
    expected_value: float

    @property
    def win_probability(self) -> float:
        """Return the value as a win probability for Monte Carlo evaluations."""
        return self.expected_value


class ActionGenerator:
    """Generates basic actions without applying them to a game state."""

    @staticmethod
    def hold_actions() -> tuple[Action, ...]:
        """Return every possible set of dice to retain (2^5 combinations)."""
        return tuple(
            Action(ActionType.HOLD, indices)
            for count in range(DICE_COUNT + 1)
            for indices in combinations(range(DICE_COUNT), count)
        )

    @staticmethod
    def reroll_actions(required_held_indices: frozenset[int] = frozenset()) -> tuple[Action, ...]:
        """Return every possible reroll choice, expressed by its held dice."""
        return tuple(
            Action(ActionType.REROLL, action.held_indices)
            for action in ActionGenerator.hold_actions()
            if required_held_indices.issubset(action.held_indices)
        )

    @staticmethod
    def score_actions(state: GameState) -> tuple[Action, ...]:
        """Return score actions for categories unused by the current participant."""
        used = state.players[state.current_player].used_categories
        return tuple(
            Action(ActionType.SCORE, selected_category=category)
            for category in ALL_CATEGORIES
            if category not in used
        )
