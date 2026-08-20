"""Data-only representation of an in-progress Yacht game."""

from dataclasses import dataclass, field
from enum import Enum

from .categories import Category, UPPER_BONUS_SCORE, UPPER_BONUS_THRESHOLD


class PlayerId(str, Enum):
    """Built-in participant identifiers for a two-player game."""

    PLAYER = "player"
    AI = "ai"


@dataclass
class PlayerState:
    """Scores recorded by one participant."""

    category_scores: dict[Category, int] = field(default_factory=dict)

    @property
    def used_categories(self) -> frozenset[Category]:
        """Categories already committed by this participant."""
        return frozenset(self.category_scores)

    @property
    def upper_total(self) -> int:
        """Recorded upper-section score, excluding the bonus."""
        return sum(score for category, score in self.category_scores.items() if category.is_upper)

    @property
    def has_upper_bonus(self) -> bool:
        """Whether the upper-section threshold has been met."""
        return self.upper_total >= UPPER_BONUS_THRESHOLD

    @property
    def total_score(self) -> int:
        """Final-to-date score including any earned upper bonus."""
        return sum(self.category_scores.values()) + (
            UPPER_BONUS_SCORE if self.has_upper_bonus else 0
        )


def _new_players() -> dict[PlayerId, PlayerState]:
    return {player: PlayerState() for player in PlayerId}


@dataclass
class GameState:
    """Mutable state container with no dice, scoring, or turn-transition rules."""

    turn: int = 0
    current_player: PlayerId = PlayerId.PLAYER
    players: dict[PlayerId, PlayerState] = field(default_factory=_new_players)
    current_dice: tuple[int, ...] | None = None
    held_indices: frozenset[int] = field(default_factory=frozenset)
    roll_count: int = 0
    game_started: bool = False
    turn_scored: bool = False
    game_over: bool = False

    @property
    def held_dice(self) -> tuple[int, ...]:
        """Values currently held, ordered by their die positions."""
        if self.current_dice is None:
            return ()
        return tuple(self.current_dice[index] for index in sorted(self.held_indices))

    @property
    def player_score(self) -> int:
        return self.players[PlayerId.PLAYER].total_score

    @property
    def ai_score(self) -> int:
        return self.players[PlayerId.AI].total_score

    @property
    def player_category_scores(self) -> dict[Category, int]:
        return self.players[PlayerId.PLAYER].category_scores

    @property
    def ai_category_scores(self) -> dict[Category, int]:
        return self.players[PlayerId.AI].category_scores

    @property
    def player_used_categories(self) -> frozenset[Category]:
        return self.players[PlayerId.PLAYER].used_categories

    @property
    def ai_used_categories(self) -> frozenset[Category]:
        return self.players[PlayerId.AI].used_categories

    @property
    def player_upper_total(self) -> int:
        return self.players[PlayerId.PLAYER].upper_total

    @property
    def ai_upper_total(self) -> int:
        return self.players[PlayerId.AI].upper_total

    @property
    def player_has_bonus(self) -> bool:
        return self.players[PlayerId.PLAYER].has_upper_bonus

    @property
    def ai_has_bonus(self) -> bool:
        return self.players[PlayerId.AI].has_upper_bonus
