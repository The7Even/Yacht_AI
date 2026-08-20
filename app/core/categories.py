"""Score categories used by the Yacht rules engine."""

from enum import Enum


class Category(str, Enum):
    """A score category that may be used once per game."""

    ONES = "ones"
    TWOS = "twos"
    THREES = "threes"
    FOURS = "fours"
    FIVES = "fives"
    SIXES = "sixes"
    CHOICE = "choice"
    FOUR_OF_A_KIND = "four_of_a_kind"
    FULL_HOUSE = "full_house"
    SMALL_STRAIGHT = "small_straight"
    LARGE_STRAIGHT = "large_straight"
    YACHT = "yacht"

    @property
    def display_name(self) -> str:
        """Human-readable Korean-independent label for UI and logs."""
        return {
            Category.ONES: "Ones",
            Category.TWOS: "Twos",
            Category.THREES: "Threes",
            Category.FOURS: "Fours",
            Category.FIVES: "Fives",
            Category.SIXES: "Sixes",
            Category.CHOICE: "Choice",
            Category.FOUR_OF_A_KIND: "4 of a Kind",
            Category.FULL_HOUSE: "Full House",
            Category.SMALL_STRAIGHT: "Small Straight",
            Category.LARGE_STRAIGHT: "Large Straight",
            Category.YACHT: "Yacht",
        }[self]

    @property
    def upper_face(self) -> int | None:
        """Return the die face scored by an upper category, if applicable."""
        return {
            Category.ONES: 1,
            Category.TWOS: 2,
            Category.THREES: 3,
            Category.FOURS: 4,
            Category.FIVES: 5,
            Category.SIXES: 6,
        }.get(self)

    @property
    def is_upper(self) -> bool:
        """Whether this category belongs to the upper section."""
        return self.upper_face is not None


UPPER_CATEGORIES: tuple[Category, ...] = tuple(
    category for category in Category if category.is_upper
)
LOWER_CATEGORIES: tuple[Category, ...] = tuple(
    category for category in Category if not category.is_upper
)
ALL_CATEGORIES: tuple[Category, ...] = tuple(Category)
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_SCORE = 35
