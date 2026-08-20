"""Dice validation and random rolling utilities."""

from collections.abc import Iterable, Sequence
import random


DICE_COUNT = 5
MIN_FACE = 1
MAX_FACE = 6


def validate_dice(dice: Iterable[int], *, expected_count: int = DICE_COUNT) -> tuple[int, ...]:
    """Validate dice faces and return an immutable representation.

    A full scoring hand must contain exactly five integers from 1 through 6.
    """
    values = tuple(dice)
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} dice, received {len(values)}.")
    if any(type(value) is not int or not MIN_FACE <= value <= MAX_FACE for value in values):
        raise ValueError("Each die must be an integer from 1 through 6.")
    return values


class DiceRoller:
    """Produces unbiased six-sided dice rolls using an injectable random source."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def roll(self, count: int = DICE_COUNT) -> tuple[int, ...]:
        """Roll *count* dice."""
        if count < 0:
            raise ValueError("Dice count cannot be negative.")
        return tuple(self._rng.randint(MIN_FACE, MAX_FACE) for _ in range(count))

    def reroll(self, dice: Sequence[int], held_indices: Iterable[int]) -> tuple[int, ...]:
        """Return a hand with only the dice at *held_indices* preserved."""
        values = validate_dice(dice)
        held = set(held_indices)
        if any(type(index) is not int or not 0 <= index < DICE_COUNT for index in held):
            raise ValueError("Held indices must be integers from 0 through 4.")
        return tuple(value if index in held else self.roll(1)[0] for index, value in enumerate(values))
