import random

import pytest

from app.core.dice import DICE_COUNT, MAX_FACE, MIN_FACE, DiceRoller, validate_dice


def test_rolls_five_valid_dice_by_default() -> None:
    dice = DiceRoller(random.Random(7)).roll()

    assert len(dice) == DICE_COUNT
    assert all(MIN_FACE <= value <= MAX_FACE for value in dice)


def test_roll_can_generate_a_requested_number_of_dice() -> None:
    assert len(DiceRoller(random.Random(7)).roll(3)) == 3


def test_reroll_preserves_only_held_dice() -> None:
    original = (1, 2, 3, 4, 5)
    rerolled = DiceRoller(random.Random(7)).reroll(original, held_indices=(1, 3))

    assert rerolled[1] == 2
    assert rerolled[3] == 4
    assert all(MIN_FACE <= value <= MAX_FACE for value in rerolled)


@pytest.mark.parametrize("dice", [(1, 2, 3, 4), (1, 2, 3, 4, 7), (True, 2, 3, 4, 5)])
def test_validate_dice_rejects_invalid_hands(dice: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        validate_dice(dice)


def test_reroll_rejects_out_of_range_held_index() -> None:
    with pytest.raises(ValueError):
        DiceRoller().reroll((1, 2, 3, 4, 5), held_indices=(5,))
