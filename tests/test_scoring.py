import pytest

from app.core.categories import Category
from app.core.scoring import ScoreCalculator


@pytest.mark.parametrize(
    ("category", "dice", "expected"),
    [
        (Category.ONES, (1, 1, 3, 4, 6), 2),
        (Category.TWOS, (2, 2, 2, 4, 6), 6),
        (Category.THREES, (3, 3, 3, 3, 6), 12),
        (Category.FOURS, (1, 4, 4, 5, 6), 8),
        (Category.FIVES, (2, 5, 5, 5, 6), 15),
        (Category.SIXES, (1, 3, 6, 6, 6), 18),
    ],
)
def test_upper_section_scores(category: Category, dice: tuple[int, ...], expected: int) -> None:
    assert ScoreCalculator.calculate(category, dice) == expected


def test_choice_scores_sum_of_all_dice() -> None:
    assert ScoreCalculator.calculate(Category.CHOICE, (1, 2, 3, 4, 6)) == 16


@pytest.mark.parametrize(
    ("dice", "expected"),
    [((4, 4, 4, 4, 6), 22), ((4, 4, 4, 5, 6), 0), ((5, 5, 5, 5, 5), 25)],
)
def test_four_of_a_kind(dice: tuple[int, ...], expected: int) -> None:
    assert ScoreCalculator.calculate(Category.FOUR_OF_A_KIND, dice) == expected


@pytest.mark.parametrize(
    ("dice", "expected"),
    [((2, 2, 2, 5, 5), 16), ((2, 2, 2, 2, 5), 0), ((6, 6, 6, 6, 6), 30)],
)
def test_full_house_includes_yacht(dice: tuple[int, ...], expected: int) -> None:
    assert ScoreCalculator.calculate(Category.FULL_HOUSE, dice) == expected


@pytest.mark.parametrize(
    ("dice", "expected"),
    [((1, 2, 3, 4, 6), 15), ((1, 2, 2, 4, 6), 0), ((3, 4, 5, 6, 6), 15)],
)
def test_small_straight(dice: tuple[int, ...], expected: int) -> None:
    assert ScoreCalculator.calculate(Category.SMALL_STRAIGHT, dice) == expected


@pytest.mark.parametrize(
    ("dice", "expected"),
    [((1, 2, 3, 4, 5), 30), ((2, 3, 4, 5, 6), 30), ((1, 2, 3, 4, 6), 0)],
)
def test_large_straight(dice: tuple[int, ...], expected: int) -> None:
    assert ScoreCalculator.calculate(Category.LARGE_STRAIGHT, dice) == expected


@pytest.mark.parametrize(
    ("dice", "expected"),
    [((3, 3, 3, 3, 3), 50), ((3, 3, 3, 3, 4), 0)],
)
def test_yacht(dice: tuple[int, ...], expected: int) -> None:
    assert ScoreCalculator.calculate(Category.YACHT, dice) == expected


def test_invalid_hand_is_rejected() -> None:
    with pytest.raises(ValueError):
        ScoreCalculator.calculate(Category.CHOICE, (1, 2, 3, 4))


@pytest.mark.parametrize("dice", [(0, 2, 3, 4, 5), (1, 2, 3, 4, 7)])
def test_invalid_die_face_is_rejected(dice: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        ScoreCalculator.calculate(Category.CHOICE, dice)


def test_calculate_all_returns_a_score_for_every_category() -> None:
    scores = ScoreCalculator.calculate_all((5, 5, 5, 5, 5))

    assert set(scores) == set(Category)
    assert scores[Category.FOUR_OF_A_KIND] == 25
    assert scores[Category.FULL_HOUSE] == 25
    assert scores[Category.YACHT] == 50
