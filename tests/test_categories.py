from app.core.categories import ALL_CATEGORIES, LOWER_CATEGORIES, UPPER_CATEGORIES, Category


def test_all_twelve_categories_are_defined_once() -> None:
    assert ALL_CATEGORIES == tuple(Category)
    assert len(ALL_CATEGORIES) == 12
    assert set(ALL_CATEGORIES) == {
        Category.ONES,
        Category.TWOS,
        Category.THREES,
        Category.FOURS,
        Category.FIVES,
        Category.SIXES,
        Category.CHOICE,
        Category.FOUR_OF_A_KIND,
        Category.FULL_HOUSE,
        Category.SMALL_STRAIGHT,
        Category.LARGE_STRAIGHT,
        Category.YACHT,
    }


def test_upper_and_lower_categories_are_partitioned_correctly() -> None:
    assert UPPER_CATEGORIES == (
        Category.ONES,
        Category.TWOS,
        Category.THREES,
        Category.FOURS,
        Category.FIVES,
        Category.SIXES,
    )
    assert len(LOWER_CATEGORIES) == 6
    assert set(UPPER_CATEGORIES).isdisjoint(LOWER_CATEGORIES)
    assert set(UPPER_CATEGORIES) | set(LOWER_CATEGORIES) == set(ALL_CATEGORIES)
