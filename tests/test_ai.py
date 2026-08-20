import pytest

from app.ai.action_generator import Action, ActionGenerator, ActionType, DecisionResult
from app.ai.ai_player import AIPlayer
from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT, DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import GameState, PlayerId


def state_with_hand(
    dice: tuple[int, ...], *, roll_count: int = 1, used: set[Category] | None = None
) -> GameState:
    state = GameState(game_started=True, turn=1, current_dice=dice, roll_count=roll_count)
    state.players[PlayerId.PLAYER].category_scores.update({category: 0 for category in used or set()})
    return state


@pytest.mark.parametrize(
    ("dice", "category"),
    [
        ((5, 5, 5, 5, 5), Category.YACHT),
        ((4, 4, 4, 4, 6), Category.FOUR_OF_A_KIND),
        ((4, 4, 4, 5, 5), Category.FULL_HOUSE),
        ((1, 2, 3, 4, 5), Category.LARGE_STRAIGHT),
        ((1, 2, 3, 4, 6), Category.SMALL_STRAIGHT),
    ],
)
def test_rule_based_ai_recognizes_completed_categories(
    dice: tuple[int, ...], category: Category
) -> None:
    decision = AIPlayer().decide(state_with_hand(dice))

    assert decision.action.type is ActionType.SCORE
    assert decision.selected_category is category


def test_ai_never_selects_a_used_category() -> None:
    decision = AIPlayer().decide(
        state_with_hand((5, 5, 5, 5, 5), used={Category.YACHT})
    )

    assert decision.selected_category is Category.FOUR_OF_A_KIND
    assert decision.selected_category is not Category.YACHT


def test_ai_holds_repeated_high_dice_for_reroll() -> None:
    decision = AIPlayer().decide(state_with_hand((5, 5, 5, 2, 6)))

    assert decision.action.type is ActionType.REROLL
    assert decision.held_indices == (0, 1, 2)
    assert "matching 5s" in decision.reasoning


def test_action_and_decision_result_validate_their_data() -> None:
    decision = DecisionResult(
        Action(ActionType.REROLL, held_indices=(0, 2)), "Keep matching dice."
    )
    assert decision.held_indices == (0, 2)
    assert decision.selected_category is None

    with pytest.raises(ValueError):
        Action(ActionType.SCORE)
    with pytest.raises(ValueError):
        Action(ActionType.REROLL, held_indices=(2, 1))


def test_action_generator_creates_all_hold_choices_and_only_unused_scores() -> None:
    state = state_with_hand((1, 2, 3, 4, 5), used={Category.CHOICE})

    assert len(ActionGenerator.hold_actions()) == 2**DICE_COUNT
    assert len(ActionGenerator.reroll_actions()) == 2**DICE_COUNT
    assert {action.selected_category for action in ActionGenerator.score_actions(state)} == set(
        ALL_CATEGORIES
    ) - {Category.CHOICE}


class ScriptedDiceRoller(DiceRoller):
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = iter(values)

    def roll(self, count: int = DICE_COUNT) -> tuple[int, ...]:
        return tuple(next(self._values) for _ in range(count))


def test_ai_decision_can_be_applied_through_game_engine() -> None:
    engine = GameEngine(ScriptedDiceRoller((5, 5, 5, 2, 6, 5, 1)))
    engine.start_game()
    engine.roll_dice()
    decision = AIPlayer().decide(engine.state)

    assert decision.action.type is ActionType.REROLL
    for index in decision.held_indices:
        engine.hold_dice(index)
    engine.roll_dice()
    decision = AIPlayer().decide(engine.state)
    score = engine.score_category(decision.selected_category)  # type: ignore[arg-type]

    assert decision.selected_category is Category.FOUR_OF_A_KIND
    assert score == 21
    assert engine.state.turn_scored is True
