from app.ai.monte_carlo_rollout_strategy import MonteCarloRolloutStrategy
from app.core.categories import Category
from app.core.game_engine import GameEngine


def state_with_dice(dice: tuple[int, ...], roll_count: int = 1):
    engine = GameEngine()
    engine.start_game()
    engine.state.current_dice = dice
    engine.state.roll_count = roll_count
    return engine.state


def test_rollout_scores_completed_yacht() -> None:
    result = MonteCarloRolloutStrategy().decide(state_with_dice((5, 5, 5, 5, 5)))
    assert result.action.selected_category is Category.YACHT


def test_rollout_scores_completed_large_straight() -> None:
    result = MonteCarloRolloutStrategy().decide(state_with_dice((2, 3, 4, 5, 6)))
    assert result.action.selected_category is Category.LARGE_STRAIGHT


def test_rollout_considers_upper_bonus() -> None:
    state = state_with_dice((6, 6, 6, 2, 3), roll_count=3)
    player = state.players[state.current_player]
    player.category_scores = {
        Category.ONES: 5,
        Category.TWOS: 10,
        Category.THREES: 15,
        Category.FOURS: 8,
        Category.FIVES: 10,
    }
    # Upper total is 48, so recording 18 Sixes points reaches 66 and earns
    # the 35-point bonus. On the final roll the rollout must bank Sixes.
    result = MonteCarloRolloutStrategy().decide(state)
    assert result.action.selected_category is Category.SIXES


def test_rollout_scores_on_third_roll() -> None:
    result = MonteCarloRolloutStrategy().decide(state_with_dice((6, 6, 2, 3, 4), roll_count=3))
    assert result.action.type.value == "score"
