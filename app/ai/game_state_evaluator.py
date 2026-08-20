"""Lightweight, opponent-aware evaluation of a Yacht game position."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import MAX_FACE, MIN_FACE
from app.core.game_state import GameState, PlayerId, PlayerState
from app.core.scoring import ScoreCalculator

from .upper_expected_value import UpperCategoryExpectedValueEvaluator


@dataclass(frozen=True)
class GameStateEvaluation:
    """Explainable components of one participant's current game position."""

    score_difference: float
    remaining_turns: int
    my_future_score: float
    opponent_future_score: float
    position_value: float


class GameStateEvaluator:
    """Evaluates score context without predicting either player's exact actions."""

    def __init__(self, upper_evaluator: UpperCategoryExpectedValueEvaluator | None = None) -> None:
        self._upper_evaluator = upper_evaluator or UpperCategoryExpectedValueEvaluator()

    def evaluate(self, state: GameState, perspective: PlayerId | None = None) -> GameStateEvaluation:
        """Return a position value from *perspective* without modifying the state."""
        player_id = perspective or state.current_player
        opponent_id = PlayerId.AI if player_id is PlayerId.PLAYER else PlayerId.PLAYER
        mine = state.players[player_id]
        opponent = state.players[opponent_id]
        remaining_turns = max(self.remaining_categories(mine), self.remaining_categories(opponent))
        my_future = self.future_score_value(mine)
        opponent_future = self.future_score_value(opponent)
        score_difference = float(mine.total_score - opponent.total_score)
        projected_difference = score_difference + my_future - opponent_future

        # A lead is more decisive as the number of unscored categories shrinks.
        stage_weight = 1.0 / (1.0 + remaining_turns)
        return GameStateEvaluation(
            score_difference=score_difference,
            remaining_turns=remaining_turns,
            my_future_score=my_future,
            opponent_future_score=opponent_future,
            position_value=projected_difference * stage_weight,
        )

    @staticmethod
    def remaining_categories(player: PlayerState) -> int:
        return len(ALL_CATEGORIES) - len(player.used_categories)

    def future_score_value(self, player: PlayerState) -> float:
        """Conservative future score estimate from each still-unused category."""
        remaining = tuple(category for category in ALL_CATEGORIES if category not in player.used_categories)
        upper = tuple(category for category in remaining if category.is_upper)
        lower = tuple(category for category in remaining if not category.is_upper)
        return self._upper_evaluator.expected_remaining_upper_score(upper) + sum(
            self._lower_category_baseline(category) for category in lower
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _lower_category_baseline(category: Category) -> float:
        """Exact one-roll baseline for a lower category, without strategy prediction."""
        if category.is_upper:
            raise ValueError("Lower baseline requested for an upper category.")
        total = sum(
            ScoreCalculator.calculate(category, dice)
            for dice in product(range(MIN_FACE, MAX_FACE + 1), repeat=5)
        )
        return total / (MAX_FACE**5)
