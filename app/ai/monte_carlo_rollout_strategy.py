"""Lightweight policy for Monte Carlo game rollouts.

This policy deliberately avoids EV enumeration. It uses cheap local scoring and
simple hold heuristics so thousands of simulated games remain practical.
"""

from collections import Counter

from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionType, DecisionResult
from .strategy import Strategy


class MonteCarloRolloutStrategy:
    """Fast deterministic policy intended only for Monte Carlo continuation."""

    _COMPLETED_PRIORITY = (
        Category.YACHT,
        Category.FOUR_OF_A_KIND,
        Category.FULL_HOUSE,
        Category.LARGE_STRAIGHT,
        Category.SMALL_STRAIGHT,
    )

    def decide(self, state: GameState) -> DecisionResult:
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("MonteCarloRolloutStrategy requires a rolled hand.")

        available = tuple(
            category
            for category in ALL_CATEGORIES
            if category not in state.players[state.current_player].used_categories
        )
        if not available:
            raise ValueError("The active player has no categories available.")

        dice = state.current_dice
        for category in self._COMPLETED_PRIORITY:
            if category in available and ScoreCalculator.calculate(category, dice) > 0:
                return DecisionResult(
                    Action(ActionType.SCORE, selected_category=category),
                    f"Rollout: secure completed {category.display_name}.",
                )

        best_category = max(
            available,
            key=lambda category: (
                ScoreCalculator.calculate(category, dice),
                -ALL_CATEGORIES.index(category),
            ),
        )
        best_score = ScoreCalculator.calculate(best_category, dice)

        if state.roll_count < MAX_ROLLS_PER_TURN:
            held = self._hold_indices(dice)
            if len(held) < len(dice):
                return DecisionResult(
                    Action(ActionType.REROLL, held),
                    "Rollout: keep the strongest local pattern and reroll the rest.",
                )

        return DecisionResult(
            Action(ActionType.SCORE, selected_category=best_category),
            f"Rollout: record {best_category.display_name} for {best_score}.",
        )

    @staticmethod
    def _hold_indices(dice: tuple[int, ...]) -> tuple[int, ...]:
        counts = Counter(dice)
        repeated = [face for face, count in counts.items() if count >= 2]
        if repeated:
            target = max(repeated, key=lambda face: (counts[face], face))
            return tuple(i for i, face in enumerate(dice) if face == target)

        unique = sorted(set(dice))
        best_run: list[int] = []
        current: list[int] = []
        for face in unique:
            if current and face != current[-1] + 1:
                current = []
            current.append(face)
            if len(current) > len(best_run):
                best_run = current[:]

        if len(best_run) >= 3:
            return tuple(i for i, face in enumerate(dice) if face in best_run)

        highest = max(dice)
        return (dice.index(highest),)
