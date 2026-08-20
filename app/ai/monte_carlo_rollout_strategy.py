"""Lightweight Yacht-aware policy for Monte Carlo game rollouts."""

from collections import Counter

from app.core.categories import ALL_CATEGORIES, Category
from app.core.game_engine import MAX_ROLLS_PER_TURN
from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionType, DecisionResult
from .strategy import Strategy


class MonteCarloRolloutStrategy:
    """Fast deterministic policy intended for Monte Carlo continuation games.

    This policy is deliberately cheaper than EV search, but it still respects
    Yacht-specific incentives: completed hands, upper-section bonus progress,
    and the value of banking a strong upper score instead of rerolling forever.
    """

    _COMPLETED_PRIORITY = (
        Category.YACHT,
        Category.LARGE_STRAIGHT,
        Category.FOUR_OF_A_KIND,
        Category.FULL_HOUSE,
        Category.SMALL_STRAIGHT,
    )

    def decide(self, state: GameState) -> DecisionResult:
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("MonteCarloRolloutStrategy requires a rolled hand.")

        player = state.players[state.current_player]
        available = tuple(
            category for category in ALL_CATEGORIES if category not in player.used_categories
        )
        if not available:
            raise ValueError("The active player has no categories available.")

        dice = state.current_dice
        scores = {category: ScoreCalculator.calculate(category, dice) for category in available}

        # Always bank a completed premium hand. This prevents rollout from
        # throwing away deterministic 30/40/50-point opportunities.
        completed = [
            category
            for category in self._COMPLETED_PRIORITY
            if category in available and scores[category] > 0
        ]
        if completed:
            category = max(completed, key=lambda item: (scores[item], -ALL_CATEGORIES.index(item)))
            return DecisionResult(
                Action(ActionType.SCORE, selected_category=category),
                f"Rollout: secure completed {category.display_name}.",
            )

        best_category = self._best_scoring_category(state, scores)
        best_score = scores[best_category]

        if state.roll_count < MAX_ROLLS_PER_TURN:
            held = self._hold_indices_for_state(state, best_category)
            reroll_value = self._reroll_value(state, held, best_category, best_score)

            # Only reroll when there is a meaningful improvement path. A strong
            # upper score is often more valuable than chasing a low-probability
            # Yacht/straight from a mediocre hand.
            if held and len(held) < len(dice) and reroll_value > best_score:
                return DecisionResult(
                    Action(ActionType.REROLL, held),
                    "Rollout: reroll because the retained pattern has positive upside.",
                )

        return DecisionResult(
            Action(ActionType.SCORE, selected_category=best_category),
            f"Rollout: bank {best_category.display_name} for {best_score}.",
        )

    @staticmethod
    def _best_scoring_category(state: GameState, scores: dict[Category, int]) -> Category:
        player = state.players[state.current_player]
        upper_total = player.upper_total

        def key(category: Category) -> tuple[float, int, int]:
            score = scores[category]
            bonus_value = 0
            if category.is_upper and not player.bonus_awarded:
                projected = upper_total + score
                if projected >= 63:
                    bonus_value = 35
                elif projected >= 50:
                    bonus_value = 8
            return (score + bonus_value, score, -ALL_CATEGORIES.index(category))

        return max(scores, key=key)

    @classmethod
    def _hold_indices_for_state(
        cls, state: GameState, target_category: Category
    ) -> tuple[int, ...]:
        dice = state.current_dice
        assert dice is not None

        # For an upper category, preserving target faces is the most direct
        # route to improving that category and the upper bonus.
        if target_category.is_upper:
            target = target_category.upper_face
            assert target is not None
            indices = tuple(i for i, face in enumerate(dice) if face == target)
            if indices:
                return indices

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

    @staticmethod
    def _reroll_value(
        state: GameState,
        held: tuple[int, ...],
        category: Category,
        current_score: int,
    ) -> float:
        """Cheap optimistic estimate of whether a reroll is worth taking."""
        dice = state.current_dice
        assert dice is not None
        missing = len(dice) - len(held)
        if missing <= 0:
            return float(current_score)

        held_values = [dice[index] for index in held]
        counts = Counter(held_values)
        max_count = max(counts.values(), default=0)

        if category is Category.YACHT:
            return 50.0 * (max_count / 5.0)
        if category is Category.FOUR_OF_A_KIND:
            return sum(held_values) + 6.0 * missing
        if category.is_upper:
            target = category.upper_face
            assert target is not None
            target_count = counts.get(target, 0)
            return float((target_count + missing * (1 / 6)) * target)
        if category in (Category.SMALL_STRAIGHT, Category.LARGE_STRAIGHT):
            run = MonteCarloRolloutStrategy._best_straight_length(held_values)
            target = 15 if category is Category.SMALL_STRAIGHT else 30
            return float(target * min(1.0, (run + missing) / (4 if target == 15 else 5)))
        if category is Category.FULL_HOUSE:
            return float(current_score + max_count * missing)
        return float(sum(held_values) + 3.5 * missing)

    @staticmethod
    def _best_straight_length(values: list[int]) -> int:
        faces = sorted(set(values))
        if not faces:
            return 0
        best = current = 1
        for left, right in zip(faces, faces[1:]):
            if right == left + 1:
                current += 1
                best = max(best, current)
            else:
                current = 1
        return best
