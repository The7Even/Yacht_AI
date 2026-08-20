"""Monte Carlo win-probability strategy for Yacht."""

from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionAlternative, ActionGenerator, ActionType, DecisionResult
from .monte_carlo_evaluator import MonteCarloWinProbabilityEvaluator
from .monte_carlo_rollout_strategy import MonteCarloRolloutStrategy
from .strategy import RuleBasedStrategy, Strategy


class WinProbabilityStrategy:
    """Choose the legal action with the highest estimated chance of winning."""

    def __init__(
        self,
        *,
        evaluator: MonteCarloWinProbabilityEvaluator | None = None,
        simulation_count: int = 300,
        max_candidates: int | None = 8,
        continuation_strategy: Strategy | None = None,
        opponent_strategy: Strategy | None = None,
    ) -> None:
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")

        # Monte Carlo rollouts need a very cheap continuation policy. The
        # rollout strategy evaluates the same Yacht rules without recursively
        # performing EV enumeration, which prevents nested search explosion.
        continuation = continuation_strategy or MonteCarloRolloutStrategy()
        opponent = opponent_strategy or RuleBasedStrategy()
        self._evaluator = evaluator or MonteCarloWinProbabilityEvaluator(
            player_strategy=continuation,
            opponent_strategy=opponent,
        )
        self._simulation_count = simulation_count
        self._max_candidates = max_candidates

    def decide(self, state: GameState) -> DecisionResult:
        """Return the legal candidate with the highest estimated win probability."""
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("WinProbabilityStrategy requires a rolled hand.")

        actions = self._candidate_actions(state, self._max_candidates)
        probabilities = self._evaluator.estimate_actions_win_probability(
            state, actions, self._simulation_count
        )
        candidates = tuple(ActionAlternative(action, probabilities[action]) for action in actions)
        ordered = tuple(
            sorted(candidates, key=lambda candidate: self._sort_key(state, candidate), reverse=True)
        )
        best = ordered[0]
        return DecisionResult(
            action=best.action,
            expected_value=None,
            alternatives=ordered[:3],
            reasoning=self._reasoning(best.action, best.expected_value),
        )

    @staticmethod
    def _candidate_actions(
        state: GameState, max_candidates: int | None = None
    ) -> tuple[Action, ...]:
        score_actions = list(ActionGenerator.score_actions(state))
        reroll_actions: list[Action] = []
        if state.roll_count < 3:
            reroll_actions = [
                action
                for action in ActionGenerator.reroll_actions(state.held_indices)
                if len(action.held_indices) < 5
            ]

        actions = score_actions + reroll_actions
        if max_candidates is None or len(actions) <= max_candidates:
            return tuple(actions)

        if not reroll_actions:
            ranked_scores = sorted(
                score_actions,
                key=lambda action: WinProbabilityStrategy._candidate_priority(state, action),
                reverse=True,
            )
            return tuple(ranked_scores[:max_candidates])

        score_slots = max(1, max_candidates // 2)
        reroll_slots = max_candidates - score_slots

        ranked_scores = sorted(
            score_actions,
            key=lambda action: WinProbabilityStrategy._candidate_priority(state, action),
            reverse=True,
        )[:score_slots]
        ranked_rerolls = sorted(
            reroll_actions,
            key=lambda action: WinProbabilityStrategy._candidate_priority(state, action),
            reverse=True,
        )[:reroll_slots]
        return tuple(ranked_scores + ranked_rerolls)

    @staticmethod
    def _candidate_priority(
        state: GameState, action: Action
    ) -> tuple[float, float, float, int, tuple[int, ...]]:
        """Cheap pre-ranking used only when candidate limiting is enabled."""
        assert state.current_dice is not None
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            score = float(ScoreCalculator.calculate(action.selected_category, state.current_dice))
            return (score, score, 0.0, 1, tuple(-index for index in action.held_indices))

        held_values = [state.current_dice[index] for index in action.held_indices]
        counts: dict[int, int] = {}
        for value in held_values:
            counts[value] = counts.get(value, 0) + 1
        duplicate_strength = max(
            (value * count * count for value, count in counts.items()), default=0
        )
        straight_length = float(WinProbabilityStrategy._best_straight_length(held_values))
        return (
            duplicate_strength,
            straight_length,
            float(sum(held_values)),
            0,
            tuple(-index for index in action.held_indices),
        )

    @staticmethod
    def _best_straight_length(values: list[int]) -> int:
        if not values:
            return 0
        faces = sorted(set(values))
        best = current = 1
        for left, right in zip(faces, faces[1:]):
            if right == left + 1:
                current += 1
                best = max(best, current)
            else:
                current = 1
        return best

    @staticmethod
    def _sort_key(
        state: GameState, candidate: ActionAlternative
    ) -> tuple[float, float, int, tuple[int, ...], str]:
        action = candidate.action
        immediate_score = 0.0
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            assert state.current_dice is not None
            immediate_score = float(
                ScoreCalculator.calculate(action.selected_category, state.current_dice)
            )
        return (
            candidate.expected_value,
            immediate_score,
            1 if action.type is ActionType.SCORE else 0,
            tuple(-index for index in action.held_indices),
            action.selected_category.value if action.selected_category is not None else "",
        )

    @staticmethod
    def _reasoning(action: Action, probability: float) -> str:
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            return (
                f"Record {action.selected_category.display_name}; "
                f"estimated win probability {probability:.1%}."
            )
        held = ", ".join(str(index) for index in action.held_indices) or "none"
        return f"Keep dice at indices [{held}]; estimated win probability {probability:.1%}."
