"""Monte Carlo win-probability strategy for Yacht."""

from app.core.game_state import GameState
from app.core.scoring import ScoreCalculator

from .action_generator import Action, ActionAlternative, ActionGenerator, ActionType, DecisionResult
from .matchup_monte_carlo_evaluator import MatchupAwareMonteCarloWinProbabilityEvaluator
from .monte_carlo_evaluator import MonteCarloWinProbabilityEvaluator
from .monte_carlo_rollout_strategy import MonteCarloRolloutStrategy
from .strategy import RuleBasedStrategy, Strategy


class WinProbabilityStrategy:
    """Choose a legal action using Monte Carlo win probability conservatively."""

    def __init__(
        self,
        *,
        evaluator: MonteCarloWinProbabilityEvaluator | None = None,
        simulation_count: int = 300,
        max_candidates: int | None = 12,
        continuation_strategy: Strategy | None = None,
        opponent_strategy: Strategy | None = None,
        reroll_advantage_threshold: float = 0.08,
        show_progress: bool = True,
    ) -> None:
        if simulation_count <= 0:
            raise ValueError("simulation_count must be positive.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        if not 0.0 <= reroll_advantage_threshold <= 1.0:
            raise ValueError("reroll_advantage_threshold must be between 0 and 1.")

        continuation = continuation_strategy or MonteCarloRolloutStrategy()
        opponent = opponent_strategy or RuleBasedStrategy()
        self._evaluator = evaluator or MatchupAwareMonteCarloWinProbabilityEvaluator(
            player_strategy=continuation,
            opponent_strategy=opponent,
            show_progress=show_progress,
        )
        self._simulation_count = simulation_count
        self._max_candidates = max_candidates
        self._reroll_advantage_threshold = reroll_advantage_threshold

    def decide(self, state: GameState) -> DecisionResult:
        if state.current_dice is None or state.roll_count == 0:
            raise ValueError("WinProbabilityStrategy requires a rolled hand.")
        actions = self._candidate_actions(state, self._max_candidates)
        probabilities = self._evaluator.estimate_actions_win_probability(state, actions, self._simulation_count)
        candidates = tuple(ActionAlternative(action, probabilities[action]) for action in actions)
        ordered = tuple(sorted(candidates, key=lambda candidate: self._sort_key(state, candidate), reverse=True))
        best = self._select_robust_action(state, ordered)
        return DecisionResult(
            action=best.action,
            expected_value=None,
            alternatives=ordered[:3],
            reasoning=self._reasoning(best.action, best.expected_value),
        )

    def _select_robust_action(self, state: GameState, ordered: tuple[ActionAlternative, ...]) -> ActionAlternative:
        best = ordered[0]
        if best.action.type is not ActionType.REROLL:
            return best
        score_candidates = tuple(candidate for candidate in ordered if candidate.action.type is ActionType.SCORE)
        if not score_candidates:
            return best
        best_score = max(score_candidates, key=lambda candidate: self._score_action_key(state, candidate))
        advantage = best.expected_value - best_score.expected_value
        if advantage < self._reroll_advantage_threshold:
            return best_score
        return best

    @staticmethod
    def _score_action_key(state: GameState, candidate: ActionAlternative) -> tuple[float, float, str]:
        action = candidate.action
        assert action.selected_category is not None
        assert state.current_dice is not None
        immediate = float(ScoreCalculator.calculate(action.selected_category, state.current_dice))
        bonus = 0.0
        player = state.players[state.current_player]
        if action.selected_category.is_upper and not player.has_upper_bonus:
            projected = player.upper_total + int(immediate)
            if projected >= 63:
                bonus = 35.0
            elif projected >= 50:
                bonus = min(8.75, max(0.0, projected - 42.0) * 0.4)
        return (immediate + bonus, immediate, action.selected_category.value)

    @staticmethod
    def _candidate_actions(state: GameState, max_candidates: int | None = None) -> tuple[Action, ...]:
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
            return tuple(
                sorted(
                    score_actions,
                    key=lambda action: WinProbabilityStrategy._candidate_priority(state, action),
                    reverse=True,
                )[:max_candidates]
            )

        # Preserve the original conservative 4/2 split. More score candidates
        # give the Monte Carlo evaluator enough banking choices to account for
        # upper-bonus and immediate-score states without spending simulations
        # on too many speculative rerolls.
        score_slots = min(len(score_actions), max_candidates - 2)
        reroll_slots = min(2, max_candidates - score_slots)
        if score_slots <= 0:
            reroll_slots = min(max_candidates, len(reroll_actions))
            score_slots = max_candidates - reroll_slots

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
    def _candidate_priority(state: GameState, action: Action) -> tuple[float, float, float, float, tuple[int, ...]]:
        assert state.current_dice is not None
        player = state.players[state.current_player]
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            score = float(ScoreCalculator.calculate(action.selected_category, state.current_dice))
            bonus = 0.0
            if action.selected_category.is_upper and not player.has_upper_bonus:
                projected = player.upper_total + int(score)
                if projected >= 63:
                    bonus = 35.0
                elif projected >= 50:
                    bonus = min(8.75, (projected - 42) * 0.4)
            return (score + bonus, score, bonus, 1.0, ())
        held_values = [state.current_dice[index] for index in action.held_indices]
        counts: dict[int, int] = {}
        for value in held_values:
            counts[value] = counts.get(value, 0) + 1
        duplicate_strength = max((value * count * count for value, count in counts.items()), default=0)
        straight_length = float(WinProbabilityStrategy._best_straight_length(held_values))
        upper_alignment = max((held_values.count(face) * face for face in range(1, 7)), default=0)
        return (float(duplicate_strength), straight_length, upper_alignment, float(sum(held_values)), tuple(-index for index in action.held_indices))

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
    def _sort_key(state: GameState, candidate: ActionAlternative) -> tuple[float, float, int, tuple[int, ...], str]:
        action = candidate.action
        immediate_score = 0.0
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            assert state.current_dice is not None
            immediate_score = float(ScoreCalculator.calculate(action.selected_category, state.current_dice))
        return (candidate.expected_value, immediate_score, 1 if action.type is ActionType.SCORE else 0, tuple(-index for index in action.held_indices), action.selected_category.value if action.selected_category is not None else "")

    @staticmethod
    def _reasoning(action: Action, probability: float) -> str:
        if action.type is ActionType.SCORE:
            assert action.selected_category is not None
            return f"Record {action.selected_category.display_name}; estimated win probability {probability:.1%}."
        held = ", ".join(str(index) for index in action.held_indices) or "none"
        return f"Keep dice at indices [{held}]; estimated win probability {probability:.1%}."
