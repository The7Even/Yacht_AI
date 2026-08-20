from __future__ import annotations

from dataclasses import dataclass

from app.ai.monte_carlo_evaluator import MonteCarloWinProbabilityEvaluator
from app.ai.strategy import DecisionResult
from app.domain.action import Action
from app.domain.game_state import GameState


@dataclass(frozen=True)
class CandidateProbability:
    action: Action
    probability: float


class WinProbabilityDiagnostic:
    """Non-invasive inspection helper for WinProbabilityStrategy decisions."""

    def __init__(self, evaluator: MonteCarloWinProbabilityEvaluator) -> None:
        self._evaluator = evaluator

    def analyze(
        self,
        state: GameState,
        actions: tuple[Action, ...],
        simulation_count: int,
    ) -> tuple[CandidateProbability, ...]:
        probabilities = self._evaluator.estimate_actions_win_probability(
            state, actions, simulation_count
        )
        return tuple(
            CandidateProbability(action, probabilities[action])
            for action in actions
            if action in probabilities
        )

    @staticmethod
    def format_report(candidates: tuple[CandidateProbability, ...]) -> str:
        ranked = sorted(candidates, key=lambda item: item.probability, reverse=True)
        lines = [
            "=== WinProbability Diagnostic ===",
            "ACTION                                      WIN%",
            "------------------------------------------------------",
        ]
        for item in ranked:
            lines.append(f"{str(item.action):<44} {item.probability:6.1%}")
        return "\n".join(lines)
