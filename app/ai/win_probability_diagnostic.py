from __future__ import annotations

from dataclasses import dataclass

from app.core.game_state import GameState

from .action_generator import Action
from .monte_carlo_evaluator import MonteCarloWinProbabilityEvaluator


@dataclass(frozen=True)
class CandidateProbability:
    action: Action
    probability: float
    average_score: float
    average_opponent_score: float


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
        stats = self._evaluator.estimate_actions_statistics(
            state, actions, simulation_count
        )
        return tuple(
            CandidateProbability(
                action,
                result.win_probability,
                result.average_score,
                result.average_opponent_score,
            )
            for action in actions
            if (result := stats.get(action)) is not None
        )

    @staticmethod
    def format_report(candidates: tuple[CandidateProbability, ...]) -> str:
        ranked = sorted(candidates, key=lambda item: item.probability, reverse=True)
        lines = [
            "=== WinProbability Diagnostic ===",
            "ACTION                                      WIN%    AVG SCORE  AVG OPP",
            "----------------------------------------------------------------------------",
        ]
        for item in ranked:
            lines.append(
                f"{str(item.action):<44} {item.probability:6.1%}"
                f"     {item.average_score:7.2f}    {item.average_opponent_score:7.2f}"
            )
        return "\n".join(lines)
