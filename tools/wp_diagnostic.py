from __future__ import annotations

import argparse

from app.ai.win_probability_diagnostic import WinProbabilityDiagnostic
from app.ai.win_probability_strategy import WinProbabilityStrategy
from app.core.game_state import GameState


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect WinProbabilityStrategy candidate probabilities.")
    parser.add_argument("--dice", nargs=5, type=int, default=[5, 5, 5, 2, 6], metavar="DIE")
    parser.add_argument("--roll", type=int, default=2, help="Current roll number (1-3).")
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    if any(die < 1 or die > 6 for die in args.dice):
        raise SystemExit("Each die must be between 1 and 6.")
    if not 1 <= args.roll <= 3:
        raise SystemExit("--roll must be between 1 and 3.")

    state = GameState()
    state.start_game()
    state.current_dice = tuple(args.dice)
    state.roll_count = args.roll

    strategy = WinProbabilityStrategy(
        simulation_count=args.simulations,
        max_candidates=args.max_candidates,
    )
    actions = strategy._candidate_actions(state, args.max_candidates)
    diagnostic = WinProbabilityDiagnostic(strategy._evaluator)
    candidates = diagnostic.analyze(state, actions, args.simulations)

    print(f"Dice: {list(args.dice)}")
    print(f"Roll: {args.roll}")
    print(f"Simulation count: {args.simulations}")
    print(f"Candidates: {len(candidates)}")
    print()
    print(diagnostic.format_report(candidates))
    print()

    result = strategy.decide(state)
    print(f"Selected: {result.action}")
    print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
