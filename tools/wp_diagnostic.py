from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running this file directly (python tools/wp_diagnostic.py) puts `tools/`
# on sys.path, not the project root. Add the root so `app` can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    # GameState is intentionally a data-only container; game lifecycle changes
    # belong to GameEngine, so construct the diagnostic state directly.
    state = GameState(
        current_dice=tuple(args.dice),
        roll_count=args.roll,
        game_started=True,
    )

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
