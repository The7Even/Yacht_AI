"""Batch sampler for first-turn FastEV analysis.

Runs only the first turn of many independent games so Choice/category behavior
can be studied without simulating complete games.

Example::
    python -m app.ai.sample_first_turn --games 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from app.core.categories import ALL_CATEGORIES
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import PlayerId
from app.core.scoring import ScoreCalculator

from .action_generator import ActionType
from .expected_value_strategy import ExpectedValueStrategy

FIELDS = (
    "timestamp", "sample", "roll", "dice", "held_before", "decision_type",
    "selected_category", "selected_ev", "candidate_1_action", "candidate_1_ev",
    "candidate_2_action", "candidate_2_ev", "candidate_3_action", "candidate_3_ev",
    "available_category_scores", "reasoning", "final_dice", "final_held",
    "final_category", "final_category_score",
)


def _action_text(action) -> str:
    if action.type is ActionType.SCORE:
        return f"SCORE:{action.selected_category.name if action.selected_category else ''}"
    return f"REROLL:HOLD[{','.join(map(str, action.held_indices))}]"


def _scores(engine: GameEngine) -> str:
    player = engine.state.players[PlayerId.PLAYER]
    dice = tuple(engine.state.current_dice or ())
    return json.dumps({
        category.name: ScoreCalculator.calculate(category, dice)
        for category in ALL_CATEGORIES if category not in player.used_categories
    }, ensure_ascii=False, sort_keys=True)


def _decision_row(engine: GameEngine, sample: int, result) -> dict[str, object]:
    candidates = list(result.alternatives[:3])
    row = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "sample": sample,
        "roll": engine.state.roll_count,
        "dice": json.dumps(list(engine.state.current_dice or ())),
        "held_before": json.dumps(sorted(engine.state.held_indices)),
        "decision_type": result.action.type.value,
        "selected_category": result.selected_category.name if result.selected_category else "",
        "selected_ev": f"{result.expected_value:.6f}" if result.expected_value is not None else "",
        "available_category_scores": _scores(engine),
        "reasoning": result.reasoning,
    }
    for i in range(3):
        row[f"candidate_{i + 1}_action"] = _action_text(candidates[i].action) if i < len(candidates) else ""
        row[f"candidate_{i + 1}_ev"] = f"{candidates[i].expected_value:.6f}" if i < len(candidates) else ""
    return row


def run(games: int, seed: int = 0, output: Path | None = None) -> Path:
    if games <= 0:
        raise ValueError("games must be positive")
    logs = Path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    if output is None:
        output = logs / f"first_turn_fast_ev_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

    rng = random.Random(seed)
    strategy = ExpectedValueStrategy()
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in range(1, games + 1):
            engine = GameEngine(dice_roller=DiceRoller(random.Random(rng.randrange(2**63))))
            engine.start_game()
            engine.roll_dice()
            while engine.state.roll_count < 3:
                result = strategy.decide(engine.state)
                if result.action.type is ActionType.SCORE:
                    row = _decision_row(engine, sample, result)
                    engine.score_category(result.action.selected_category)
                    row["final_dice"] = json.dumps(list(engine.state.current_dice or ()))
                    row["final_held"] = json.dumps(sorted(engine.state.held_indices))
                    row["final_category"] = result.action.selected_category.name
                    row["final_category_score"] = ScoreCalculator.calculate(result.action.selected_category, engine.state.current_dice or ())
                    writer.writerow({f: row.get(f, "") for f in FIELDS})
                    break
                desired = frozenset(result.action.held_indices)
                for index in engine.state.held_indices - desired:
                    engine.unhold_dice(index)
                for index in desired - engine.state.held_indices:
                    engine.hold_dice(index)
                engine.roll_dice()
            else:
                # Defensive fallback; normal strategy should score by the third roll.
                result = strategy.decide(engine.state)
                row = _decision_row(engine, sample, result)
                if result.action.type is ActionType.SCORE:
                    engine.score_category(result.action.selected_category)
                    row["final_dice"] = json.dumps(list(engine.state.current_dice or ()))
                    row["final_held"] = json.dumps(sorted(engine.state.held_indices))
                    row["final_category"] = result.action.selected_category.name
                    row["final_category_score"] = ScoreCalculator.calculate(result.action.selected_category, engine.state.current_dice or ())
                writer.writerow({f: row.get(f, "") for f in FIELDS})

            if sample == games or sample % max(1, games // 100) == 0:
                percent = sample / games * 100
                width = 40
                filled = int(width * percent / 100)
                print(f"\rFirst Turn [{ '#' * filled + '.' * (width-filled) }] {percent:6.2f}% | {sample}/{games} samples", end="", flush=True)

    print(f"\nSaved: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample FastEV decisions for first-turn analysis.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.games, args.seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
