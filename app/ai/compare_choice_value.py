"""Compare taking Choice now against preserving it for the rest of the game.

This is an experiment, not a strategy change. It generates first-turn states
using FastEV, then branches from the same completed first-turn dice:

A) score Choice immediately
B) score the best available non-Choice category immediately

Each branch is then completed with the normal FastEV policy. Paired branches
use the same subsequent dice seed, making the comparison less sensitive to
unrelated random rolls. The reported final score is the total score of both
players; the two branches differ only in the forced first-turn category.

Example::
    python -m app.ai.compare_choice_value --samples 1000 --rollouts 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import PlayerId
from app.core.scoring import ScoreCalculator

from .action_generator import ActionType
from .expected_value_strategy import ExpectedValueStrategy

FIELDS = (
    "sample", "first_dice", "choice_score", "best_other_category", "best_other_score",
    "score_gap", "baseline_category", "branch_rollout", "choice_total_score",
    "other_total_score", "delta_choice_minus_other", "choice_better", "other_better", "draw",
)


def _play_turn_to_score(engine: GameEngine, strategy: ExpectedValueStrategy) -> tuple[tuple[int, ...], Category]:
    engine.start_game()
    engine.roll_dice()
    while True:
        result = strategy.decide(engine.state)
        action = result.action
        if action.type is ActionType.SCORE:
            category = action.selected_category
            assert category is not None
            return tuple(engine.state.current_dice or ()), category
        desired = frozenset(action.held_indices)
        for index in engine.state.held_indices - desired:
            engine.unhold_dice(index)
        for index in desired - engine.state.held_indices:
            engine.hold_dice(index)
        engine.roll_dice()


def _score_first_turn(engine: GameEngine, category: Category) -> None:
    engine.score_category(category)
    if not engine.is_game_over():
        engine.end_turn()


def _finish_game(engine: GameEngine, strategy: ExpectedValueStrategy) -> tuple[int, int]:
    while not engine.is_game_over():
        while not engine.state.turn_scored:
            if engine.state.current_dice is None:
                engine.roll_dice()
            result = strategy.decide(engine.state)
            action = result.action
            if action.type is ActionType.SCORE:
                assert action.selected_category is not None
                engine.score_category(action.selected_category)
            else:
                desired = frozenset(action.held_indices)
                for index in engine.state.held_indices - desired:
                    engine.unhold_dice(index)
                for index in desired - engine.state.held_indices:
                    engine.hold_dice(index)
                engine.roll_dice()
        if not engine.is_game_over():
            engine.end_turn()
    return (
        engine.state.players[PlayerId.PLAYER].total_score,
        engine.state.players[PlayerId.AI].total_score,
    )


def _paired_branch(template: GameEngine, first_category: Category, seed: int, strategy: ExpectedValueStrategy) -> tuple[int, int]:
    branch = copy.deepcopy(template)
    # GameEngine.roll_dice() reads the private _dice_roller field. The previous
    # version accidentally assigned a new public `dice_roller` attribute, so
    # every rollout kept using the copied RNG state and produced identical
    # results. Inject the seeded roller into the field the engine actually uses.
    branch._dice_roller = DiceRoller(random.Random(seed))
    _score_first_turn(branch, first_category)
    return _finish_game(branch, strategy)


def _print_progress(done: int, total: int, samples: int, rollouts: int) -> None:
    """Print one in-place progress line, including the current sample/rollout."""
    pct = done / total * 100 if total else 100.0
    width = 40
    filled = int(width * pct / 100)
    bar = "#" * filled + "." * (width - filled)
    current_sample = min(samples, done // max(1, rollouts * 2) + (1 if done % max(1, rollouts * 2) else 0))
    branch_in_pair = done % 2
    current_rollout = ((done // 2) % max(1, rollouts)) + (1 if done < total else 0)
    if done == total:
        current_sample = samples
        current_rollout = rollouts
        branch_in_pair = 0
    branch_name = "Choice" if branch_in_pair == 1 else "Other"
    text = (
        f"\rChoice Value [{bar}] {pct:6.2f}% | {done}/{total} branches "
        f"| sample {current_sample}/{samples} | rollout {current_rollout}/{rollouts} | {branch_name}"
    )
    print(text, end="", flush=True)


def run(samples: int, rollouts: int, seed: int, output: Path | None = None) -> Path:
    if samples <= 0 or rollouts <= 0:
        raise ValueError("samples and rollouts must be positive")
    logs = Path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    if output is None:
        output = logs / f"choice_value_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

    rng = random.Random(seed)
    strategy = ExpectedValueStrategy()
    total = samples * rollouts * 2
    done = 0

    print(
        f"Choice Value: starting {samples} samples × {rollouts} rollouts "
        f"({total} branch simulations)"
    )
    _print_progress(done, total, samples, rollouts)

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in range(1, samples + 1):
            first_seed = rng.randrange(2**63)
            base = GameEngine(dice_roller=DiceRoller(random.Random(first_seed)))
            first_dice, baseline_category = _play_turn_to_score(base, strategy)

            choice_score = ScoreCalculator.calculate(Category.CHOICE, first_dice)
            available = [c for c in ALL_CATEGORIES if c is not Category.CHOICE]
            best_other = max(available, key=lambda c: ScoreCalculator.calculate(c, first_dice))
            best_other_score = ScoreCalculator.calculate(best_other, first_dice)
            template = copy.deepcopy(base)

            for rollout in range(1, rollouts + 1):
                branch_seed = rng.randrange(2**63)

                choice_result = _paired_branch(template, Category.CHOICE, branch_seed, strategy)
                done += 1
                _print_progress(done, total, samples, rollouts)

                other_result = _paired_branch(template, best_other, branch_seed, strategy)
                done += 1
                _print_progress(done, total, samples, rollouts)

                choice_total = choice_result[0] + choice_result[1]
                other_total = other_result[0] + other_result[1]
                delta = choice_total - other_total

                writer.writerow({
                    "sample": sample,
                    "first_dice": json.dumps(list(first_dice)),
                    "choice_score": choice_score,
                    "best_other_category": best_other.name,
                    "best_other_score": best_other_score,
                    "score_gap": choice_score - best_other_score,
                    "baseline_category": baseline_category.name,
                    "branch_rollout": rollout,
                    "choice_total_score": choice_total,
                    "other_total_score": other_total,
                    "delta_choice_minus_other": delta,
                    "choice_better": int(delta > 0),
                    "other_better": int(delta < 0),
                    "draw": int(delta == 0),
                })
            handle.flush()

    _print_progress(total, total, samples, rollouts)
    print(f"\nSaved: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare taking Choice against the best alternative on turn one.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.samples, args.rollouts, args.seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
