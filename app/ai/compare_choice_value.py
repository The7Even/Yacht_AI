"""Compare taking Choice now against preserving it for the rest of the game.

This is an experiment, not a strategy change. It generates first-turn states
using FastEV, then branches from the same completed first-turn dice:

A) score Choice immediately
B) score the best available non-Choice category immediately

Each branch is then completed with the normal FastEV policy. Paired branches
use the same subsequent dice seed, making the comparison less sensitive to
unrelated random rolls.

The expensive paired rollouts are processed in parallel worker processes so
large experiments do not spend most of their time waiting on a single CPU core.

Example::
    python -m app.ai.compare_choice_value --samples 500 --rollouts 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_WORKER_STRATEGY: ExpectedValueStrategy | None = None


def _worker_init() -> None:
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = ExpectedValueStrategy()


def _worker_strategy() -> ExpectedValueStrategy:
    global _WORKER_STRATEGY
    if _WORKER_STRATEGY is None:
        _WORKER_STRATEGY = ExpectedValueStrategy()
    return _WORKER_STRATEGY


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
    # GameEngine.roll_dice() reads the private _dice_roller field.
    branch._dice_roller = DiceRoller(random.Random(seed))
    _score_first_turn(branch, first_category)
    return _finish_game(branch, strategy)


def _simulate_sample(payload: tuple[int, GameEngine, tuple[int, ...], Category, int, int]) -> list[dict[str, object]]:
    """Run all paired rollouts for one first-turn sample in one worker."""
    sample, template, first_dice, baseline_category, choice_score, best_other_score = payload
    strategy = _worker_strategy()
    best_other = max(
        (category for category in ALL_CATEGORIES if category is not Category.CHOICE),
        key=lambda category: ScoreCalculator.calculate(category, first_dice),
    )
    rows: list[dict[str, object]] = []
    # A deterministic local RNG gives each rollout a distinct continuation seed.
    rollout_rng = random.Random((sample << 32) ^ choice_score ^ best_other_score)
    for rollout in range(1, _ROLLS_PER_TASK + 1):
        branch_seed = rollout_rng.randrange(2**63)
        choice_result = _paired_branch(template, Category.CHOICE, branch_seed, strategy)
        other_result = _paired_branch(template, best_other, branch_seed, strategy)
        choice_total = choice_result[0] + choice_result[1]
        other_total = other_result[0] + other_result[1]
        delta = choice_total - other_total
        rows.append({
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
    return rows


def _print_progress(done_branches: int, total_branches: int, samples: int, rollouts: int) -> None:
    pct = done_branches / total_branches * 100 if total_branches else 100.0
    width = 40
    filled = int(width * pct / 100)
    bar = "#" * filled + "." * (width - filled)
    completed_samples = done_branches // (rollouts * 2)
    text = (
        f"\rChoice Value [{bar}] {pct:6.2f}% | {done_branches}/{total_branches} branches "
        f"| {completed_samples}/{samples} samples | workers"
    )
    print(text, end="", flush=True)


def run(samples: int, rollouts: int, seed: int, output: Path | None = None, workers: int | None = None) -> Path:
    global _ROLLS_PER_TASK
    if samples <= 0 or rollouts <= 0:
        raise ValueError("samples and rollouts must be positive")
    logs = Path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    if output is None:
        output = logs / f"choice_value_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

    rng = random.Random(seed)
    strategy = ExpectedValueStrategy()
    total_branches = samples * rollouts * 2
    _ROLLS_PER_TASK = rollouts
    max_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(
        f"Choice Value: starting {samples} samples × {rollouts} rollouts "
        f"({total_branches} branch simulations, {max_workers} workers)"
    )
    _print_progress(0, total_branches, samples, rollouts)

    payloads: list[tuple[int, GameEngine, tuple[int, ...], Category, int, int]] = []
    for sample in range(1, samples + 1):
        first_seed = rng.randrange(2**63)
        base = GameEngine(dice_roller=DiceRoller(random.Random(first_seed)))
        first_dice, baseline_category = _play_turn_to_score(base, strategy)
        choice_score = ScoreCalculator.calculate(Category.CHOICE, first_dice)
        best_other = max(
            (category for category in ALL_CATEGORIES if category is not Category.CHOICE),
            key=lambda category: ScoreCalculator.calculate(category, first_dice),
        )
        best_other_score = ScoreCalculator.calculate(best_other, first_dice)
        payloads.append((sample, copy.deepcopy(base), first_dice, baseline_category, choice_score, best_other_score))

    completed_samples = 0
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as executor:
            futures = [executor.submit(_simulate_sample, payload) for payload in payloads]
            for future in as_completed(futures):
                rows = future.result()
                writer.writerows(rows)
                handle.flush()
                completed_samples += 1
                _print_progress(completed_samples * rollouts * 2, total_branches, samples, rollouts)

    _print_progress(total_branches, total_branches, samples, rollouts)
    print(f"\nSaved: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare taking Choice against the best alternative on turn one.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.samples, args.rollouts, args.seed, args.output, args.workers)
    return 0


if __name__ == "__main__":
    import os
    _ROLLS_PER_TASK = 10
    raise SystemExit(main())
