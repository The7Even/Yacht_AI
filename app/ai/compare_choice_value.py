"""Compare taking Choice now against preserving it for the rest of the game.

The experiment stores large runs as 10,000-row CSV chunks. Each run gets its
own timestamped directory, and every completed chunk also receives a compact
summary CSV so the full result remains easy to inspect and share.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, median

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
CHUNK_ROWS = 10_000

_WORKER_STRATEGY: ExpectedValueStrategy | None = None
_WORKER_ROLLOUTS = 10


def _worker_init(rollouts: int) -> None:
    global _WORKER_STRATEGY, _WORKER_ROLLOUTS
    _WORKER_STRATEGY = ExpectedValueStrategy()
    _WORKER_ROLLOUTS = rollouts


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


def _new_first_turn_engine(first_dice: tuple[int, ...], seed: int) -> GameEngine:
    engine = GameEngine(dice_roller=DiceRoller(random.Random(seed)))
    engine.start_game()
    engine.state.current_dice = tuple(first_dice)
    engine.state.roll_count = 1
    return engine


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


def _paired_branch(
    first_dice: tuple[int, ...], first_category: Category, seed: int, strategy: ExpectedValueStrategy
) -> tuple[int, int]:
    branch = _new_first_turn_engine(first_dice, seed)
    _score_first_turn(branch, first_category)
    return _finish_game(branch, strategy)


def _simulate_sample(payload: tuple[int, tuple[int, ...], Category, int, int, int, int]) -> list[dict[str, object]]:
    sample, first_dice, baseline_category, choice_score, best_other_score, seed, best_other_index = payload
    strategy = _worker_strategy()
    best_other = ALL_CATEGORIES[best_other_index]
    rollout_rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for rollout in range(1, _WORKER_ROLLOUTS + 1):
        branch_seed = rollout_rng.randrange(2**63)
        choice_result = _paired_branch(first_dice, Category.CHOICE, branch_seed, strategy)
        other_result = _paired_branch(first_dice, best_other, branch_seed, strategy)
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


def _print_progress(done_branches: int, total_branches: int, samples: int, rollouts: int, workers: int) -> None:
    pct = done_branches / total_branches * 100 if total_branches else 100.0
    width = 40
    filled = int(width * pct / 100)
    bar = "#" * filled + "." * (width - filled)
    completed_samples = done_branches // (rollouts * 2)
    print(
        f"\rChoice Value [{bar}] {pct:6.2f}% | {done_branches}/{total_branches} branches "
        f"| {completed_samples}/{samples} samples | {workers} workers",
        end="", flush=True,
    )


def _print_prepare_progress(done: int, total: int) -> None:
    pct = done / total * 100 if total else 100.0
    print(f"\rPreparing first-turn states: {pct:6.2f}% | {done}/{total} samples", end="", flush=True)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [int(row["delta_choice_minus_other"]) for row in rows]
    choice_wins = sum(int(row["choice_better"]) for row in rows)
    other_wins = sum(int(row["other_better"]) for row in rows)
    draws = sum(int(row["draw"]) for row in rows)
    return {
        "rows": len(rows),
        "choice_wins": choice_wins,
        "other_wins": other_wins,
        "draws": draws,
        "choice_win_rate": choice_wins / len(rows) if rows else 0.0,
        "other_win_rate": other_wins / len(rows) if rows else 0.0,
        "draw_rate": draws / len(rows) if rows else 0.0,
        "mean_delta": mean(deltas) if deltas else 0.0,
        "median_delta": median(deltas) if deltas else 0.0,
        "min_delta": min(deltas) if deltas else 0,
        "max_delta": max(deltas) if deltas else 0,
    }


def _write_summary(path: Path, rows: list[dict[str, object]], label: str) -> None:
    summary = _summarize(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["chunk", label])
        for key, value in summary.items():
            writer.writerow([key, value])


def _write_chunk(path: Path, rows: list[dict[str, object]], chunk_number: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = path.with_name(f"{path.stem}_summary.csv")
    _write_summary(summary_path, rows, f"{chunk_number:02d}")
    return summary_path


def run(samples: int, rollouts: int, seed: int, output: Path | None = None, workers: int | None = None) -> Path:
    if samples <= 0 or rollouts <= 0:
        raise ValueError("samples and rollouts must be positive")

    # Windows does not allow ':' in directory names, so keep the requested
    # human-readable MM.DD HH:MM form while replacing only ':' with '_'.
    if output is None:
        timestamp = datetime.now().strftime("%m.%d %H_%M")
        run_dir = Path("logs") / f"choice_value_{timestamp}"
    else:
        run_dir = output
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    strategy = ExpectedValueStrategy()
    total_branches = samples * rollouts * 2
    max_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(
        f"Choice Value: starting {samples} samples × {rollouts} rollouts "
        f"({total_branches} branch simulations, {max_workers} workers)"
    )
    _print_progress(0, total_branches, samples, rollouts, max_workers)

    payloads: list[tuple[int, tuple[int, ...], Category, int, int, int, int]] = []
    _print_prepare_progress(0, samples)
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
        payloads.append((
            sample, first_dice, baseline_category,
            choice_score, best_other_score, rng.randrange(2**63),
            ALL_CATEGORIES.index(best_other),
        ))
        _print_prepare_progress(sample, samples)
    print()

    all_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    chunk_number = 1
    completed_branches = 0

    def flush_chunk() -> None:
        nonlocal chunk_rows, chunk_number
        if not chunk_rows:
            return
        filename = f"choice_value_{datetime.now().strftime('%m%d%H%M')}_{chunk_number:02d}.csv"
        path = run_dir / filename
        _write_chunk(path, chunk_rows, chunk_number)
        print(f"\nSaved chunk {chunk_number:02d}: {path} ({len(chunk_rows):,} rows)")
        chunk_number += 1
        chunk_rows = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(rollouts,)) as executor:
        futures = [executor.submit(_simulate_sample, payload) for payload in payloads]
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                chunk_rows.append(row)
                all_rows.append(row)
                completed_branches += 2
                if len(chunk_rows) >= CHUNK_ROWS:
                    flush_chunk()
            _print_progress(completed_branches, total_branches, samples, rollouts, max_workers)

    flush_chunk()

    final_summary = run_dir / f"choice_value_{datetime.now().strftime('%m%d%H%M')}_final_summary.csv"
    _write_summary(final_summary, all_rows, "final")
    _print_progress(total_branches, total_branches, samples, rollouts, max_workers)
    print(f"\nFinal summary: {final_summary}")
    print(f"Total rows written: {len(all_rows):,}")
    return run_dir


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
    raise SystemExit(main())
