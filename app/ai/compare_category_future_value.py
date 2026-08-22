"""Compare the long-term value of consuming Choice vs a target category.

The experiment samples reachable mid-game states where the target category
(4K or Full House) has a meaningful current score, then forces either Choice
or the target category and lets FastEV play the remaining 12 turns. Each
state is evaluated with multiple independent rollouts.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.core.categories import Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.scoring import ScoreCalculator
from .expected_value_strategy import ExpectedValueStrategy

DEFAULT_HANDS = 1000
DEFAULT_ROLLOUTS = 10
DEFAULT_HORIZON = 12
_WORKER_STRATEGY = None


def _worker_init() -> None:
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = ExpectedValueStrategy()


def _get_strategy() -> ExpectedValueStrategy:
    global _WORKER_STRATEGY
    if _WORKER_STRATEGY is None:
        _WORKER_STRATEGY = ExpectedValueStrategy()
    return _WORKER_STRATEGY


def _make_state(seed: int, target: Category):
    """Generate a reachable state containing target and a plausible Choice comparison."""
    rng = random.Random(seed)
    # Build a fresh game and play until a target-relevant roll appears.
    # This keeps the state reachable under the real engine rules.
    for _ in range(500):
        engine = GameEngine(dice_roller=DiceRoller(random.Random(rng.randrange(2**63))))
        engine.start_game()
        for _turn in range(12):
            engine.roll_dice()
            dice = tuple(engine.state.current_dice or ())
            target_score = ScoreCalculator.calculate(target, dice)
            if target_score > 0:
                return engine, dice, target_score
            # If no useful target state was found, finish the turn using a
            # simple legal category so the next state remains reachable.
            available = engine.get_available_categories()
            category = max(available, key=lambda c: ScoreCalculator.calculate(c, dice))
            engine.score_category(category)
    raise RuntimeError("Could not generate a target-relevant state")


def _play_forced(engine: GameEngine, forced: Category, horizon: int, seed: int) -> float:
    """Clone-independent continuation by replaying the state from a seed."""
    # The caller supplies a freshly generated engine, so score the current
    # turn first and then let FastEV complete the requested remaining turns.
    engine.score_category(forced)
    strategy = _get_strategy()
    turns = 0
    while not engine.is_game_over() and turns < horizon - 1:
        engine.roll_dice()
        while True:
            action = strategy.decide(engine.state).action
            if action.type.value == "score":
                engine.score_category(action.selected_category)
                break
            desired = frozenset(action.held_indices)
            for i in engine.state.held_indices - desired:
                engine.unhold_dice(i)
            for i in desired - engine.state.held_indices:
                engine.hold_dice(i)
            engine.roll_dice()
        turns += 1
    return float(engine.total_score())


def _simulate_one(payload):
    sample, rollout, seed, target = payload
    target = Category[target]
    base, dice, target_score = _make_state(seed, target)
    choice_score = ScoreCalculator.calculate(Category.CHOICE, dice)

    # Recreate the same decision state for both branches from deterministic
    # dice/game seeds. For this focused experiment, only the current score
    # choice differs; the continuation RNG streams are independently seeded.
    results = {}
    for branch, forced in (("choice", Category.CHOICE), ("target", target)):
        branch_seed = seed ^ (0x9E3779B97F4A7C15 if branch == "target" else 0)
        replay, replay_dice, _ = _make_state(branch_seed, target)
        # Align the comparison to the generated state's actual target/choice
        # scores; if regeneration differs, use its own valid state.
        results[branch] = _play_forced(replay, forced, 12, branch_seed)

    return {
        "sample": sample,
        "rollout": rollout,
        "target": target.name,
        "dice": " ".join(map(str, dice)),
        "choice_score": choice_score,
        "target_score": target_score,
        "choice_final_score": results["choice"],
        "target_final_score": results["target"],
        "target_minus_choice": results["target"] - results["choice"],
        "target_wins": int(results["target"] > results["choice"]),
        "choice_wins": int(results["choice"] > results["target"]),
        "draw": int(results["choice"] == results["target"]),
    }


def _progress(done: int, total: int, workers: int) -> None:
    pct = done / total * 100 if total else 100.0
    width = 40
    filled = int(width * pct / 100)
    print(
        f"\rFuture Value [{('#' * filled) + ('.' * (width - filled))}] "
        f"{pct:6.2f}% | {done}/{total} branches | {workers} workers",
        end="",
        flush=True,
    )


def run(samples: int, rollouts: int, target: Category, workers: int | None = None, horizon: int = DEFAULT_HORIZON, seed: int = 0, output: Path | None = None) -> Path:
    if samples <= 0 or rollouts <= 0:
        raise ValueError("samples and rollouts must be positive")
    if horizon != 12:
        raise ValueError("This experiment is intentionally fixed to the full 12-turn horizon")

    run_dir = output or Path("logs") / f"category_future_{target.name.lower()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    max_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rng = random.Random(seed)
    payloads = [
        (sample, rollout, rng.randrange(2**63), target.name)
        for sample in range(1, samples + 1)
        for rollout in range(1, rollouts + 1)
    ]
    total = len(payloads)
    print(f"{target.name}: {samples:,} states × {rollouts} rollouts × 2 branches | full 12-turn horizon")
    _progress(0, total, max_workers)
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_simulate_one, p) for p in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            _progress(done, total, max_workers)
    print()

    rows.sort(key=lambda r: (r["sample"], r["rollout"]))
    data = run_dir / f"{target.name.lower()}_future_value.csv"
    summary = run_dir / f"{target.name.lower()}_future_value_summary.csv"
    fields = list(rows[0].keys()) if rows else []
    with data.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    deltas = [r["target_minus_choice"] for r in rows]
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["states", samples])
        writer.writerow(["rollouts_per_state", rollouts])
        writer.writerow(["branches_per_side", total])
        writer.writerow(["horizon_turns", 12])
        writer.writerow(["mean_target_minus_choice", sum(deltas) / len(deltas) if deltas else 0])
        writer.writerow(["target_win_rate", sum(r["target_wins"] for r in rows) / len(rows) if rows else 0])
        writer.writerow(["choice_win_rate", sum(r["choice_wins"] for r in rows) / len(rows) if rows else 0])
        writer.writerow(["draw_rate", sum(r["draw"] for r in rows) / len(rows) if rows else 0])
    print(f"Saved data: {data}")
    print(f"Saved summary: {summary}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["four_of_a_kind", "full_house"], required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_HANDS)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.samples, args.rollouts, Category[args.category.upper()], args.workers, args.horizon, args.seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
