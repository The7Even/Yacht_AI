"""Compare Choice against 4K / Full House as a long-term category decision.

A target hand is fixed, then two otherwise identical futures are simulated:
one branch consumes Choice and the other consumes the target category.  The
remaining categories are played by the current FastEV strategy using the same
future RNG seed, so the difference isolates the value of leaving the category
available for later.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine, MAX_ROLLS_PER_TURN
from app.core.game_state import PlayerId
from app.core.scoring import ScoreCalculator
from .action_generator import ActionType
from .expected_value_strategy import ExpectedValueStrategy

FIELDS = (
    "sample", "category", "dice", "choice_score", "target_score", "score_gap",
    "choice_final_score", "target_final_score", "delta_target_minus_choice",
    "choice_bonus", "target_bonus",
)
_WORKER_STRATEGY: ExpectedValueStrategy | None = None


def _worker_init() -> None:
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = ExpectedValueStrategy()


def _strategy() -> ExpectedValueStrategy:
    global _WORKER_STRATEGY
    if _WORKER_STRATEGY is None:
        _WORKER_STRATEGY = ExpectedValueStrategy()
    return _WORKER_STRATEGY


def _candidate_hands(category: Category) -> tuple[tuple[int, ...], ...]:
    hands = combinations_with_replacement(range(1, 7), 5)
    if category is Category.FOUR_OF_A_KIND:
        return tuple(hand for hand in hands if max(hand.count(face) for face in set(hand)) >= 4 and ScoreCalculator.calculate(category, hand) > 0)
    if category is Category.FULL_HOUSE:
        return tuple(hand for hand in hands if sorted(hand.count(face) for face in set(hand)) == [2, 3])
    raise ValueError("Only FOUR_OF_A_KIND and FULL_HOUSE are supported.")


_HANDS = {
    Category.FOUR_OF_A_KIND: _candidate_hands(Category.FOUR_OF_A_KIND),
    Category.FULL_HOUSE: _candidate_hands(Category.FULL_HOUSE),
}


def _play_future(forced_category: Category, initial_dice: tuple[int, ...], seed: int) -> tuple[int, bool]:
    """Score the forced category, then let FastEV play every remaining category."""
    engine = GameEngine(dice_roller=DiceRoller(random.Random(seed)))
    engine.start_game()
    engine.state.current_dice = initial_dice
    engine.state.roll_count = MAX_ROLLS_PER_TURN
    engine.score_category(forced_category)
    strategy = _strategy()

    while not engine.state.game_over:
        engine.state.current_player = PlayerId.PLAYER
        engine.state.current_dice = None
        engine.state.held_indices = frozenset()
        engine.state.roll_count = 0
        engine.state.turn_scored = False
        while not engine.state.turn_scored:
            engine.roll_dice()
            decision = strategy.decide(engine.state)
            if decision.action.type is ActionType.SCORE:
                assert decision.action.selected_category is not None
                engine.score_category(decision.action.selected_category)
                break
            desired = frozenset(decision.action.held_indices)
            current = engine.state.held_indices
            for index in sorted(current - desired):
                engine.unhold_dice(index)
            for index in sorted(desired - current):
                engine.hold_dice(index)
            if engine.state.roll_count >= MAX_ROLLS_PER_TURN:
                decision = strategy.decide(engine.state)
                assert decision.action.selected_category is not None
                engine.score_category(decision.action.selected_category)
                break
    player = engine.state.players[PlayerId.PLAYER]
    return player.total_score, player.has_upper_bonus


def _observe(payload: tuple[int, Category, tuple[int, ...], int]) -> dict[str, object]:
    sample, category, dice, seed = payload
    choice_score = ScoreCalculator.calculate(Category.CHOICE, dice)
    target_score = ScoreCalculator.calculate(category, dice)
    # Both branches receive the same future RNG stream.
    choice_final, choice_bonus = _play_future(Category.CHOICE, dice, seed)
    target_final, target_bonus = _play_future(category, dice, seed)
    return {
        "sample": sample,
        "category": category.value,
        "dice": "".join(map(str, dice)),
        "choice_score": choice_score,
        "target_score": target_score,
        "score_gap": target_score - choice_score,
        "choice_final_score": choice_final,
        "target_final_score": target_final,
        "delta_target_minus_choice": target_final - choice_final,
        "choice_bonus": int(choice_bonus),
        "target_bonus": int(target_bonus),
    }


def _progress(done: int, total: int, workers: int) -> None:
    pct = done / total * 100 if total else 100.0
    width = 40
    filled = int(width * pct / 100)
    print(f"\r4K/FH Future Value [{('#' * filled) + ('.' * (width - filled))}] {pct:6.2f}% | {done}/{total} comparisons | {workers} workers", end="", flush=True)


def run(samples: int, category: Category, seed: int = 0, workers: int | None = None, output: Path | None = None) -> Path:
    if samples <= 0:
        raise ValueError("samples must be positive")
    run_dir = output or Path("logs") / f"category_future_value_{datetime.now().strftime('%m.%d %H_%M')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    max_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rng = random.Random(seed)
    hands = _HANDS[category]
    payloads = [(i, category, hands[rng.randrange(len(hands))], rng.randrange(2**63)) for i in range(1, samples + 1)]

    print(f"Category Future Value: {category.value}, {samples:,} comparisons ({max_workers} workers)")
    _progress(0, samples, max_workers)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_observe, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            _progress(done, samples, max_workers)
    print()
    rows.sort(key=lambda row: int(row["sample"]))
    stamp = datetime.now().strftime("%m%d%H%M")
    data_path = run_dir / f"category_future_value_{category.value}_{stamp}.csv"
    summary_path = run_dir / f"category_future_value_{category.value}_{stamp}_summary.csv"

    with data_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

    deltas = [float(row["delta_target_minus_choice"]) for row in rows]
    gaps = [int(row["score_gap"]) for row in rows]
    target_wins = sum(delta > 0 for delta in deltas)
    choice_wins = sum(delta < 0 for delta in deltas)
    draws = len(deltas) - target_wins - choice_wins
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["category", category.value])
        writer.writerow(["comparisons", len(rows)])
        writer.writerow(["mean_score_gap_target_minus_choice", sum(gaps) / len(gaps)])
        writer.writerow(["mean_final_delta_target_minus_choice", sum(deltas) / len(deltas)])
        writer.writerow(["target_wins", target_wins])
        writer.writerow(["target_win_rate", target_wins / len(deltas)])
        writer.writerow(["choice_wins", choice_wins])
        writer.writerow(["choice_win_rate", choice_wins / len(deltas)])
        writer.writerow(["draws", draws])
        writer.writerow(["draw_rate", draws / len(deltas)])
        writer.writerow(["target_bonus_rate", sum(int(row["target_bonus"]) for row in rows) / len(rows)])
        writer.writerow(["choice_bonus_rate", sum(int(row["choice_bonus"]) for row in rows) / len(rows)])

    print(f"Saved data: {data_path}")
    print(f"Saved summary: {summary_path}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Choice vs 4K/Full House future value.")
    parser.add_argument("--category", choices=(Category.FOUR_OF_A_KIND.value, Category.FULL_HOUSE.value), required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.samples, Category(args.category), args.seed, args.workers, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
