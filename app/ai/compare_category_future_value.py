"""Compare the full-game value of consuming Choice vs 4K/Full House by turn.

Each sample reaches a randomly selected PLAYER turn using FastEV, then looks
for a positive target-category first roll. The exact same reachable state is
copied into Choice and target branches, and both branches are played through
the remaining PLAYER turns. Results include the originating turn so future
value can be analyzed by game phase.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.core.categories import Category
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine, MAX_ROLLS_PER_TURN
from app.core.game_state import PlayerId
from app.core.scoring import ScoreCalculator
from .action_generator import ActionType
from .expected_value_strategy import ExpectedValueStrategy

DEFAULT_HANDS = 1000
DEFAULT_ROLLOUTS = 10
DEFAULT_PLAYER_TURNS = 12
DEFAULT_WORKERS = 8
_WORKER_STRATEGY = None


def _worker_init() -> None:
    global _WORKER_STRATEGY
    _WORKER_STRATEGY = ExpectedValueStrategy()


def _get_strategy() -> ExpectedValueStrategy:
    global _WORKER_STRATEGY
    if _WORKER_STRATEGY is None:
        _WORKER_STRATEGY = ExpectedValueStrategy()
    return _WORKER_STRATEGY


def _play_turn(engine: GameEngine, strategy: ExpectedValueStrategy) -> None:
    """Play the current player's turn using FastEV until it is scored."""
    while not engine.state.turn_scored:
        if engine.state.roll_count == 0:
            engine.roll_dice()
        decision = strategy.decide(engine.state)
        action = decision.action
        if action.type is ActionType.SCORE:
            if action.selected_category is None:
                raise RuntimeError("FastEV returned SCORE without a category")
            engine.score_category(action.selected_category)
            return

        desired_held = frozenset(action.held_indices)
        current_held = engine.state.held_indices
        for index in sorted(current_held - desired_held):
            engine.unhold_dice(index)
        for index in sorted(desired_held - current_held):
            engine.hold_dice(index)

        if engine.state.roll_count >= MAX_ROLLS_PER_TURN:
            raise RuntimeError("FastEV requested a reroll after the third roll")
        engine.roll_dice()


def _play_ai_turn(engine: GameEngine, strategy: ExpectedValueStrategy) -> None:
    if engine.state.current_player is not PlayerId.AI:
        raise RuntimeError("Expected AI turn")
    _play_turn(engine, strategy)
    engine.finish_ai_turn()


def _play_player_turn(engine: GameEngine, strategy: ExpectedValueStrategy) -> None:
    if engine.state.current_player is not PlayerId.PLAYER:
        raise RuntimeError("Expected PLAYER turn")
    _play_turn(engine, strategy)
    if not engine.state.game_over:
        engine.end_turn()


def _reach_candidate_state(seed: int, target: Category) -> tuple[GameEngine, tuple[int, ...], int, int]:
    """Reach a random PLAYER turn and return a target-positive first roll.

    Earlier PLAYER turns are played normally by FastEV, including intervening
    AI turns. A candidate is accepted only while both Choice and target remain
    available, so the comparison is meaningful.
    """
    rng = random.Random(seed)
    strategy = _get_strategy()

    for _attempt in range(2000):
        roller_seed = rng.randrange(2**63)
        engine = GameEngine(dice_roller=DiceRoller(random.Random(roller_seed)))
        engine.start_game()
        target_turn = rng.randrange(DEFAULT_PLAYER_TURNS)  # 0-based

        for completed in range(target_turn):
            _play_player_turn(engine, strategy)
            if engine.state.game_over:
                break
            _play_ai_turn(engine, strategy)

        if engine.state.game_over or engine.state.current_player is not PlayerId.PLAYER:
            continue
        if target not in engine.get_available_categories() or Category.CHOICE not in engine.get_available_categories():
            continue

        engine.roll_dice()
        dice = tuple(engine.state.current_dice or ())
        target_score = ScoreCalculator.calculate(target, dice)
        if target_score > 0:
            return engine, dice, target_score, target_turn + 1

    raise RuntimeError(f"Could not generate a reachable {target.value} state")


def _continue_after_forced_score(engine: GameEngine) -> float:
    """Continue from a scored PLAYER turn through the player's final turn."""
    strategy = _get_strategy()
    if engine.state.game_over:
        return float(engine.state.player_score)

    engine.end_turn()  # scored PLAYER -> AI
    while len(engine.state.player_category_scores) < DEFAULT_PLAYER_TURNS:
        if engine.state.current_player is PlayerId.AI:
            _play_ai_turn(engine, strategy)
        else:
            _play_player_turn(engine, strategy)
    return float(engine.state.player_score)


def _simulate_one(payload: tuple[int, int, int, str]) -> dict[str, object]:
    sample, rollout, seed, target_name = payload
    target = Category[target_name]
    base, dice, target_score, turn = _reach_candidate_state(seed, target)
    choice_score = ScoreCalculator.calculate(Category.CHOICE, dice)

    choice_branch = copy.deepcopy(base)
    target_branch = copy.deepcopy(base)
    choice_branch.score_category(Category.CHOICE)
    target_branch.score_category(target)

    choice_final = _continue_after_forced_score(choice_branch)
    target_final = _continue_after_forced_score(target_branch)

    return {
        "sample": sample,
        "rollout": rollout,
        "turn": turn,
        "target": target.value,
        "dice": " ".join(map(str, dice)),
        "choice_score": choice_score,
        "target_score": target_score,
        "choice_final_score": choice_final,
        "target_final_score": target_final,
        "target_minus_choice": target_final - choice_final,
        "target_wins": int(target_final > choice_final),
        "choice_wins": int(choice_final > target_final),
        "draw": int(target_final == choice_final),
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


def run(
    samples: int,
    rollouts: int,
    target: Category,
    workers: int | None = None,
    seed: int = 0,
    output: Path | None = None,
) -> Path:
    if samples <= 0 or rollouts <= 0:
        raise ValueError("samples and rollouts must be positive")

    run_dir = output or Path("logs") / f"category_future_by_turn_{target.name.lower()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    max_workers = workers if workers is not None else DEFAULT_WORKERS
    max_workers = max(1, max_workers)

    rng = random.Random(seed)
    payloads = [
        (sample, rollout, rng.randrange(2**63), target.name)
        for sample in range(1, samples + 1)
        for rollout in range(1, rollouts + 1)
    ]
    total = len(payloads)

    print(
        f"{target.name}: {samples:,} states × {rollouts} rollouts × 2 branches | "
        f"random PLAYER turn across {DEFAULT_PLAYER_TURNS} turns | {max_workers} workers"
    )
    _progress(0, total, max_workers)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_simulate_one, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            _progress(done, total, max_workers)
    print()

    rows.sort(key=lambda row: (int(row["turn"]), int(row["sample"]), int(row["rollout"])))
    data = run_dir / f"{target.name.lower()}_future_value_by_turn.csv"
    summary = run_dir / f"{target.name.lower()}_future_value_by_turn_summary.csv"
    fields = list(rows[0].keys()) if rows else []
    with data.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate by originating PLAYER turn so the effect can be inspected
    # without needing a separate analysis script.
    by_turn = run_dir / f"{target.name.lower()}_future_value_by_turn_aggregate.csv"
    with by_turn.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "turn", "observations", "mean_target_minus_choice",
            "target_win_rate", "choice_win_rate", "draw_rate",
        ])
        for turn in range(1, DEFAULT_PLAYER_TURNS + 1):
            subset = [row for row in rows if int(row["turn"]) == turn]
            if not subset:
                continue
            deltas = [float(row["target_minus_choice"]) for row in subset]
            writer.writerow([
                turn,
                len(subset),
                sum(deltas) / len(deltas),
                sum(int(row["target_wins"]) for row in subset) / len(subset),
                sum(int(row["choice_wins"]) for row in subset) / len(subset),
                sum(int(row["draw"]) for row in subset) / len(subset),
            ])

    deltas = [float(row["target_minus_choice"]) for row in rows]
    with summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["states", samples])
        writer.writerow(["rollouts_per_state", rollouts])
        writer.writerow(["branches_per_side", total])
        writer.writerow(["player_turns_evaluated", DEFAULT_PLAYER_TURNS])
        writer.writerow(["workers", max_workers])
        writer.writerow(["mean_target_minus_choice", sum(deltas) / len(deltas) if deltas else 0])
        writer.writerow(["target_win_rate", sum(int(row["target_wins"]) for row in rows) / len(rows) if rows else 0])
        writer.writerow(["choice_win_rate", sum(int(row["choice_wins"]) for row in rows) / len(rows) if rows else 0])
        writer.writerow(["draw_rate", sum(int(row["draw"]) for row in rows) / len(rows) if rows else 0])

    print(f"Saved data: {data}")
    print(f"Saved turn aggregate: {by_turn}")
    print(f"Saved summary: {summary}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["four_of_a_kind", "full_house"], required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_HANDS)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(
        args.samples,
        args.rollouts,
        Category[args.category.upper()],
        args.workers,
        args.seed,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
