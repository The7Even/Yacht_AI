"""Batch sampler for analyzing FastEV decisions across many games.

Run from the repository root with, for example::

    python -m app.ai.sample_fast_ev --games 100

The sampler uses the real GameEngine and ExpectedValueStrategy (FastEV) but does
not open the GUI. It writes one analysis CSV under ``logs/`` containing one row
per decision plus game-result rows. Each decision keeps the selected action,
the top candidate EVs, and immediate scores for every currently available
category so later analysis can distinguish an EV decision from a category bias.
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

from .action_generator import ActionType, DecisionResult
from .expected_value_strategy import ExpectedValueStrategy


FIELDS = (
    "timestamp", "game", "turn", "player", "roll", "dice", "held_before",
    "decision_type", "selected_category", "selected_ev",
    "candidate_1_action", "candidate_1_ev", "candidate_2_action", "candidate_2_ev",
    "candidate_3_action", "candidate_3_ev", "available_category_scores", "reasoning",
    "game_event", "player_total", "ai_total", "winner",
)


def _action_text(action) -> str:
    if action.type is ActionType.SCORE:
        return f"SCORE:{action.selected_category.name if action.selected_category else ''}"
    return f"REROLL:HOLD[{','.join(map(str, action.held_indices))}]"


def _available_scores(engine: GameEngine) -> str:
    state = engine.state
    player = state.players[state.current_player]
    dice = tuple(state.current_dice or ())
    values = {
        category.name: ScoreCalculator.calculate(category, dice)
        for category in ALL_CATEGORIES
        if category not in player.used_categories
    }
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def _decision_row(game_no: int, engine: GameEngine, result: DecisionResult) -> dict[str, object]:
    state = engine.state
    candidates = list(result.alternatives[:3])
    row: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "game": game_no,
        "turn": state.turn,
        "player": state.current_player.name,
        "roll": state.roll_count,
        "dice": json.dumps(list(state.current_dice or ())),
        "held_before": json.dumps(sorted(state.held_indices)),
        "decision_type": result.action.type.value,
        "selected_category": result.selected_category.name if result.selected_category else "",
        "selected_ev": f"{result.expected_value:.6f}" if result.expected_value is not None else "",
        "available_category_scores": _available_scores(engine),
        "reasoning": result.reasoning,
        "game_event": "DECISION",
    }
    for index in range(3):
        if index < len(candidates):
            row[f"candidate_{index + 1}_action"] = _action_text(candidates[index].action)
            row[f"candidate_{index + 1}_ev"] = f"{candidates[index].expected_value:.6f}"
        else:
            row[f"candidate_{index + 1}_action"] = ""
            row[f"candidate_{index + 1}_ev"] = ""
    return row


def _play_game(engine: GameEngine, strategy: ExpectedValueStrategy, game_no: int, writer: csv.DictWriter) -> None:
    engine.start_game()
    while not engine.is_game_over():
        while not engine.state.turn_scored:
            if engine.state.current_dice is None:
                engine.roll_dice()

            result = strategy.decide(engine.state)
            row = _decision_row(game_no, engine, result)
            writer.writerow({field: row.get(field, "") for field in FIELDS})

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

    player_score = engine.state.players[PlayerId.PLAYER].total_score
    ai_score = engine.state.players[PlayerId.AI].total_score
    winner = "AI" if ai_score > player_score else "PLAYER" if player_score > ai_score else "DRAW"
    writer.writerow({
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "game": game_no,
        "game_event": "GAME_RESULT",
        "player_total": player_score,
        "ai_total": ai_score,
        "winner": winner,
    })


def run(games: int, seed: int | None, output: Path | None = None) -> Path:
    if games <= 0:
        raise ValueError("games must be positive")

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output = logs_dir / f"fast_ev_sample_{stamp}.csv"

    rng = random.Random(seed)
    strategy = ExpectedValueStrategy()
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for game_no in range(1, games + 1):
            game_seed = rng.randrange(0, 2**63) if seed is not None else None
            engine = GameEngine(dice_roller=DiceRoller(random.Random(game_seed)))
            _play_game(engine, strategy, game_no, writer)
            handle.flush()
            print(f"[{game_no}/{games}] complete")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect FastEV decision samples without opening the GUI.")
    parser.add_argument("--games", type=int, default=100, help="Number of complete self-play games to sample (default: 100).")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducible samples (default: 0).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path. Defaults to logs/fast_ev_sample_YYYYMMDDHHMMSS.csv.")
    args = parser.parse_args()
    path = run(args.games, args.seed, args.output)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
