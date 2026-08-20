"""Development-only console runner for a complete automated Yacht game."""

from collections.abc import Callable

from app.ai.action_generator import ActionType, DecisionResult
from app.ai.ai_player import AIPlayer
from app.core.dice import DiceRoller
from app.core.game_engine import GameEngine
from app.core.game_state import PlayerId


def play_console_game(
    *, dice_roller: DiceRoller | None = None, output: Callable[[str], None] = print
) -> GameEngine:
    """Run a full game using the baseline policy for both participants."""
    engine = GameEngine(dice_roller)
    policy = AIPlayer()
    engine.start_game()
    output("Yacht console game started.")

    while not engine.is_game_over():
        state = engine.state
        output(f"Turn {state.turn}: {state.current_player.value}")
        engine.roll_dice()
        _print_roll(engine, output)
        decision = policy.decide(state)

        while decision.action.type is ActionType.REROLL:
            _apply_reroll_choice(engine, decision)
            output(f"  {decision.reasoning}")
            engine.roll_dice()
            _print_roll(engine, output)
            decision = policy.decide(state)

        category = decision.selected_category
        if decision.action.type is not ActionType.SCORE or category is None:
            raise RuntimeError("Baseline policy must finish its turn by scoring.")
        score = engine.score_category(category)
        player = state.players[state.current_player]
        output(f"  Score {category.display_name}: {score}")
        output(
            f"  Total: {player.total_score}; upper: {player.upper_total}; "
            f"bonus: {'yes' if player.has_upper_bonus else 'no'}"
        )
        if not engine.is_game_over():
            engine.end_turn()

    _print_final_result(engine, output)
    return engine


def _apply_reroll_choice(engine: GameEngine, decision: DecisionResult) -> None:
    desired = frozenset(decision.held_indices)
    for index in engine.state.held_indices - desired:
        engine.unhold_dice(index)
    for index in desired - engine.state.held_indices:
        engine.hold_dice(index)


def _print_roll(engine: GameEngine, output: Callable[[str], None]) -> None:
    state = engine.state
    output(f"  Roll {state.roll_count}: {state.current_dice}; held: {state.held_dice}")


def _print_final_result(engine: GameEngine, output: Callable[[str], None]) -> None:
    state = engine.state
    output("Game over.")
    output(f"Player: {state.player_score} (bonus: {'yes' if state.player_has_bonus else 'no'})")
    output(f"AI: {state.ai_score} (bonus: {'yes' if state.ai_has_bonus else 'no'})")
    if state.player_score == state.ai_score:
        output("Winner: draw")
    else:
        winner = PlayerId.PLAYER if state.player_score > state.ai_score else PlayerId.AI
        output(f"Winner: {winner.value}")


if __name__ == "__main__":
    play_console_game()
