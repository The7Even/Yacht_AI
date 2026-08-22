"""Game progression and state validation for Yacht."""

from .categories import ALL_CATEGORIES, Category
from .dice import DICE_COUNT, DiceRoller
from .game_state import GameState, PlayerId
from .scoring import ScoreCalculator


MAX_ROLLS_PER_TURN = 3


class GameEngine:
    """Applies valid player actions to a :class:`GameState`."""

    def __init__(self, dice_roller: DiceRoller | None = None) -> None:
        self._dice_roller = dice_roller or DiceRoller()
        self.state = GameState()
        self.last_ai_action = None
        self.last_ai_reasoning = ""

    def start_game(self) -> GameState:
        """Replace any previous game with a fresh game at the player's first turn."""
        self.state = GameState(game_started=True, turn=1)
        self.last_ai_action = None
        self.last_ai_reasoning = ""
        self.start_turn()
        return self.state

    def start_turn(self) -> GameState:
        """Prepare the current participant's next turn without rolling dice."""
        self._require_active_game()
        if self.state.current_dice is not None or self.state.roll_count or self.state.turn_scored:
            raise RuntimeError("The current turn is already in progress.")
        return self.state

    def roll_dice(self) -> tuple[int, ...]:
        """Roll all dice initially, then reroll only dice that are not held."""
        self._require_turn_in_progress()
        if self.state.roll_count >= MAX_ROLLS_PER_TURN:
            raise RuntimeError("A turn cannot contain more than three rolls.")

        if self.state.current_dice is None:
            self.state.current_dice = self._dice_roller.roll(DICE_COUNT)
        else:
            self.state.current_dice = self._dice_roller.reroll(
                self.state.current_dice, self.state.held_indices
            )
        self.state.roll_count += 1
        return self.state.current_dice

    def hold_dice(self, index: int) -> None:
        """Mark one rolled die to be retained during a future reroll."""
        self._require_turn_with_dice()
        self._validate_index(index)
        if index in self.state.held_indices:
            raise ValueError(f"Die at index {index} is already held.")
        self.state.held_indices = self.state.held_indices | {index}

    def unhold_dice(self, index: int) -> None:
        """Remove the hold from one die."""
        self._require_turn_with_dice()
        self._validate_index(index)
        if index not in self.state.held_indices:
            raise ValueError(f"Die at index {index} is not held.")
        self.state.held_indices = self.state.held_indices - {index}

    def get_available_categories(self) -> tuple[Category, ...]:
        """Return unused categories for the active participant.

        After the game ends this becomes a read-only query returning no
        categories, so the GUI can safely render the final state without
        accidentally treating the query as a player action.
        """
        if self.state.game_over:
            return ()
        self._require_active_game()
        used = self.state.players[self.state.current_player].used_categories
        return tuple(category for category in ALL_CATEGORIES if category not in used)

    def get_current_scores(self) -> dict[Category, int]:
        """Return potential scores in every currently available category."""
        if self.state.game_over:
            return {}
        self._require_turn_with_dice()
        return {
            category: ScoreCalculator.calculate(category, self.state.current_dice)
            for category in self.get_available_categories()
        }

    def score_category(self, category: Category) -> int:
        """Commit the current hand to one unused category and return its score."""
        self._require_turn_with_dice()
        if not isinstance(category, Category):
            raise ValueError("category must be a Category value.")
        if category not in self.get_available_categories():
            raise ValueError(f"Category {category.value} has already been used.")

        score = ScoreCalculator.calculate(category, self.state.current_dice)
        self.state.players[self.state.current_player].category_scores[category] = score
        self.state.turn_scored = True
        self.state.held_indices = frozenset()
        if all(len(player.category_scores) == len(ALL_CATEGORIES) for player in self.state.players.values()):
            self.state.game_over = True
        return score

    def end_turn(self, player_only: bool = False) -> GameState:
        """Finish a scored turn and advance to the next participant.

        ``player_only=True`` is retained as the PySide6 integration entry point.
        It now means "finish the PLAYER turn, play the AI turn, then prepare the
        next PLAYER round" instead of skipping the AI entirely.
        """
        self._require_active_game()
        if not self.state.turn_scored:
            raise RuntimeError("A category must be scored before ending the turn.")
        if self.state.game_over:
            return self.state

        if player_only:
            if self.state.current_player is not PlayerId.PLAYER:
                raise RuntimeError("Player-only turns require the PLAYER to be active.")
            self._finish_player_and_play_ai()
            return self.state

        self.state.current_player = (
            PlayerId.AI if self.state.current_player is PlayerId.PLAYER else PlayerId.PLAYER
        )
        self.state.turn += 1
        self.state.current_dice = None
        self.state.held_indices = frozenset()
        self.state.roll_count = 0
        self.state.turn_scored = False
        self.start_turn()
        return self.state

    def _finish_player_and_play_ai(self) -> None:
        """Run one complete AI turn immediately after the PLAYER scores."""
        from app.ai.expected_value_strategy import ExpectedValueStrategy
        from app.ai.action_generator import ActionType

        self.state.current_player = PlayerId.AI
        self.state.current_dice = None
        self.state.held_indices = frozenset()
        self.state.roll_count = 0
        self.state.turn_scored = False

        strategy = ExpectedValueStrategy()
        self.last_ai_action = None
        self.last_ai_reasoning = ""

        while self.state.roll_count < MAX_ROLLS_PER_TURN and not self.state.turn_scored:
            self.roll_dice()
            decision = strategy.decide(self.state)
            self.last_ai_action = decision.action
            self.last_ai_reasoning = decision.reasoning

            if decision.action.type is ActionType.SCORE:
                assert decision.action.selected_category is not None
                self.score_category(decision.action.selected_category)
                break

            desired_held = frozenset(decision.action.held_indices)
            current_held = self.state.held_indices
            for index in sorted(current_held - desired_held):
                self.unhold_dice(index)
            for index in sorted(desired_held - current_held):
                self.hold_dice(index)

        if not self.state.turn_scored:
            decision = strategy.decide(self.state)
            self.last_ai_action = decision.action
            self.last_ai_reasoning = decision.reasoning
            if decision.action.selected_category is None:
                raise RuntimeError("AI failed to choose a category after the final roll.")
            self.score_category(decision.action.selected_category)

        if self.state.game_over:
            return

        self.state.current_player = PlayerId.PLAYER
        self.state.turn += 1
        self.state.current_dice = None
        self.state.held_indices = frozenset()
        self.state.roll_count = 0
        self.state.turn_scored = False
        self.start_turn()

    def is_game_over(self) -> bool:
        """Whether the current game has reached its configured end condition."""
        return self.state.game_over

    def _require_active_game(self) -> None:
        if not self.state.game_started:
            raise RuntimeError("Start a game before taking an action.")
        if self.state.game_over:
            raise RuntimeError("The game is already over.")

    def _require_turn_in_progress(self) -> None:
        self._require_active_game()
        if self.state.turn_scored:
            raise RuntimeError("End the scored turn before taking another action.")

    def _require_turn_with_dice(self) -> None:
        self._require_turn_in_progress()
        if self.state.current_dice is None:
            raise RuntimeError("Roll dice before selecting or scoring a category.")

    @staticmethod
    def _validate_index(index: int) -> None:
        if type(index) is not int or not 0 <= index < DICE_COUNT:
            raise ValueError(f"Die index must be an integer from 0 through {DICE_COUNT - 1}.")
