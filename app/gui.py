"""PySide6 desktop prototype for the Yacht game.

Run with::

    python -m app.gui

The first UI milestone intentionally focuses on a complete human turn loop:
roll -> hold/unhold -> reroll -> select a category -> advance turn.
The existing core GameEngine remains the single source of truth for rules.
"""

from __future__ import annotations

import sys
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .core.categories import ALL_CATEGORIES, Category
from .core.game_engine import MAX_ROLLS_PER_TURN, GameEngine
from .core.game_state import PlayerId


APP_STYLE = """
QMainWindow, QWidget { background: #0b1424; color: #e8edf7; }
QFrame#card { background: #111e31; border: 1px solid #233653; border-radius: 12px; }
QFrame#playerCard { background: #122846; border: 1px solid #2367b7; border-radius: 12px; }
QFrame#aiCard { background: #281923; border: 1px solid #8d3342; border-radius: 12px; }
QLabel#title { font-size: 34px; font-weight: 800; color: #f4f7fb; }
QLabel#muted { color: #8797ad; }
QLabel#score { font-size: 25px; font-weight: 700; }
QLabel#section { font-size: 17px; font-weight: 700; }
QPushButton { background: #1a2a41; border: 1px solid #314966; border-radius: 8px; padding: 9px 12px; color: #e8edf7; }
QPushButton:hover { background: #223957; }
QPushButton:disabled { color: #65758b; background: #162234; border-color: #243247; }
QPushButton#roll { background: #245eb1; border-color: #3476d6; font-size: 16px; font-weight: 700; padding: 13px; }
QPushButton#roll:hover { background: #2c6dca; }
QPushButton#newGame { background: #1b304b; }
QPushButton#category { min-height: 48px; text-align: left; }
QPushButton#category[available="true"] { background: #172a43; border-color: #365a80; }
QPushButton#category[available="true"]:hover { background: #203c5d; border-color: #5c8ec5; }
QPushButton#die { background: #f3f5f8; color: #172033; border: 3px solid #56657a; border-radius: 10px; font-size: 28px; font-weight: 800; min-width: 72px; min-height: 72px; }
QPushButton#die:hover { border-color: #8da1ba; }
QPushButton#die[held="true"] { border: 4px solid #2e8cff; background: #e9f2ff; }
QLabel#held { color: #5aa0ff; font-weight: 700; }
QLabel#status { background: #0e1929; border: 1px solid #20324b; border-radius: 8px; padding: 10px; color: #b8c5d7; }
QLabel#log { color: #aab8ca; }
"""


DIE_GLYPHS = {
    1: "⚀",
    2: "⚁",
    3: "⚂",
    4: "⚃",
    5: "⚄",
    6: "⚅",
}

CATEGORY_SHORT = {
    Category.ONES: "Aces",
    Category.TWOS: "Twos",
    Category.THREES: "Threes",
    Category.FOURS: "Fours",
    Category.FIVES: "Fives",
    Category.SIXES: "Sixes",
    Category.CHOICE: "Choice",
    Category.FOUR_OF_A_KIND: "4 of a Kind",
    Category.FULL_HOUSE: "Full House",
    Category.SMALL_STRAIGHT: "Small Straight",
    Category.LARGE_STRAIGHT: "Large Straight",
    Category.YACHT: "Yacht",
}


class YachtWindow(QMainWindow):
    """Interactive prototype backed by the existing GameEngine."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = GameEngine()
        self.die_buttons: list[QPushButton] = []
        self.category_buttons: dict[Category, QPushButton] = {}
        self.score_labels: dict[tuple[PlayerId, Category], QLabel] = {}
        self.log_label: QLabel | None = None
        self.status_label: QLabel | None = None
        self.roll_label: QLabel | None = None
        self.turn_label: QLabel | None = None
        self.player_total: QLabel | None = None
        self.ai_total: QLabel | None = None
        self.roll_button: QPushButton | None = None
        self._build_ui()
        self.start_new_game()

    def _build_ui(self) -> None:
        self.setWindowTitle("Yacht Game Prototype")
        self.resize(1360, 900)
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(18, 16, 18, 12)
        main.setSpacing(12)

        header = QHBoxLayout()
        new_game = QPushButton("＋  새 게임")
        new_game.setObjectName("newGame")
        new_game.clicked.connect(self.start_new_game)
        header.addWidget(new_game)
        header.addStretch()
        title = QLabel("⚓  YACHT  ⚓")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        rules = QPushButton("⚙  규칙")
        rules.clicked.connect(self.show_rules)
        header.addWidget(rules)
        main.addLayout(header)

        players = QHBoxLayout()
        players.addWidget(self._player_header("PLAYER", "player"))
        center = QFrame()
        center.setObjectName("card")
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(18, 10, 18, 10)
        turn_caption = QLabel("턴")
        turn_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.turn_label = QLabel("1 / 12")
        self.turn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.turn_label.setStyleSheet("font-size: 22px; font-weight: 800;")
        center_l.addWidget(turn_caption)
        center_l.addWidget(self.turn_label)
        players.addWidget(center, 0)
        players.addWidget(self._player_header("AI (FastEV)", "ai"))
        main.addLayout(players)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_scorecard(), 3)
        body.addWidget(self._build_play_area(), 5)
        body.addWidget(self._build_ai_panel(), 3)
        main.addLayout(body, 1)

        footer = QHBoxLayout()
        footer.addWidget(QLabel("Yacht Game Prototype v0.1"))
        tip = QLabel("💡 팁: 주사위를 클릭하면 HOLD할 수 있어")
        tip.setObjectName("muted")
        footer.addStretch()
        footer.addWidget(tip)
        footer.addStretch()
        footer.addWidget(QLabel("사운드: ON"))
        main.addLayout(footer)

    def _player_header(self, name: str, side: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("playerCard" if side == "player" else "aiCard")
        layout = QHBoxLayout(frame)
        icon = QLabel("👤" if side == "player" else "🤖")
        name_label = QLabel(name)
        name_label.setStyleSheet(
            "font-size: 18px; font-weight: 800; color: #64a9ff;"
            if side == "player" else
            "font-size: 18px; font-weight: 800; color: #ff7f91;"
        )
        total = QLabel("0")
        total.setObjectName("score")
        if side == "player":
            self.player_total = total
        else:
            self.ai_total = total
        layout.addWidget(icon)
        layout.addWidget(name_label)
        layout.addStretch()
        layout.addWidget(QLabel("총점"))
        layout.addWidget(total)
        return frame

    def _build_scorecard(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("점수판")
        title.setObjectName("section")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        grid.addWidget(QLabel("카테고리"), 0, 0)
        grid.addWidget(QLabel("PLAYER"), 0, 1)
        grid.addWidget(QLabel("AI"), 0, 2)

        for row, category in enumerate(ALL_CATEGORIES, 1):
            grid.addWidget(QLabel(CATEGORY_SHORT[category]), row, 0)
            for col, player in ((1, PlayerId.PLAYER), (2, PlayerId.AI)):
                label = QLabel("-")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.score_labels[(player, category)] = label
                grid.addWidget(label, row, col)

        bonus_row = len(ALL_CATEGORIES) + 1
        grid.addWidget(QLabel("상단 합계 / 보너스"), bonus_row, 0)
        self.player_upper_label = QLabel("0 / -")
        self.ai_upper_label = QLabel("0 / -")
        grid.addWidget(self.player_upper_label, bonus_row, 1)
        grid.addWidget(self.ai_upper_label, bonus_row, 2)
        layout.addLayout(grid)
        return card

    def _build_play_area(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        caption = QLabel("남은 주사위 굴리기")
        caption.setObjectName("section")
        layout.addWidget(caption)
        self.roll_label = QLabel("0 / 3")
        self.roll_label.setObjectName("muted")
        layout.addWidget(self.roll_label)

        dice_row = QHBoxLayout()
        dice_row.setSpacing(10)
        for index in range(5):
            button = QPushButton("-")
            button.setObjectName("die")
            button.setProperty("held", False)
            button.clicked.connect(lambda _checked=False, i=index: self.toggle_hold(i))
            dice_row.addWidget(button, 1)
            self.die_buttons.append(button)
        layout.addLayout(dice_row)

        self.status_label = QLabel("새 게임을 시작했어")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.roll_button = QPushButton("🎲  주사위 굴리기")
        self.roll_button.setObjectName("roll")
        self.roll_button.clicked.connect(self.roll_dice)
        layout.addWidget(self.roll_button)

        cat_title = QLabel("카테고리 선택")
        cat_title.setObjectName("section")
        layout.addWidget(cat_title)
        cat_grid = QGridLayout()
        for i, category in enumerate(ALL_CATEGORIES):
            button = QPushButton(CATEGORY_SHORT[category])
            button.setObjectName("category")
            button.setProperty("available", False)
            button.clicked.connect(lambda _checked=False, c=category: self.score_category(c))
            self.category_buttons[category] = button
            cat_grid.addWidget(button, i // 3, i % 3)
        layout.addLayout(cat_grid)
        return card

    def _build_ai_panel(self) -> QFrame:
        outer = QFrame()
        outer.setObjectName("card")
        layout = QVBoxLayout(outer)
        title = QLabel("AI 진행 상황")
        title.setObjectName("section")
        layout.addWidget(title)
        ai_status = QLabel("AI (FastEV)\n\n게임 엔진 연결 준비 완료\n\n현재는 PLAYER 프로토타입 단계야")
        ai_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ai_status.setObjectName("status")
        layout.addWidget(ai_status, 1)
        log_title = QLabel("게임 로그")
        log_title.setObjectName("section")
        layout.addWidget(log_title)
        self.log_label = QLabel("게임 시작!\nPLAYER 턴이야")
        self.log_label.setObjectName("log")
        self.log_label.setWordWrap(True)
        layout.addWidget(self.log_label)
        return outer

    def start_new_game(self) -> None:
        self.engine.start_game()
        self._refresh()
        self._set_status("PLAYER 턴이야 · 주사위를 굴려줘")

    def roll_dice(self) -> None:
        try:
            dice = self.engine.roll_dice()
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"굴림 {self.engine.state.roll_count}/{MAX_ROLLS_PER_TURN} · 주사위를 눌러 HOLD할 수 있어")
        self._refresh(dice)

    def toggle_hold(self, index: int) -> None:
        if self.engine.state.current_dice is None:
            return
        try:
            if index in self.engine.state.held_indices:
                self.engine.unhold_dice(index)
            else:
                self.engine.hold_dice(index)
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc))
            return
        self._refresh()

    def score_category(self, category: Category) -> None:
        try:
            score = self.engine.score_category(category)
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc))
            return

        if self.engine.is_game_over():
            self._refresh()
            QMessageBox.information(self, "게임 종료", f"게임이 끝났어!\nPLAYER {self.engine.state.player_score}점")
            return

        self._set_status(f"{CATEGORY_SHORT[category]}에 {score}점 기록 · 다음은 AI 턴이야")
        self.engine.end_turn()
        # The prototype keeps the AI side as a placeholder until its turn runner is wired in.
        self._refresh()
        self._set_status("AI 턴은 다음 UI 단계에서 연결할 예정이야 · 새 게임으로 계속 테스트할 수 있어")

    def _refresh(self, dice: Iterable[int] | None = None) -> None:
        state = self.engine.state
        dice = tuple(dice) if dice is not None else state.current_dice
        for i, button in enumerate(self.die_buttons):
            value = dice[i] if dice and i < len(dice) else None
            button.setText(DIE_GLYPHS.get(value, "-") if value else "-")
            button.setProperty("held", i in state.held_indices)
            button.style().unpolish(button)
            button.style().polish(button)
            button.setEnabled(value is not None and state.current_player is PlayerId.PLAYER and not state.turn_scored)

        self.roll_label.setText(f"{state.roll_count} / {MAX_ROLLS_PER_TURN}")
        self.roll_button.setEnabled(
            state.current_player is PlayerId.PLAYER
            and not state.turn_scored
            and state.roll_count < MAX_ROLLS_PER_TURN
        )

        available = set(self.engine.get_available_categories()) if state.current_dice else set()
        scores = self.engine.get_current_scores() if state.current_dice else {}
        for category, button in self.category_buttons.items():
            is_available = category in available
            button.setProperty("available", is_available)
            button.setText(
                f"{CATEGORY_SHORT[category]}\n{scores[category]}점"
                if is_available else
                f"{CATEGORY_SHORT[category]}\n사용됨"
            )
            button.setEnabled(is_available)
            button.style().unpolish(button)
            button.style().polish(button)

        for category in ALL_CATEGORIES:
            for player in (PlayerId.PLAYER, PlayerId.AI):
                score = state.players[player].category_scores.get(category)
                self.score_labels[(player, category)].setText("-" if score is None else str(score))

        self.player_total.setText(str(state.player_score))
        self.ai_total.setText(str(state.ai_score))
        self.player_upper_label.setText(f"{state.player_upper_total} / {'35' if state.player_has_bonus else '-'}")
        self.ai_upper_label.setText(f"{state.ai_upper_total} / {'35' if state.ai_has_bonus else '-'}")
        # One game has 12 category rounds; each round contains two participant turns.
        round_no = min(12, (state.turn + 1) // 2)
        self.turn_label.setText(f"{round_no} / 12")

    def _set_status(self, text: str) -> None:
        if self.status_label:
            self.status_label.setText(text)
        if self.log_label:
            self.log_label.setText(text)

    def show_rules(self) -> None:
        QMessageBox.information(
            self,
            "Yacht 규칙",
            "• 플레이어마다 12개 카테고리를 한 번씩 사용해\n"
            "• 한 턴에 최대 3번 굴릴 수 있어\n"
            "• 굴린 주사위를 클릭하면 HOLD할 수 있어\n"
            "• 마지막 굴림 후 사용하지 않은 카테고리 하나를 선택해 점수를 기록해\n"
            "• 상단 6개 합계가 63점 이상이면 35점 보너스가 붙어\n"
            "• 모든 카테고리를 사용하면 게임이 끝나",
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Yacht Game Prototype")
    app.setFont(QFont("Segoe UI", 10))
    window = YachtWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
