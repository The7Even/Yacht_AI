from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 34px; font-weight: 800; color: #f4f7fb; }
QLabel#muted { color: #9aaac0; font-size: 12px; }
QLabel#score { font-size: 27px; font-weight: 800; }
QLabel#section { font-size: 15px; font-weight: 800; }
QLabel#subsection { font-size: 12px; font-weight: 800; color: #b7c6d9; padding-top: 0px; }
QLabel#scorecardTitle { font-size: 12px; font-weight: 800; color: #e8edf7; }
QLabel#scoreGap { font-size: 14px; font-weight: 700; color: #8fbfff; }
QLabel#scoreGapNegative { font-size: 14px; font-weight: 700; color: #ff9aaa; }
QLabel#playerHeader { color: #64a9ff; font-size: 11px; font-weight: 800; }
QLabel#aiHeader { color: #ff7f91; font-size: 11px; font-weight: 800; }
QLabel#playerCell { background: #14375f; border: 1px solid #285f98; border-radius: 5px; padding: 4px 2px; color: #78b5ff; font-size: 13px; font-weight: 800; min-height: 22px; }
QLabel#aiCell { background: #4a2731; border: 1px solid #8b4654; border-radius: 5px; padding: 4px 2px; color: #ff9aaa; font-size: 13px; font-weight: 800; min-height: 22px; }
QPushButton { background: #1a2a41; border: 1px solid #314966; border-radius: 8px; padding: 9px 12px; color: #e8edf7; font-size: 12px; }
QPushButton:hover { background: #223957; }
QPushButton:disabled { color: #65758b; background: #162234; border-color: #243247; }
QPushButton#roll { background: #245eb1; border-color: #3476d6; font-size: 16px; font-weight: 800; padding: 13px; }
QPushButton#roll:hover { background: #2c6dca; }
QPushButton#newGame { background: #1b304b; }
QPushButton#category { min-height: 48px; text-align: left; padding: 5px 8px; font-size: 12px; }
QPushButton#category[special="true"] { min-height: 62px; }
QPushButton#category[available="true"] { background: #172a43; border-color: #365a80; }
QPushButton#category[available="true"]:hover { background: #203c5d; border-color: #5c8ec5; }
QPushButton#die { background: #f3f5f8; color: #172033; border: 3px solid #56657a; border-radius: 12px; min-width: 108px; min-height: 108px; padding: 4px; }
QPushButton#die:hover { border-color: #8da1ba; }
QPushButton#die[held="true"] { border: 4px solid #2e8cff; background: #e9f2ff; }
QLabel#rollRemaining { background: #1d4f8f; border: 1px solid #3d83db; border-radius: 15px; padding: 5px 12px; color: #dcebff; font-size: 12px; font-weight: 800; }
QLabel#status { background: #0e1929; border: 1px solid #20324b; border-radius: 8px; padding: 10px; color: #b8c5d7; }
QLabel#log { color: #b5c2d3; font-size: 12px; }
QFrame#scoreRow { background: #0e1929; border: 1px solid #1f3048; border-radius: 6px; }
"""


CATEGORY_SHORT = {
    Category.ONES: "Ones",
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
UPPER_CATEGORIES = (
    Category.ONES,
    Category.TWOS,
    Category.THREES,
    Category.FOURS,
    Category.FIVES,
    Category.SIXES,
)
SPECIAL_CATEGORIES = tuple(c for c in ALL_CATEGORIES if c not in UPPER_CATEGORIES)


class YachtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.engine = GameEngine()
        self.die_buttons: list[QPushButton] = []
        self.category_buttons: dict[Category, QPushButton] = {}
        self.score_labels: dict[tuple[PlayerId, Category], QLabel] = {}

        self.log_label = None
        self.status_label = None
        self.roll_label = None
        self.turn_label = None
        self.player_total = None
        self.ai_total = None
        self.roll_button = None
        self.player_upper_label = None
        self.ai_upper_label = None
        self.player_bonus_summary = None
        self.ai_bonus_summary = None
        self.player_table_total = None
        self.ai_table_total = None
        self.center_total_label = None
        self.gap_label = None

        self.dice_icons = self._load_dice_icons()
        self.category_icons = self._load_category_icons()

        self._build_ui()
        self.start_new_game()

    def _load_dice_icons(self) -> dict[int, QIcon]:
        icons = {}
        asset_dir = Path(__file__).resolve().parent / "assets" / "dice"
        for value in range(1, 7):
            path = asset_dir / f"dice_{value}.svg"
            if path.exists():
                icons[value] = QIcon(str(path))
        return icons

    def _load_category_icons(self) -> dict[Category, QIcon]:
        icons = {}
        asset_dir = Path(__file__).resolve().parent / "assets" / "categories"
        names = {
            Category.CHOICE: "choice",
            Category.FOUR_OF_A_KIND: "four_of_a_kind",
            Category.FULL_HOUSE: "full_house",
            Category.SMALL_STRAIGHT: "small_straight",
            Category.LARGE_STRAIGHT: "large_straight",
            Category.YACHT: "yacht",
        }
        for category, name in names.items():
            path = asset_dir / f"{name}.svg"
            if path.exists():
                icons[category] = QIcon(str(path))
        return icons

    def _build_ui(self) -> None:
        self.setWindowTitle("Yacht Game Prototype")
        self.resize(1360, 900)
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(18, 16, 18, 12)
        main.setSpacing(10)

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
        players.setSpacing(10)
        players.addWidget(self._player_header("PLAYER", "player"))
        players.addWidget(self._build_center_summary(), 0)
        players.addWidget(self._player_header("AI (FastEV)", "ai"))
        main.addLayout(players)

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_scorecard(), 5)
        body.addWidget(self._build_play_area(), 14)
        body.addWidget(self._build_ai_panel(), 5)
        main.addLayout(body, 1)

        footer = QHBoxLayout()
        footer.addWidget(QLabel("Yacht Game Prototype v0.5"))
        tip = QLabel("💡 주사위를 클릭하면 HOLD할 수 있습니다.")
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
        layout.setContentsMargins(14, 10, 14, 10)

        layout.addWidget(QLabel("👤" if side == "player" else "🤖"))

        name_label = QLabel(name)
        name_label.setStyleSheet(
            "font-size:18px;font-weight:800;color:#64a9ff;background:transparent;"
            if side == "player"
            else "font-size:18px;font-weight:800;color:#ff7f91;background:transparent;"
        )

        total = QLabel("0")
        total.setObjectName("score")
        total.setStyleSheet(
            "color:#64a9ff;background:transparent;"
            if side == "player"
            else "color:#ff7f91;background:transparent;"
        )

        if side == "player":
            self.player_total = total
        else:
            self.ai_total = total

        layout.addWidget(name_label)
        layout.addStretch()
        layout.addWidget(QLabel("총점"))
        layout.addWidget(total)
        return frame

    def _build_center_summary(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 7, 14, 7)

        caption = QLabel("현재 경기")
        caption.setObjectName("muted")
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(caption)

        self.center_total_label = QLabel("0 : 0")
        self.center_total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.center_total_label.setStyleSheet("font-size:20px;font-weight:800;")
        layout.addWidget(self.center_total_label)

        self.gap_label = QLabel("점수 차이 0점 · 동점")
        self.gap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gap_label)

        self.turn_label = QLabel("턴 1 / 12")
        self.turn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.turn_label.setObjectName("muted")
        layout.addWidget(self.turn_label)
        return frame

    def _configure_score_columns(self, grid: QGridLayout) -> None:
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

    def _build_scorecard(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(7, 2, 7, 6)
        layout.setSpacing(3)

        header = QGridLayout()
        header.setContentsMargins(5, 0, 5, 0)
        header.setHorizontalSpacing(4)
        header.setVerticalSpacing(0)

        title = QLabel("점수판")
        title.setObjectName("scorecardTitle")
        ph = QLabel("PLAYER")
        ph.setObjectName("playerHeader")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ah = QLabel("AI")
        ah.setObjectName("aiHeader")
        ah.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title, 0, 0)
        header.addWidget(ph, 0, 1)
        header.addWidget(ah, 0, 2)
        self._configure_score_columns(header)
        layout.addLayout(header)

        layout.addWidget(self._score_section("기본 점수", UPPER_CATEGORIES))

        upper_summary = QFrame()
        upper_summary.setObjectName("scoreRow")
        summary = QGridLayout(upper_summary)
        summary.setContentsMargins(5, 3, 5, 3)
        summary.setHorizontalSpacing(4)
        summary.setVerticalSpacing(3)
        summary.addWidget(QLabel("상단 합계"), 0, 0)

        self.player_upper_label = QLabel("0 / 63")
        self.player_upper_label.setObjectName("playerCell")
        self.player_upper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ai_upper_label = QLabel("0 / 63")
        self.ai_upper_label.setObjectName("aiCell")
        self.ai_upper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary.addWidget(self.player_upper_label, 0, 1)
        summary.addWidget(self.ai_upper_label, 0, 2)

        summary.addWidget(QLabel("보너스"), 1, 0)
        self.player_bonus_summary = QLabel("-")
        self.player_bonus_summary.setObjectName("playerCell")
        self.player_bonus_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ai_bonus_summary = QLabel("-")
        self.ai_bonus_summary.setObjectName("aiCell")
        self.ai_bonus_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        summary.addWidget(self.player_bonus_summary, 1, 1)
        summary.addWidget(self.ai_bonus_summary, 1, 2)
        self._configure_score_columns(summary)
        layout.addWidget(upper_summary)

        layout.addWidget(self._score_section("특수 족보", SPECIAL_CATEGORIES))

        total_row = QFrame()
        total_row.setObjectName("scoreRow")
        total = QGridLayout(total_row)
        total.setContentsMargins(5, 3, 5, 3)
        total.setHorizontalSpacing(4)
        total.addWidget(QLabel("총합"), 0, 0)

        self.player_table_total = QLabel("0")
        self.player_table_total.setObjectName("playerCell")
        self.player_table_total.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ai_table_total = QLabel("0")
        self.ai_table_total.setObjectName("aiCell")
        self.ai_table_total.setAlignment(Qt.AlignmentFlag.AlignCenter)
        total.addWidget(self.player_table_total, 0, 1)
        total.addWidget(self.ai_table_total, 0, 2)
        self._configure_score_columns(total)
        layout.addWidget(total_row)
        return card

    def _score_section(self, title: str, categories: Iterable[Category]) -> QFrame:
        wrap = QFrame()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        subsection = QLabel(title)
        subsection.setObjectName("subsection")
        layout.addWidget(subsection)

        grid = QGridLayout()
        grid.setContentsMargins(5, 0, 5, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)
        self._configure_score_columns(grid)

        for row, category in enumerate(categories):
            name = QLabel(CATEGORY_SHORT[category])
            name.setStyleSheet("font-size:13px;")
            grid.addWidget(name, row, 0)

            for column, player in ((1, PlayerId.PLAYER), (2, PlayerId.AI)):
                value = QLabel("-")
                value.setObjectName(
                    "playerCell" if player is PlayerId.PLAYER else "aiCell"
                )
                value.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.score_labels[player, category] = value
                grid.addWidget(value, row, column)

        layout.addLayout(grid)
        return wrap
