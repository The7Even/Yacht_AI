from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .ai.expected_value_strategy import ExpectedValueStrategy
from .ai.action_generator import ActionType
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
QLabel#subsection { font-size: 12px; font-weight: 800; color: #b7c6d9; }
QLabel#scorecardTitle { font-size: 13px; font-weight: 800; color: #e8edf7; }
QLabel#scoreGap { font-size: 14px; font-weight: 700; color: #8fbfff; }
QLabel#scoreGapNegative { font-size: 14px; font-weight: 700; color: #ff9aaa; }
QLabel#playerHeader { color: #64a9ff; font-size: 12px; font-weight: 800; }
QLabel#aiHeader { color: #ff7f91; font-size: 12px; font-weight: 800; }
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
QLabel#aiStatus { background: #0e1929; border: 1px solid #20324b; border-radius: 8px; padding: 12px; color: #d4deeb; font-size: 12px; }
QFrame#scoreRow { background: #0e1929; border: 1px solid #1f3048; border-radius: 6px; }
"""

CATEGORY_SHORT = {Category.ONES: "Ones", Category.TWOS: "Twos", Category.THREES: "Threes", Category.FOURS: "Fours", Category.FIVES: "Fives", Category.SIXES: "Sixes", Category.CHOICE: "Choice", Category.FOUR_OF_A_KIND: "4 of a Kind", Category.FULL_HOUSE: "Full House", Category.SMALL_STRAIGHT: "Small Straight", Category.LARGE_STRAIGHT: "Large Straight", Category.YACHT: "Yacht"}
UPPER_CATEGORIES = (Category.ONES, Category.TWOS, Category.THREES, Category.FOURS, Category.FIVES, Category.SIXES)
SPECIAL_CATEGORIES = tuple(c for c in ALL_CATEGORIES if c not in UPPER_CATEGORIES)


class YachtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.engine = GameEngine()
        self.ai_strategy = ExpectedValueStrategy()
        self.ai_active = False
        self.ai_status_label = None
        self.ai_timer = QTimer(self)
        self.ai_timer.setSingleShot(True)
        self.ai_timer.timeout.connect(self._run_ai_decision)
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
        icons: dict[int, QIcon] = {}
        asset_dir = Path(__file__).resolve().parent / "assets" / "dice"
        for value in range(1, 7):
            path = asset_dir / f"dice_{value}.svg"
            if path.exists(): icons[value] = QIcon(str(path))
        return icons

    def _load_category_icons(self) -> dict[Category, QIcon]:
        icons: dict[Category, QIcon] = {}
        asset_dir = Path(__file__).resolve().parent / "assets" / "categories"
        names = {Category.CHOICE: "choice", Category.FOUR_OF_A_KIND: "four_of_a_kind", Category.FULL_HOUSE: "full_house", Category.SMALL_STRAIGHT: "small_straight", Category.LARGE_STRAIGHT: "large_straight", Category.YACHT: "yacht"}
        for category, name in names.items():
            path = asset_dir / f"{name}.svg"
            if path.exists(): icons[category] = QIcon(str(path))
        return icons

    def _build_ui(self) -> None:
        self.setWindowTitle("Yacht Game Prototype")
        self.resize(1360, 900)
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(APP_STYLE)
        root = QWidget(); self.setCentralWidget(root)
        main = QVBoxLayout(root); main.setContentsMargins(18, 16, 18, 12); main.setSpacing(10)
        header = QHBoxLayout(); new_game = QPushButton("＋  새 게임"); new_game.setObjectName("newGame"); new_game.clicked.connect(self.start_new_game); header.addWidget(new_game); header.addStretch(); title = QLabel("⚓  YACHT  ⚓"); title.setObjectName("title"); header.addWidget(title); header.addStretch(); rules = QPushButton("⚙  규칙"); rules.clicked.connect(self.show_rules); header.addWidget(rules); main.addLayout(header)
        players = QHBoxLayout(); players.setSpacing(10); players.addWidget(self._player_header("PLAYER", "player")); players.addWidget(self._build_center_summary(), 0); players.addWidget(self._player_header("AI (FastEV)", "ai")); main.addLayout(players)
        body = QHBoxLayout(); body.setSpacing(10); body.addWidget(self._build_scorecard(), 5); body.addWidget(self._build_play_area(), 14); body.addWidget(self._build_ai_panel(), 5); main.addLayout(body, 1)
        footer = QHBoxLayout(); footer.addWidget(QLabel("Yacht Game Prototype v0.5")); tip = QLabel("💡 주사위를 클릭하면 HOLD할 수 있습니다."); tip.setObjectName("muted"); footer.addStretch(); footer.addWidget(tip); footer.addStretch(); footer.addWidget(QLabel("사운드: ON")); main.addLayout(footer)

    def _player_header(self, name: str, side: str) -> QFrame:
        frame = QFrame(); frame.setObjectName("playerCard" if side == "player" else "aiCard")
        layout = QHBoxLayout(frame); layout.setContentsMargins(14, 10, 14, 10); layout.addWidget(QLabel("👤" if side == "player" else "🤖"))
        name_label = QLabel(name); name_label.setStyleSheet("font-size:18px;font-weight:800;color:#64a9ff;background:transparent;" if side == "player" else "font-size:18px;font-weight:800;color:#ff7f91;background:transparent;")
        total = QLabel("0"); total.setObjectName("score"); total.setStyleSheet("color:#64a9ff;background:transparent;" if side == "player" else "color:#ff7f91;background:transparent;")
        if side == "player": self.player_total = total
        else: self.ai_total = total
        layout.addWidget(name_label); layout.addStretch(); layout.addWidget(QLabel("총점")); layout.addWidget(total)
        return frame

    def _build_center_summary(self) -> QFrame:
        frame = QFrame(); frame.setObjectName("card"); layout = QVBoxLayout(frame); layout.setContentsMargins(14, 7, 14, 7)
        caption = QLabel("현재 경기"); caption.setObjectName("muted"); caption.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(caption)
        self.center_total_label = QLabel("0 : 0"); self.center_total_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.center_total_label.setStyleSheet("font-size:20px;font-weight:800;"); layout.addWidget(self.center_total_label)
        self.gap_label = QLabel("점수 차이 0점 · 동점"); self.gap_label.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(self.gap_label)
        self.turn_label = QLabel("턴 1 / 12"); self.turn_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.turn_label.setObjectName("muted"); layout.addWidget(self.turn_label)
        return frame

    @staticmethod
    def _configure_score_columns(grid: QGridLayout) -> None:
        grid.setColumnStretch(0, 2); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)

    def _build_scorecard(self) -> QFrame:
        card = QFrame(); card.setObjectName("card"); layout = QVBoxLayout(card); layout.setContentsMargins(7, 2, 7, 6); layout.setSpacing(3)
        header = QGridLayout(); header.setContentsMargins(5, 0, 5, 0); header.setHorizontalSpacing(4); title = QLabel("점수판"); title.setObjectName("scorecardTitle"); ph = QLabel("PLAYER"); ph.setObjectName("playerHeader"); ph.setAlignment(Qt.AlignmentFlag.AlignCenter); ah = QLabel("AI"); ah.setObjectName("aiHeader"); ah.setAlignment(Qt.AlignmentFlag.AlignCenter); header.addWidget(title, 0, 0); header.addWidget(ph, 0, 1); header.addWidget(ah, 0, 2); self._configure_score_columns(header); layout.addLayout(header); layout.addWidget(self._score_section("기본 점수", UPPER_CATEGORIES))
        upper_summary = QFrame(); upper_summary.setObjectName("scoreRow"); summary = QGridLayout(upper_summary); summary.setContentsMargins(5, 3, 5, 3); summary.setHorizontalSpacing(4); summary.setVerticalSpacing(3); summary.addWidget(QLabel("상단 합계"), 0, 0)
        self.player_upper_label = QLabel("0 / 63"); self.player_upper_label.setObjectName("playerCell"); self.player_upper_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.ai_upper_label = QLabel("0 / 63"); self.ai_upper_label.setObjectName("aiCell"); self.ai_upper_label.setAlignment(Qt.AlignmentFlag.AlignCenter); summary.addWidget(self.player_upper_label, 0, 1); summary.addWidget(self.ai_upper_label, 0, 2); summary.addWidget(QLabel("보너스"), 1, 0)
        self.player_bonus_summary = QLabel("-"); self.player_bonus_summary.setObjectName("playerCell"); self.player_bonus_summary.setAlignment(Qt.AlignmentFlag.AlignCenter); self.ai_bonus_summary = QLabel("-"); self.ai_bonus_summary.setObjectName("aiCell"); self.ai_bonus_summary.setAlignment(Qt.AlignmentFlag.AlignCenter); summary.addWidget(self.player_bonus_summary, 1, 1); summary.addWidget(self.ai_bonus_summary, 1, 2); self._configure_score_columns(summary); layout.addWidget(upper_summary); layout.addWidget(self._score_section("특수 족보", SPECIAL_CATEGORIES))
        total_row = QFrame(); total_row.setObjectName("scoreRow"); total = QGridLayout(total_row); total.setContentsMargins(5, 3, 5, 3); total.setHorizontalSpacing(4); total.addWidget(QLabel("총합"), 0, 0); self.player_table_total = QLabel("0"); self.player_table_total.setObjectName("playerCell"); self.player_table_total.setAlignment(Qt.AlignmentFlag.AlignCenter); self.ai_table_total = QLabel("0"); self.ai_table_total.setObjectName("aiCell"); self.ai_table_total.setAlignment(Qt.AlignmentFlag.AlignCenter); total.addWidget(self.player_table_total, 0, 1); total.addWidget(self.ai_table_total, 0, 2); self._configure_score_columns(total); layout.addWidget(total_row)
        return card

    def _score_section(self, title: str, categories: Iterable[Category]) -> QFrame:
        wrap = QFrame(); layout = QVBoxLayout(wrap); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(1); subsection = QLabel(title); subsection.setObjectName("subsection"); layout.addWidget(subsection); grid = QGridLayout(); grid.setContentsMargins(5, 0, 5, 0); grid.setHorizontalSpacing(4); grid.setVerticalSpacing(3); self._configure_score_columns(grid)
        for row, category in enumerate(categories):
            name = QLabel(CATEGORY_SHORT[category]); name.setStyleSheet("font-size:13px;"); grid.addWidget(name, row, 0)
            for column, player in ((1, PlayerId.PLAYER), (2, PlayerId.AI)):
                value = QLabel("-"); value.setObjectName("playerCell" if player is PlayerId.PLAYER else "aiCell"); value.setAlignment(Qt.AlignmentFlag.AlignCenter); self.score_labels[player, category] = value; grid.addWidget(value, row, column)
        layout.addLayout(grid); return wrap

    def _build_play_area(self) -> QFrame:
        card = QFrame(); card.setObjectName("card"); layout = QVBoxLayout(card); layout.setContentsMargins(14, 9, 14, 9); layout.setSpacing(6); top = QHBoxLayout(); caption = QLabel("주사위 굴리기"); caption.setObjectName("section"); top.addWidget(caption); top.addStretch(); self.roll_label = QLabel("3회 남음"); self.roll_label.setObjectName("rollRemaining"); top.addWidget(self.roll_label); layout.addLayout(top)
        dice = QHBoxLayout(); dice.setSpacing(10)
        for index in range(5):
            button = QPushButton("-"); button.setObjectName("die"); button.clicked.connect(lambda _=False, i=index: self.toggle_hold(i)); self.die_buttons.append(button); dice.addWidget(button, 1)
        layout.addLayout(dice)
        self.status_label = QLabel("0/3회 굴렸습니다. 주사위를 클릭하면 HOLD할 수 있습니다."); self.status_label.setObjectName("status"); self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(self.status_label)
        self.roll_button = QPushButton("🎲  주사위 굴리기"); self.roll_button.setObjectName("roll"); self.roll_button.clicked.connect(self.roll_dice); layout.addWidget(self.roll_button)
        section = QLabel("카테고리 선택"); section.setObjectName("section"); layout.addWidget(section)
        sub = QLabel("기본 점수 (Ones ~ Sixes)"); sub.setObjectName("subsection"); layout.addWidget(sub)
        upper_grid = QGridLayout(); upper_grid.setHorizontalSpacing(5); upper_grid.setVerticalSpacing(5)
        for index, category in enumerate(UPPER_CATEGORIES):
            button = QPushButton(); button.setObjectName("category"); button.setIcon(self.dice_icons.get(index + 1, QIcon())); button.setIconSize(QSize(34, 34)); button.setProperty("available", False); button.clicked.connect(lambda _=False, cat=category: self.score_category(cat)); self.category_buttons[category] = button; upper_grid.addWidget(button, index // 3, index % 3)
        layout.addLayout(upper_grid)
        sub2 = QLabel("특수 족보"); sub2.setObjectName("subsection"); layout.addWidget(sub2)
        special_grid = QGridLayout(); special_grid.setHorizontalSpacing(5); special_grid.setVerticalSpacing(5)
        for index, category in enumerate(SPECIAL_CATEGORIES):
            button = QPushButton(); button.setObjectName("category"); button.setProperty("special", True); button.setIcon(self.category_icons.get(category, QIcon())); button.setIconSize(QSize(72, 42)); button.setProperty("available", False); button.clicked.connect(lambda _=False, cat=category: self.score_category(cat)); self.category_buttons[category] = button; special_grid.addWidget(button, index // 3, index % 3)
        layout.addLayout(special_grid); return card

    def _build_ai_panel(self) -> QFrame:
        outer = QFrame(); outer.setObjectName("card"); layout = QVBoxLayout(outer); layout.setContentsMargins(10, 9, 10, 9); title = QLabel("AI 진행 상황"); title.setObjectName("section"); layout.addWidget(title); self.ai_status_label = QLabel("AI (FastEV)\n\n게임 엔진 연결 준비가 완료되었습니다.\n\nPLAYER 턴을 기다리는 중입니다."); self.ai_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.ai_status_label.setObjectName("aiStatus"); self.ai_status_label.setWordWrap(True); layout.addWidget(self.ai_status_label, 1); log_title = QLabel("게임 로그"); log_title.setObjectName("section"); layout.addWidget(log_title); self.log_label = QLabel("게임이 시작되었습니다.\nPLAYER 턴입니다."); self.log_label.setObjectName("log"); self.log_label.setWordWrap(True); layout.addWidget(self.log_label); return outer

    def start_new_game(self) -> None:
        self.ai_timer.stop(); self.ai_active = False; self.engine.start_game(); self._refresh(); self._set_status("0/3회 굴렸습니다. 주사위를 클릭하면 HOLD할 수 있습니다."); self._set_ai_status("AI (FastEV)\n\nPLAYER 턴을 기다리는 중입니다.")

    def roll_dice(self) -> None:
        if self.ai_active: return
        try: dice = self.engine.roll_dice()
        except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); return
        self._refresh(dice)
        self._set_status("3/3회 굴렸습니다. 아래에서 기록할 점수 항목을 선택해 주세요." if self.engine.state.roll_count >= MAX_ROLLS_PER_TURN else f"{self.engine.state.roll_count}/{MAX_ROLLS_PER_TURN}회 굴렸습니다. 주사위를 클릭하면 HOLD할 수 있습니다.")

    def toggle_hold(self, index: int) -> None:
        if self.ai_active or self.engine.state.current_dice is None: return
        try:
            if index in self.engine.state.held_indices: self.engine.unhold_dice(index)
            else: self.engine.hold_dice(index)
        except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); return
        self._refresh()

    def score_category(self, category: Category) -> None:
        if self.ai_active: return
        try: score = self.engine.score_category(category)
        except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); return
        self._refresh()
        if self.engine.is_game_over(): self._finish_game(score, category); return
        try: self.engine.begin_ai_turn()
        except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); return
        self.ai_active = True; self._refresh(); self._set_status(f"{CATEGORY_SHORT[category]}에 {score}점을 기록했습니다. AI 턴을 시작합니다."); self._set_ai_status("AI (FastEV)\n\nAI 턴을 준비하고 있습니다..."); self.ai_timer.start(500)

    def _run_ai_decision(self) -> None:
        if not self.ai_active: return
        state = self.engine.state
        if state.current_player is not PlayerId.AI or state.turn_scored:
            self._finish_ai_turn(); return
        if state.current_dice is None:
            try: dice = self.engine.roll_dice()
            except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); self._set_ai_status(f"AI (FastEV)\n\n오류: {exc}"); self._finish_ai_turn(); return
            self._refresh(dice); roll_no = state.roll_count; self._set_status(f"AI가 {roll_no}/3회 주사위를 굴렸습니다."); self._set_ai_status(f"AI (FastEV)\n\n{roll_no}/3회 굴림 완료\n\n현재 주사위 조합을 분석하는 중입니다..."); self.ai_timer.start(450); return

        try: decision = self.ai_strategy.decide(state)
        except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); self._set_ai_status(f"AI (FastEV)\n\n분석 오류: {exc}"); self._finish_ai_turn(); return
        self.engine.last_ai_action = decision.action; self.engine.last_ai_reasoning = decision.reasoning
        if decision.action.type is ActionType.SCORE:
            category = decision.action.selected_category; assert category is not None; score = self.engine.score_category(category); self._refresh(); self._set_status(f"AI가 {CATEGORY_SHORT[category]}을(를) 선택해 {score}점을 기록합니다."); self._set_ai_status(f"AI (FastEV)\n\n기록 선택\n{CATEGORY_SHORT[category]} · {score}점\n\n{decision.reasoning}"); self.ai_timer.start(800); return

        desired = frozenset(decision.action.held_indices); current = self.engine.state.held_indices
        for index in sorted(current - desired): self.engine.unhold_dice(index)
        for index in sorted(desired - current): self.engine.hold_dice(index)
        self._refresh(); held_text = ", ".join(str(i + 1) for i in sorted(desired)) if desired else "없음"; self._set_status(f"AI가 {len(desired)}개 주사위를 HOLD했습니다. 다음 굴림을 준비합니다."); self._set_ai_status(f"AI (FastEV)\n\n{state.roll_count}/3회 굴림\nHOLD: {held_text}\n\n{decision.reasoning}"); self.ai_timer.start(650)

    def _finish_ai_turn(self) -> None:
        if not self.ai_active: return
        if self.engine.state.turn_scored:
            try: self.engine.finish_ai_turn()
            except (RuntimeError, ValueError) as exc: self._set_status(str(exc)); self._set_ai_status(f"AI (FastEV)\n\n턴 종료 오류: {exc}"); self.ai_active = False; return
        self.ai_active = False; self._refresh()
        if self.engine.is_game_over(): self._show_final_result(); return
        self._set_status(f"AI 턴이 끝났습니다. 다음은 PLAYER {self.engine.state.turn}턴입니다."); self._set_ai_status("AI (FastEV)\n\nAI 턴 완료\n\nPLAYER의 다음 턴을 기다리는 중입니다.")

    def _finish_game(self, score: int, category: Category) -> None:
        self._set_status(f"{CATEGORY_SHORT[category]}에 {score}점을 기록했습니다. 게임이 종료되었습니다."); self._set_ai_status("AI (FastEV)\n\n게임 종료\n\n모든 카테고리가 사용되었습니다."); self._show_final_result()

    def _show_final_result(self) -> None:
        QMessageBox.information(self, "게임 종료", f"12턴이 모두 완료되었습니다.\nPLAYER {self.engine.state.player_score}점\nAI {self.engine.state.ai_score}점")

    def _refresh(self, dice: Iterable[int] | None = None) -> None:
        state = self.engine.state; dice_values = tuple(dice) if dice is not None else state.current_dice
        for index, button in enumerate(self.die_buttons):
            value = dice_values[index] if dice_values and index < len(dice_values) else None
            if value and value in self.dice_icons: button.setIcon(self.dice_icons[value]); button.setIconSize(QSize(98, 98)); button.setText("")
            else: button.setIcon(QIcon()); button.setText("-")
            button.setProperty("held", index in state.held_indices); button.style().unpolish(button); button.style().polish(button); button.setEnabled(value is not None and not self.ai_active and state.current_player is PlayerId.PLAYER and not state.turn_scored)
        remaining = max(0, MAX_ROLLS_PER_TURN - state.roll_count); self.roll_label.setText(f"{remaining}회 남음"); self.roll_button.setEnabled(not self.ai_active and state.current_player is PlayerId.PLAYER and not state.turn_scored and state.roll_count < MAX_ROLLS_PER_TURN)
        available: set[Category] = set(); scores: dict[Category, int] = {}
        # score_category() marks the turn as scored before _refresh() is called.
        # Do not query turn-only engine APIs during that short transition state.
        if state.current_dice and state.current_player is PlayerId.PLAYER and not self.ai_active and not state.turn_scored:
            available = set(self.engine.get_available_categories()); scores = self.engine.get_current_scores()
        for category, button in self.category_buttons.items():
            is_available = category in available; button.setProperty("available", is_available); button.setText(f"{CATEGORY_SHORT[category]}\n사용 가능 점수: {scores[category]}점" if is_available else f"{CATEGORY_SHORT[category]}\n사용됨"); button.setEnabled(is_available and not self.ai_active); button.style().unpolish(button); button.style().polish(button)
        for category in ALL_CATEGORIES:
            for player in (PlayerId.PLAYER, PlayerId.AI):
                score = state.players[player].category_scores.get(category); self.score_labels[player, category].setText("-" if score is None else str(score))
        self.player_total.setText(str(state.player_score)); self.ai_total.setText(str(state.ai_score)); self.center_total_label.setText(f"{state.player_score} : {state.ai_score}")
        diff = state.player_score - state.ai_score
        gap_text, gap_object = ((f"점수 차이 {diff}점 · PLAYER 우세", "scoreGap") if diff > 0 else (f"점수 차이 {abs(diff)}점 · AI 우세", "scoreGapNegative") if diff < 0 else ("점수 차이 0점 · 동점", "scoreGap"))
        self.gap_label.setText(gap_text); self.gap_label.setObjectName(gap_object); self.gap_label.style().unpolish(self.gap_label); self.gap_label.style().polish(self.gap_label); self.player_upper_label.setText(f"{state.player_upper_total} / 63"); self.ai_upper_label.setText(f"{state.ai_upper_total} / 63"); self.player_bonus_summary.setText("획득 (+35)" if state.player_has_bonus else "-"); self.ai_bonus_summary.setText("획득 (+35)" if state.ai_has_bonus else "-"); self.player_table_total.setText(str(state.player_score)); self.ai_table_total.setText(str(state.ai_score)); self.turn_label.setText(f"턴 {min(12, state.turn)} / 12")

    def _set_status(self, text: str) -> None:
        if self.status_label: self.status_label.setText(text)
        if self.log_label: self.log_label.setText(text)

    def _set_ai_status(self, text: str) -> None:
        if self.ai_status_label: self.ai_status_label.setText(text)

    def show_rules(self) -> None:
        QMessageBox.information(self, "Yacht 규칙", "• 플레이어마다 12개 카테고리를 한 번씩 사용합니다.\n• 한 턴에 최대 3번 굴릴 수 있습니다.\n• 굴린 주사위를 클릭하면 HOLD할 수 있습니다.\n• 마지막으로 사용하지 않은 카테고리 하나를 선택해 점수를 기록합니다.\n• 상단 6개 합계가 63점 이상이면 35점 보너스를 획득합니다.\n• 모든 카테고리를 사용하면 게임이 종료됩니다.")


def main() -> int:
    app = QApplication(sys.argv); app.setApplicationName("Yacht Game Prototype"); app.setFont(QFont("Segoe UI", 10)); window = YachtWindow(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())