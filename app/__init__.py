"""Yacht AI application package and GUI session logging adapter.

The prototype GUI already emits a useful verbose debug stream through
``yacht.gui``. This package initializer keeps that stream available on the
console while adding two user-facing conveniences without changing the game
engine: a compact on-screen event log and a per-session CSV trace.
"""
from __future__ import annotations

import csv
import logging
import re
from datetime import datetime
from pathlib import Path

_SESSION_STAMP = datetime.now().strftime("%Y%m%d%H%M%S")
_CSV_PATH = Path.cwd() / f"gamelog_{_SESSION_STAMP}.csv"
_FIELDS = [
    "timestamp", "event", "player", "turn", "roll", "dice", "held", "action",
    "category", "score", "expected_value", "reasoning", "player_total", "ai_total",
    "player_upper", "ai_upper", "bonus",
]

_csv_file = _CSV_PATH.open("w", encoding="utf-8-sig", newline="")
_csv_writer = csv.DictWriter(_csv_file, fieldnames=_FIELDS)
_csv_writer.writeheader()
_csv_file.flush()


def _write_csv(**values: object) -> None:
    row = {field: values.get(field, "") for field in _FIELDS}
    _csv_writer.writerow(row)
    _csv_file.flush()


def _structured_debug(message: str) -> None:
    """Parse the existing GUI debug stream into analysis-friendly rows."""
    now = datetime.now().isoformat(timespec="milliseconds")
    row: dict[str, object] = {"timestamp": now, "event": "DEBUG", "reasoning": message}
    patterns = [
        ("ROLL", r"\[(PLAYER|AI)\]\[TURN (\d+)\]\[ROLL (\d+)/3\] dice=(.*?) held=\[(.*?)\]"),
        ("DECISION", r"\[AI\]\[TURN (\d+)\]\[DECISION\] roll=(\d+)/3 hold=\[(.*?)\] EV=([\d.-]+) reasoning=(.*)"),
        ("SCORE", r"\[(PLAYER|AI)\]\[TURN (\d+)\]\[SCORE\] (.*?)=(\d+) total=(\d+)"),
    ]
    for kind, pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        if kind == "ROLL":
            row.update(event=kind, player=match.group(1), turn=match.group(2), roll=match.group(3), dice=match.group(4), held=match.group(5))
        elif kind == "DECISION":
            row.update(event=kind, player="AI", turn=match.group(1), roll=match.group(2), held=match.group(3), expected_value=match.group(4), action="HOLD", reasoning=match.group(5))
        else:
            row.update(event=kind, player=match.group(1), turn=match.group(2), category=match.group(3), score=match.group(4), action="SCORE")
        break
    _write_csv(**row)


class _CsvHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _structured_debug(record.getMessage())
        except Exception:
            pass


_logger = logging.getLogger("yacht.gui")
_csv_handler = _CsvHandler(level=logging.DEBUG)
_logger.addHandler(_csv_handler)


def _install_qt_adapters() -> None:
    from PySide6.QtWidgets import QLabel, QPlainTextEdit

    original_append = QPlainTextEdit.appendPlainText
    if not getattr(original_append, "_yacht_adapter", False):
        def append_plain_text(self, text: str) -> None:
            if self.objectName() != "logView":
                return original_append(self, text)
            if text.startswith("[GAME] 새 게임 시작"):
                return original_append(self, "새 게임을 시작했습니다.")
            if text.startswith("[BONUS]"):
                return original_append(self, text.replace("[BONUS] ", ""))
            if text.startswith("[GAME OVER]"):
                return original_append(self, "게임 종료 · " + text.removeprefix("[GAME OVER] "))
            match = re.search(r"\[(PLAYER|AI)\]\[TURN (\d+)\]\[SCORE\] (.*?)=(\d+)", text)
            if match:
                return original_append(self, f"{match.group(1)} · {match.group(3)}에 {match.group(4)}점을 기록했습니다.")
            return None
        append_plain_text._yacht_adapter = True
        QPlainTextEdit.appendPlainText = append_plain_text

    original_set_text = QLabel.setText
    if not getattr(original_set_text, "_yacht_adapter", False):
        def set_text(self, text: str) -> None:
            if self.objectName() != "aiStatus":
                return original_set_text(self, text)
            source = text or ""
            if "오류" in source:
                display = "AI (FastEV)\n\n계산 중 오류가 발생했습니다."
            elif "게임 종료" in source:
                display = "AI (FastEV)\n\n게임이 종료되었습니다."
            elif "기록 선택" in source or "점수 항목을 선택" in source:
                display = "AI (FastEV)\n\n점수 항목을 선택했습니다.\n\n다음 턴을 계산하는 중입니다..."
            elif "최종" in source or "최종 주사위" in source:
                display = "AI (FastEV)\n\n최종 주사위 조합을 분석하는 중입니다...\n\n기록할 점수를 계산합니다."
            elif "HOLD" in source or "굴림 완료" in source or "굴림" in source:
                display = "AI (FastEV)\n\n주사위 조합을 분석하는 중입니다...\n\n다음 행동을 계산합니다."
            elif "턴 완료" in source:
                display = "AI (FastEV)\n\nAI 턴을 마쳤습니다.\n\nPLAYER의 다음 턴을 기다리는 중입니다."
            elif "준비" in source or "기다리는" in source:
                display = "AI (FastEV)\n\nAI 턴을 준비하고 있습니다..."
            else:
                display = "AI (FastEV)\n\n다음 행동을 계산하고 있습니다..."
            return original_set_text(self, display)
        set_text._yacht_adapter = True
        QLabel.setText = set_text


def _install_bonus_trace() -> None:
    from .core.game_engine import GameEngine
    original_score = GameEngine.score_category
    if getattr(original_score, "_yacht_bonus_adapter", False):
        return

    def score_category(self, category):
        player = self.state.current_player
        before = self.state.player_has_bonus if player.name == "PLAYER" else self.state.ai_has_bonus
        score = original_score(self, category)
        after = self.state.player_has_bonus if player.name == "PLAYER" else self.state.ai_has_bonus
        if not before and after:
            name = "PLAYER" if player.name == "PLAYER" else "AI"
            message = f"[BONUS] {name}가 63점 상단 보너스(+35점)를 획득했습니다."
            try:
                from PySide6.QtWidgets import QApplication
                for widget in QApplication.allWidgets():
                    if widget.objectName() == "logView":
                        widget.appendPlainText(message)
                        break
            except Exception:
                pass
            _write_csv(event="BONUS", player=name, turn=self.state.turn, bonus="+35", player_total=self.state.player_score, ai_total=self.state.ai_score, player_upper=self.state.player_upper_total, ai_upper=self.state.ai_upper_total)
            _logger.info(message)
        return score

    score_category._yacht_bonus_adapter = True
    GameEngine.score_category = score_category


try:
    _install_qt_adapters()
except Exception:
    pass

try:
    _install_bonus_trace()
except Exception:
    pass


def get_session_csv_path() -> Path:
    """Return the current GUI session's analysis CSV path."""
    return _CSV_PATH
