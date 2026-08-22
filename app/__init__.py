"""Runtime adapters for the Yacht prototype GUI."""
from __future__ import annotations

import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path.cwd() / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_STAMP = datetime.now().strftime("%Y%m%d%H%M%S")
_CSV_PATH = _LOG_DIR / f"gamelog_{_SESSION_STAMP}.csv"
_FIELDS = ["timestamp", "event", "player", "turn", "roll", "dice", "held", "action", "category", "score", "expected_value", "reasoning", "player_total", "ai_total", "player_upper", "ai_upper", "bonus", "raw_message"]
_csv_file = _CSV_PATH.open("w", encoding="utf-8-sig", newline="")
_csv_writer = csv.DictWriter(_csv_file, fieldnames=_FIELDS)
_csv_writer.writeheader()
_csv_file.flush()


def _write_csv(**values: object) -> None:
    _csv_writer.writerow({field: values.get(field, "") for field in _FIELDS})
    _csv_file.flush()


def log_console_event(message: str, *, level: str = "DEBUG") -> None:
    """Persist every GUI console event, parsing known Yacht fields when possible."""
    row: dict[str, object] = {"timestamp": datetime.now().isoformat(timespec="milliseconds"), "event": level, "raw_message": message}
    patterns = (
        ("ROLL", r"\[(PLAYER|AI)\]\[TURN (\d+)\]\[ROLL (\d+)/3\] dice=(.*?) held=\[(.*?)\]"),
        ("DECISION", r"\[AI\]\[TURN (\d+)\]\[DECISION\] roll=(\d+)/3.*?EV=([\d.-]+).*?reasoning=(.*)"),
        ("SCORE", r"\[(PLAYER|AI)\]\[TURN (\d+)\]\[SCORE\] (.*?)=(\d+) total=(\d+)"),
        ("GAME_OVER", r"\[GAME OVER\] result=(.*?) player=(\d+) ai=(\d+)"),
        ("BONUS", r"\[BONUS\] (.*)"),
    )
    for event, pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        row["event"] = event
        if event == "ROLL":
            row.update(player=match.group(1), turn=match.group(2), roll=match.group(3), dice=match.group(4), held=match.group(5), action="ROLL")
        elif event == "DECISION":
            row.update(player="AI", turn=match.group(1), roll=match.group(2), expected_value=match.group(3), reasoning=match.group(4), action="DECISION")
        elif event == "SCORE":
            row.update(player=match.group(1), turn=match.group(2), category=match.group(3), score=match.group(4), action="SCORE")
            row["player_total" if match.group(1) == "PLAYER" else "ai_total"] = match.group(5)
        elif event == "GAME_OVER":
            row.update(reasoning=match.group(1), player_total=match.group(2), ai_total=match.group(3), action="GAME_OVER")
        else:
            row["reasoning"] = match.group(1)
        break
    _write_csv(**row)


class _ConsoleCsvHandler(logging.Handler):
    """Preserve the existing console output while writing the same event to CSV."""
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        print(f"[{record.levelname}] {message}", file=sys.stdout, flush=True)
        try:
            log_console_event(message, level=record.levelname)
        except Exception:
            pass


_gui_logger = logging.getLogger("yacht.gui")
if not any(isinstance(handler, _ConsoleCsvHandler) for handler in _gui_logger.handlers):
    _gui_logger.addHandler(_ConsoleCsvHandler(level=logging.DEBUG))
    _gui_logger.setLevel(logging.DEBUG)
    _gui_logger.propagate = False


def get_session_csv_path() -> Path:
    return _CSV_PATH


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
            if "오류" in source: display = "AI (FastEV)\n\n계산 중 오류가 발생했습니다."
            elif "게임 종료" in source: display = "AI (FastEV)\n\n게임이 종료되었습니다."
            elif "기록 선택" in source or "점수 항목을 선택" in source: display = "AI (FastEV)\n\n점수 항목을 선택했습니다.\n\n다음 턴을 계산하는 중입니다..."
            elif "최종" in source or "최종 주사위" in source: display = "AI (FastEV)\n\n최종 주사위 조합을 분석하는 중입니다...\n\n기록할 점수를 계산합니다."
            elif "HOLD" in source or "굴림 완료" in source or "굴림" in source: display = "AI (FastEV)\n\n주사위 조합을 분석하는 중입니다...\n\n다음 행동을 계산합니다."
            elif "턴 완료" in source: display = "AI (FastEV)\n\nAI 턴을 마쳤습니다.\n\nPLAYER의 다음 턴을 기다리는 중입니다."
            elif "준비" in source or "기다리는" in source: display = "AI (FastEV)\n\nAI 턴을 준비하고 있습니다..."
            else: display = "AI (FastEV)\n\n다음 행동을 계산하고 있습니다..."
            return original_set_text(self, display)
        set_text._yacht_adapter = True
        QLabel.setText = set_text


def _install_bonus_trace() -> None:
    from .core.game_engine import GameEngine
    original_score = GameEngine.score_category
    if getattr(original_score, "_yacht_bonus_adapter", False): return
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
                    if widget.objectName() == "logView": widget.appendPlainText(message); break
            except Exception: pass
            log_console_event(message, level="INFO")
        return score
    score_category._yacht_bonus_adapter = True
    GameEngine.score_category = score_category

try: _install_qt_adapters()
except Exception: pass
try: _install_bonus_trace()
except Exception: pass
