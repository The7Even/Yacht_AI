"""Launch the existing Yacht UI with the trained neural category model.

The underlying YachtWindow already uses ExpectedValueStrategy. After the
category evaluator patch, that strategy automatically uses
``models/yacht_ai_brain.pth`` for category selection when the file exists.
This launcher only changes the visible AI label/status from FastEV to Neural AI.
"""

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel

from .gui import YachtWindow


class NeuralYachtWindow(YachtWindow):
    """Existing UI with labels identifying the trained neural model."""

    def __init__(self) -> None:
        super().__init__()
        self._rename_ai_labels()

    def _rename_ai_labels(self) -> None:
        for label in self.findChildren(QLabel):
            text = label.text()
            if "AI (FastEV)" in text:
                label.setText(text.replace("AI (FastEV)", "AI (Neural)"))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Yacht Game Prototype - Neural AI")
    app.setFont(QFont("Segoe UI", 10))
    window = NeuralYachtWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
