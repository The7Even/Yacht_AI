"""Inference adapter for the trained Yacht category-selection network.

The supplied ``yacht_ai_brain.pth`` contains only a PyTorch state_dict. Its
30-input / 12-output shape is compatible with a natural Yacht representation:
five dice, each one-hot encoded across six faces, followed by one output per
score category in ``ALL_CATEGORIES`` order.

This adapter deliberately handles only category selection. FastEV remains
responsible for deciding whether and which dice to reroll.
"""

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from app.core.categories import ALL_CATEGORIES, Category
from app.core.dice import DICE_COUNT, MAX_FACE, MIN_FACE, validate_dice


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yacht_ai_brain.pth"


class YachtBrain(nn.Module):
    """Network architecture matching the supplied trained state_dict."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralCategorySelector:
    """Load the trained network and choose an available score category."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model = YachtBrain()
        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def encode_dice(dice: Iterable[int]) -> torch.Tensor:
        """Encode five dice as 30 binary features: five 6-way one-hot blocks."""
        values = validate_dice(dice)
        features = torch.zeros(30, dtype=torch.float32)
        for index, face in enumerate(values):
            features[index * 6 + (face - MIN_FACE)] = 1.0
        return features

    @property
    def category_order(self) -> tuple[Category, ...]:
        return ALL_CATEGORIES

    def predict(self, dice: Iterable[int]) -> tuple[float, ...]:
        """Return the network's 12 raw output values in category order."""
        features = self.encode_dice(dice).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(features)[0]
        return tuple(float(value) for value in outputs)

    def select(
        self,
        dice: Iterable[int],
        available_categories: Iterable[Category],
    ) -> tuple[Category, float, dict[Category, float]]:
        """Choose the highest-valued legal category.

        Used categories are masked out before argmax, so the trained network
        can never request an already-used category.
        """
        available = tuple(available_categories)
        if not available:
            raise ValueError("At least one category must be available.")
        outputs = self.predict(dice)
        values = {category: outputs[index] for index, category in enumerate(ALL_CATEGORIES)}
        category = max(
            available,
            key=lambda candidate: (values[candidate], -ALL_CATEGORIES.index(candidate)),
        )
        return category, values[category], values
