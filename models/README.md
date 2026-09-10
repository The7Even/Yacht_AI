# Trained Yacht AI model

Place the trained PyTorch state_dict at:

`models/yacht_ai_brain.pth`

The runtime expects the supplied 30-input / 12-output network:

- 30 inputs: five dice encoded as five 6-way one-hot blocks
- 12 outputs: `ALL_CATEGORIES` order (`Ones` through `Yacht`)
- Architecture: `30 → 256 → 256 → 128 → 12` with ReLU activations between layers

The neural model selects the score category. FastEV continues to choose HOLD/REROLL actions so the model's category output is not mixed with FastEV's expected-value scale.
