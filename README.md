# taming-losses-gently

A small collection of robust and smooth loss functions for ML training, implemented
as pure PyTorch functions.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from taming_losses_gently import charbonnier_loss, focal_loss, label_smoothing_cross_entropy, dice_loss
```

See `src/taming_losses_gently/losses.py` for each function's docstring.

## Testing

```bash
pytest
```
