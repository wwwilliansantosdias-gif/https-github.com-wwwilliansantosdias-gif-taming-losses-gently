# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small Python library of robust/smooth loss functions for ML training (PyTorch). Functions live
in a single module and are exposed at the package root; there is no framework beyond that — no
training loop, CLI, or config system in this repo.

## Commands

```bash
# Set up (editable install with dev deps: pytest, ruff)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run the full test suite
.venv/bin/pytest -q

# Run a single test
.venv/bin/pytest -q tests/test_losses.py::test_focal_loss_is_non_negative

# Lint
.venv/bin/ruff check .
```

There is no separate build step — the package is pure Python, installed via `pip install -e`.

## Architecture

- `src/taming_losses_gently/losses.py` — all loss functions live here as pure, differentiable
  functions of `torch.Tensor` (not `nn.Module` classes). Each takes raw logits (pre-sigmoid /
  pre-softmax) rather than probabilities, and a `reduction` arg (`"mean"` / `"sum"` / `"none"`)
  where applicable, mirroring `torch.nn.functional` conventions. A shared `_reduce()` helper
  implements the reduction so new losses should call it rather than reimplementing.
- `src/taming_losses_gently/__init__.py` — re-exports every public loss function; this is the
  intended import surface (`from taming_losses_gently import focal_loss`, not deep imports).
- `tests/test_losses.py` — one test module for all losses. Tests check numerical properties
  (non-negativity, boundedness, zero/near-zero at the "perfect prediction" case, equivalence to a
  known reference like `F.cross_entropy` at a limiting parameter value) rather than golden values.
- Packaging is `pyproject.toml` (setuptools, src-layout). Ruff config (line-length 100, `E`/`F`/
  `I`/`UP` rules) and pytest config both live in `pyproject.toml` — there are no separate `.flake8`
  / `ruff.toml` / `pytest.ini` files.

## Conventions for adding a new loss

- Accept raw logits, not post-activation probabilities (apply `sigmoid`/`softmax` internally).
- Support a `reduction` parameter via the shared `_reduce()` helper when the loss reduces over a
  batch dimension.
- Add a docstring explaining what problem the loss addresses (not just the formula) — see existing
  functions for the expected level of detail.
- Add tests asserting a mathematical property (bounds, a known-equivalent limiting case, or
  behavior at a trivial input) rather than a hardcoded numeric expectation.
