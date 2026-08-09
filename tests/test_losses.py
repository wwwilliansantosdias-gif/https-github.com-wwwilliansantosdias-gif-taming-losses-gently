import torch

from taming_losses_gently import (
    charbonnier_loss,
    dice_loss,
    focal_loss,
    label_smoothing_cross_entropy,
)


def test_charbonnier_loss_at_equal_inputs_equals_eps():
    x = torch.randn(8, 3)
    eps = 1e-3
    assert torch.isclose(charbonnier_loss(x, x, eps=eps), torch.tensor(eps), atol=1e-6)


def test_charbonnier_loss_reduction_none_matches_shape():
    pred = torch.randn(4, 5)
    target = torch.randn(4, 5)
    loss = charbonnier_loss(pred, target, reduction="none")
    assert loss.shape == pred.shape
    assert torch.all(loss > 0)


def test_focal_loss_is_non_negative():
    logits = torch.randn(16, requires_grad=True)
    targets = torch.randint(0, 2, (16,))
    loss = focal_loss(logits, targets)
    assert loss.item() >= 0
    loss.backward()
    assert logits.grad is not None


def test_focal_loss_down_weights_confident_correct_predictions():
    targets = torch.ones(4)
    confident_logits = torch.full((4,), 10.0)
    unconfident_logits = torch.full((4,), 0.5)
    confident_loss = focal_loss(confident_logits, targets, reduction="mean")
    unconfident_loss = focal_loss(unconfident_logits, targets, reduction="mean")
    assert confident_loss < unconfident_loss


def test_label_smoothing_cross_entropy_matches_plain_ce_when_smoothing_zero():
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    smoothed = label_smoothing_cross_entropy(logits, targets, smoothing=0.0)
    plain = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(smoothed, plain, atol=1e-5)


def test_dice_loss_is_near_zero_for_perfect_prediction():
    targets = torch.randint(0, 2, (2, 16)).float()
    logits = (targets * 2 - 1) * 10  # confident correct logits
    loss = dice_loss(logits, targets)
    assert loss.item() < 1e-2


def test_dice_loss_is_bounded():
    logits = torch.randn(4, 32)
    targets = torch.randint(0, 2, (4, 32)).float()
    loss = dice_loss(logits, targets)
    assert 0 <= loss.item() <= 1
