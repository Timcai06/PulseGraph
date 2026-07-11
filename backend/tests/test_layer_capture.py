import torch
from torch import nn

from app.runtime.model_loader import forward_with_layer_capture


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_layer_capture_includes_weight_and_gradient_stats_after_backward() -> None:
    model = _tiny_model()
    images = torch.randn(3, 4)
    loss = model(images).sum()
    loss.backward()  # grads populated, mirroring the post-step sampling point

    _, layers = forward_with_layer_capture(model, images[:1])

    by_id = {layer["layer_id"]: layer for layer in layers}
    assert set(by_id) == {"0", "2"}
    for layer in by_id.values():
        assert isinstance(layer["weight_std"], float)
        assert isinstance(layer["gradient_norm"], float)
        assert layer["gradient_norm"] > 0


def test_layer_capture_leaves_gradient_none_for_pure_inference() -> None:
    model = _tiny_model()

    _, layers = forward_with_layer_capture(model, torch.randn(1, 4))

    for layer in layers:
        assert layer["gradient_norm"] is None
        assert isinstance(layer["weight_std"], float)
