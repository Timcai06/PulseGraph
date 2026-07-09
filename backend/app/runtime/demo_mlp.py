from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn

from app.inspector.fx_tracer import trace_model_graph
from app.runtime.inference_output import classification_output
from app.runtime.mnist_data import load_test_samples, test_sample_indices_by_digit, trained_model_path
from app.schemas import GraphEdge, GraphNode, LayerSnapshot, ModelGraph, PredictionResponse


class DemoMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def forward_steps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        flattened = self.net[0](images)
        hidden_raw = self.net[1](flattened)
        hidden_active = self.net[2](hidden_raw)
        logits = self.net[3](hidden_active)
        probabilities = torch.softmax(logits, dim=1)
        return {
            "input": images,
            "flatten": flattened,
            "linear1": hidden_raw,
            "relu1": hidden_active,
            "linear2": logits,
            "softmax": probabilities,
        }


@lru_cache(maxsize=1)
def get_demo_model() -> tuple[DemoMLP, str]:
    """Build the demo MLP once, preferring the trained exercise checkpoint."""
    torch.manual_seed(7)
    model = DemoMLP()
    weights = "random"
    checkpoint = trained_model_path()
    if checkpoint.exists():
        try:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            weights = "trained"
        except (RuntimeError, ValueError, OSError):
            weights = "random"
    model.eval()
    return model, weights


DEMO_NODE_NAMES = {"net.0": "flatten", "net.1": "linear1", "net.2": "relu1", "net.3": "linear2"}


@lru_cache(maxsize=1)
def demo_graph() -> ModelGraph:
    """Trace the trusted demo model with torch.fx instead of hand-writing the graph."""
    model, _ = get_demo_model()
    graph = trace_model_graph(model, torch.zeros(1, 1, 28, 28), rename=DEMO_NODE_NAMES)
    # forward() returns logits; the runtime applies softmax as an extra step.
    graph.nodes.append(
        GraphNode(id="softmax", label="Softmax", kind="Softmax", input_shape=[10], output_shape=[10], confidence="trusted")
    )
    graph.edges.append(GraphEdge(id="linear2->softmax", source="linear2", target="softmax"))
    return graph


def _synthetic_digit(label: int) -> torch.Tensor:
    image = torch.zeros(1, 1, 28, 28)
    center_x = 8 + (label % 5) * 3
    center_y = 8 + (label // 5) * 8
    for y in range(28):
        for x in range(28):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            stroke = max(0.0, 1.0 - distance / (4 + label * 0.12))
            ring = 0.45 if abs(distance - (5 + label * 0.1)) < 0.7 else 0.0
            image[0, 0, y, x] = max(stroke, ring)
    return image


def sample_digit(index: int) -> tuple[torch.Tensor, int, str]:
    """Pick a digit whose label is index % 10, preferring real MNIST test samples."""
    label = index % 10
    data = load_test_samples()
    by_digit = test_sample_indices_by_digit()
    if data is not None and by_digit is not None and by_digit[label]:
        images, _ = data
        candidates = by_digit[label]
        position = candidates[(index // 10) % len(candidates)]
        return images[position : position + 1], label, "mnist"
    return _synthetic_digit(label), label, "synthetic"


def _snapshot(layer_id: str, tensor: torch.Tensor, input_shape: list[int] | None = None) -> LayerSnapshot:
    data = tensor.detach().float().cpu()
    sparsity = float((data == 0).float().mean().item()) if data.numel() else 0.0
    return LayerSnapshot(
        layer_id=layer_id,
        input_shape=input_shape,
        output_shape=list(tensor.shape[1:]) if tensor.dim() > 1 else list(tensor.shape),
        activation_mean=float(data.mean().item()) if data.numel() else 0.0,
        activation_sparsity=sparsity,
    )


def run_demo_forward(index: int = 0) -> PredictionResponse:
    model, weights = get_demo_model()
    image, label, sample_source = sample_digit(index)
    with torch.no_grad():
        steps = model.forward_steps(image)
    probabilities = steps["softmax"].squeeze(0)
    probability_values = [float(value) for value in probabilities.tolist()]
    prediction = int(probabilities.argmax().item())
    layers = [
        _snapshot("flatten", steps["flatten"], [1, 28, 28]),
        _snapshot("linear1", steps["linear1"], [784]),
        _snapshot("relu1", steps["relu1"], [128]),
        _snapshot("linear2", steps["linear2"], [128]),
        _snapshot("softmax", steps["softmax"], [10]),
    ]
    return PredictionResponse(
        task="classification",
        output=classification_output(label=label, prediction=prediction, probabilities=probability_values),
        sample_index=index,
        label=label,
        prediction=prediction,
        weights=weights,
        sample_source=sample_source,
        image_shape=[1, 28, 28],
        image_pixels=[float(value) for value in image.squeeze(0).squeeze(0).flatten().tolist()],
        probabilities=probability_values,
        graph=demo_graph(),
        layers=layers,
    )
