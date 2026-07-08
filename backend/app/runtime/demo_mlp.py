from __future__ import annotations

import math

import torch
from torch import nn

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


def create_demo_model() -> DemoMLP:
    torch.manual_seed(7)
    model = DemoMLP()
    model.eval()
    return model


def demo_graph() -> ModelGraph:
    nodes = [
        GraphNode(id="input", label="Input", kind="Input", output_shape=[1, 28, 28], confidence="trusted"),
        GraphNode(id="flatten", label="Flatten", kind="Flatten", input_shape=[1, 28, 28], output_shape=[784], confidence="trusted"),
        GraphNode(
            id="linear1",
            label="Linear 784 -> 128",
            kind="Linear",
            input_shape=[784],
            output_shape=[128],
            param_count=(784 * 128) + 128,
            confidence="trusted",
        ),
        GraphNode(id="relu1", label="ReLU", kind="ReLU", input_shape=[128], output_shape=[128], confidence="trusted"),
        GraphNode(
            id="linear2",
            label="Linear 128 -> 10",
            kind="Linear",
            input_shape=[128],
            output_shape=[10],
            param_count=(128 * 10) + 10,
            confidence="trusted",
        ),
        GraphNode(id="softmax", label="Softmax", kind="Softmax", input_shape=[10], output_shape=[10], confidence="trusted"),
    ]
    edges = [
        GraphEdge(id="input->flatten", source="input", target="flatten"),
        GraphEdge(id="flatten->linear1", source="flatten", target="linear1"),
        GraphEdge(id="linear1->relu1", source="linear1", target="relu1"),
        GraphEdge(id="relu1->linear2", source="relu1", target="linear2"),
        GraphEdge(id="linear2->softmax", source="linear2", target="softmax"),
    ]
    return ModelGraph(nodes=nodes, edges=edges)


def sample_digit(index: int) -> tuple[torch.Tensor, int]:
    label = index % 10
    image = torch.zeros(1, 1, 28, 28)
    center_x = 8 + (label % 5) * 3
    center_y = 8 + (label // 5) * 8
    for y in range(28):
        for x in range(28):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            stroke = max(0.0, 1.0 - distance / (4 + label * 0.12))
            ring = 0.45 if abs(distance - (5 + label * 0.1)) < 0.7 else 0.0
            image[0, 0, y, x] = max(stroke, ring)
    return image, label


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
    model = create_demo_model()
    image, label = sample_digit(index)
    with torch.no_grad():
        steps = model.forward_steps(image)
    probabilities = steps["softmax"].squeeze(0)
    prediction = int(probabilities.argmax().item())
    layers = [
        _snapshot("flatten", steps["flatten"], [1, 28, 28]),
        _snapshot("linear1", steps["linear1"], [784]),
        _snapshot("relu1", steps["relu1"], [128]),
        _snapshot("linear2", steps["linear2"], [128]),
        _snapshot("softmax", steps["softmax"], [10]),
    ]
    return PredictionResponse(
        sample_index=index,
        label=label,
        prediction=prediction,
        probabilities=[float(value) for value in probabilities.tolist()],
        graph=demo_graph(),
        layers=layers,
    )

