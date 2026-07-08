from pathlib import Path

import torch

import app.inspector.pt_inspector as pt_inspector
from app.inspector.pt_inspector import inspect_pt_file


def _write_marker(path: str) -> None:
    Path(path).write_text("executed")


def test_inspect_plain_state_dict(tmp_path: Path) -> None:
    model_path = tmp_path / "mlp.pt"
    torch.save(
        {
            "net.1.weight": torch.randn(128, 784),
            "net.1.bias": torch.randn(128),
            "net.3.weight": torch.randn(10, 128),
            "net.3.bias": torch.randn(10),
        },
        model_path,
    )

    result = inspect_pt_file(model_path)

    assert result.safe is True
    assert result.mode == "state_dict"
    assert len(result.tensors) == 4
    assert result.tensors[0].shape
    linear_nodes = [node for node in result.graph.nodes if node.kind == "Linear"]
    assert [node.output_shape for node in linear_nodes] == [[128], [10]]


def test_inspect_checkpoint_bundle(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": {"layer.weight": torch.randn(4, 2), "layer.bias": torch.randn(4)},
        },
        model_path,
    )

    result = inspect_pt_file(model_path)

    assert result.mode == "checkpoint"
    assert any("model_state_dict" in warning for warning in result.warnings)
    assert len(result.graph.nodes) == 2


def test_inspection_uses_weights_only_loading(monkeypatch, tmp_path: Path) -> None:
    calls = []
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"layer.weight": torch.ones(1, 1)}

    monkeypatch.setattr(pt_inspector.torch, "load", fake_load)

    result = inspect_pt_file(model_path)

    assert result.mode == "state_dict"
    assert calls[0][1]["weights_only"] is True
    assert calls[0][1]["map_location"] == "cpu"


def test_malicious_pickle_does_not_execute_on_inspection(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    model_path = tmp_path / "hostile.pt"

    class HostilePayload:
        def __reduce__(self):
            return (_write_marker, (str(marker),))

    torch.save({"payload": HostilePayload()}, model_path)

    result = inspect_pt_file(model_path)

    assert result.safe is True
    assert result.mode == "unknown"
    assert not marker.exists()
    assert result.warnings == [
        "Safe weights-only loading failed.",
        "The artifact could not be decoded as tensor weights without unsafe deserialization.",
    ]
