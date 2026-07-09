from pathlib import Path

from app.resources.contract import load_training_resource


def test_rgb_pattern_example_resource_loads_and_exposes_samples() -> None:
    resource_path = Path(__file__).resolve().parents[2] / "examples" / "resources" / "cifar_rgb_resource.py"

    resource = load_training_resource(resource_path, source_root=resource_path.parent)

    assert resource.name == "rgb_pattern_example"
    assert resource.input_shape == [3, 32, 32]
    assert resource.class_names == [
        "pattern_00",
        "pattern_01",
        "pattern_02",
        "pattern_03",
        "pattern_04",
        "pattern_05",
        "pattern_06",
        "pattern_07",
        "pattern_08",
        "pattern_09",
    ]
    assert resource.metadata["data_source"] == "synthetic-rgb-patterns"
    images, labels = resource.train_batch(step=1, batch_size=4)
    assert list(images.shape) == [4, 3, 32, 32]
    assert labels.tolist() == [0, 1, 2, 3]
    sample, label = resource.inference_sample(3)
    assert list(sample.shape) == [3, 32, 32]
    assert label == 3
