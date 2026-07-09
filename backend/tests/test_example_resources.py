from pathlib import Path

from app.resources.contract import load_training_resource


def test_cifar_rgb_example_resource_loads_and_exposes_samples() -> None:
    resource_path = Path(__file__).resolve().parents[2] / "examples" / "resources" / "cifar_rgb_resource.py"

    resource = load_training_resource(resource_path, source_root=resource_path.parent)

    assert resource.name == "cifar_rgb_example"
    assert resource.input_shape == [3, 32, 32]
    assert resource.class_names == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    images, labels = resource.train_batch(step=1, batch_size=4)
    assert list(images.shape) == [4, 3, 32, 32]
    assert labels.tolist() == [0, 1, 2, 3]
    sample, label = resource.inference_sample(3)
    assert list(sample.shape) == [3, 32, 32]
    assert label == 3
