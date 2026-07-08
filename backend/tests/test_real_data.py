import pytest

from app.runtime import demo_mlp, mnist_data
from app.runtime.demo_mlp import get_demo_model, run_demo_forward


def _clear_caches() -> None:
    get_demo_model.cache_clear()
    mnist_data.load_train_samples.cache_clear()
    mnist_data.load_test_samples.cache_clear()
    mnist_data.test_sample_indices_by_digit.cache_clear()


@pytest.fixture(autouse=True)
def reset_caches():
    _clear_caches()
    yield
    _clear_caches()


def test_falls_back_to_synthetic_data_when_paths_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PULSEGRAPH_MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("PULSEGRAPH_MNIST_DIR", str(tmp_path / "missing-dir"))

    result = run_demo_forward(4)

    assert result.label == 4
    assert result.weights == "random"
    assert result.sample_source == "synthetic"
    assert len(result.image_pixels) == 28 * 28


@pytest.mark.skipif(
    not mnist_data.DEFAULT_MODEL_PATH.exists() or not mnist_data.DEFAULT_MNIST_RAW_DIR.exists(),
    reason="trained checkpoint or MNIST raw data not present in this checkout",
)
def test_uses_trained_weights_and_real_mnist_samples() -> None:
    result = run_demo_forward(7)

    assert result.label == 7
    assert result.weights == "trained"
    assert result.sample_source == "mnist"
    # A trained MLP should be confident and correct on an easy test digit.
    assert result.prediction == 7
    assert max(result.probabilities) > 0.5


@pytest.mark.skipif(
    not mnist_data.DEFAULT_MNIST_RAW_DIR.exists(),
    reason="MNIST raw data not present in this checkout",
)
def test_sample_digit_label_contract_holds_on_real_data() -> None:
    for index in range(20):
        _, label, source = demo_mlp.sample_digit(index)
        assert label == index % 10
        assert source == "mnist"
