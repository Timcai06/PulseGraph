from app.runtime.demo_mlp import demo_graph, run_demo_forward


def test_demo_graph_has_expected_flow() -> None:
    graph = demo_graph()

    assert [node.id for node in graph.nodes] == ["input", "flatten", "linear1", "relu1", "linear2", "softmax"]
    assert graph.edges[0].source == "input"
    assert graph.edges[-1].target == "softmax"


def test_demo_forward_returns_prediction_and_layers() -> None:
    result = run_demo_forward(3)

    assert result.sample_index == 3
    assert result.label == 3
    assert 0 <= result.prediction <= 9
    assert len(result.image_pixels) == 28 * 28
    assert all(0.0 <= pixel <= 1.0 for pixel in result.image_pixels)
    assert len(result.probabilities) == 10
    assert abs(sum(result.probabilities) - 1.0) < 1e-5
    assert len(result.layers) == 5
