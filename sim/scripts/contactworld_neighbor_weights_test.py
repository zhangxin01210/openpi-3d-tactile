"""Research gates: no source leakage, exact prior initialization, convex output."""
import numpy as np
import torch

from contactworld_neighbor_weights import WeightNet, infer, prior_logits, retrieve, scale_features, softmax


def test_retrieval_excludes_entire_query_episode_and_keeps_distinct_sources():
    source = np.repeat(np.arange(12), 3)
    x = np.arange(len(source), dtype=np.float32)[:, None]
    q = np.array([0, 3, 35])
    ix, d = retrieve(x, source, q, np.arange(len(source)), k=10)
    assert not (source[ix] == source[q, None]).any()
    assert all(len(set(source[row])) == 10 for row in ix)
    assert np.isfinite(d).all()


def test_scaling_uses_fit_episodes_only():
    data = {"state": np.array([[0., 2.], [2., 4.], [100., -100.]], np.float32)}
    first, stats, _ = scale_features(data, np.array([0, 1]), "S")
    data["state"][2] *= 1000
    second, stats2, _ = scale_features(data, np.array([0, 1]), "S")
    np.testing.assert_array_equal(first[:2], second[:2])
    np.testing.assert_array_equal(stats["state"][0], stats2["state"][0])


def test_network_initializes_to_similarity_and_is_convex_without_target_input():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(15, 5)).astype(np.float32)
    action = rng.normal(size=(15, 4, 6)).astype(np.float32)
    query = np.array([12, 13, 14])
    candidates = np.tile(np.arange(10), (3, 1))
    distance = np.tile(np.linspace(.1, 1, 10), (3, 1)).astype(np.float32)
    model = WeightNet(5, 1)
    pred, weights = infer(model, x, action, query, candidates, distance, [(0, 5)], 10, 1., "cpu")
    np.testing.assert_allclose(weights, softmax(prior_logits(distance, 10, 1.)), atol=1e-7)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(1), 1., atol=1e-6)
    assert (pred >= action[candidates].min(1)-1e-6).all()
    assert (pred <= action[candidates].max(1)+1e-6).all()
    modified_actions = action.copy()
    modified_actions[query] = 1e5  # Held-out query labels never enter forward.
    pred2, weights2 = infer(model, x, modified_actions, query, candidates, distance, [(0, 5)], 10, 1., "cpu")
    np.testing.assert_array_equal(pred, pred2)
    np.testing.assert_array_equal(weights, weights2)


def test_nonzero_gradient_reaches_weight_correction():
    model = WeightNet(5, 1)
    q, c = torch.randn(3, 5), torch.randn(3, 10, 5)
    w = model(q, c, torch.zeros(3, 10), [(0, 5)])
    loss = (w*torch.arange(10)).sum()
    loss.backward()
    assert model.net[-1].weight.grad.abs().sum() > 0
