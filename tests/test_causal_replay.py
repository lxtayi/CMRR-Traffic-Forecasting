import numpy as np

from cmrr.causal_error_memory import causal_memory
from cmrr.multiscale_causal_router import seasonal_features


def test_future_target_cannot_change_earlier_corrections() -> None:
    pred = np.zeros((10, 3, 2), dtype=np.float32)
    true = np.ones_like(pred)
    changed = true.copy()
    changed[4, 2, 0] = 100.0

    original, _ = causal_memory(pred, true, alpha=0.2, clip=10.0, node_weight=0.8)
    counterfactual, _ = causal_memory(pred, changed, alpha=0.2, clip=10.0, node_weight=0.8)

    # Horizon 3 at origin 4 matures only at replay index 7.
    np.testing.assert_array_equal(original[:7], counterfactual[:7])


def test_zero_target_does_not_create_residual_feedback() -> None:
    pred = np.zeros((5, 1, 1), dtype=np.float32)
    true = np.zeros_like(pred)
    corrected, _ = causal_memory(pred, true, alpha=1.0, clip=10.0, node_weight=1.0)
    np.testing.assert_array_equal(corrected, pred)


def test_periodic_features_use_only_requested_lag() -> None:
    pred = np.zeros((8, 2, 1), dtype=np.float32)
    true = np.ones_like(pred)
    features = seasonal_features(pred, true, lags=(3,), clip=40.0)
    np.testing.assert_array_equal(features[:3], 0.0)
    np.testing.assert_allclose(features[3:, :, :, 0], 1.0)
