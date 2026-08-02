import numpy as np

from emot.tracking.kalman import KalmanBoxTracker, state_to_xyxy, xyxy_to_state


def test_xyxy_state_round_trip():
    box = np.array([10.0, 20.0, 60.0, 120.0])
    state = xyxy_to_state(box)
    recovered = state_to_xyxy(state)
    assert np.allclose(box, recovered)


def test_initial_prediction_matches_input_box():
    box = np.array([10.0, 20.0, 60.0, 120.0])
    kf = KalmanBoxTracker(box)
    assert np.allclose(kf.xyxy, box, atol=1e-6)


def test_filter_tracks_constant_velocity_motion():
    kf = KalmanBoxTracker(np.array([0.0, 0.0, 50.0, 100.0]))
    for step in range(1, 15):
        kf.predict()
        true_box = np.array([step * 10.0, 0.0, step * 10.0 + 50.0, 100.0])
        kf.update(true_box)
    kf.predict()
    predicted = kf.xyxy
    # After many updates on a constant-velocity target, the filter should
    # have learned the velocity and predict close to the true next position.
    expected_next = np.array([150.0, 0.0, 200.0, 100.0])
    assert np.allclose(predicted, expected_next, atol=5.0)


def test_mahalanobis_gate_small_for_close_match_large_for_far():
    kf = KalmanBoxTracker(np.array([0.0, 0.0, 50.0, 100.0]))
    close = kf.mahalanobis_gate(np.array([1.0, 1.0, 51.0, 101.0]))
    far = kf.mahalanobis_gate(np.array([500.0, 500.0, 550.0, 600.0]))
    assert close < far
