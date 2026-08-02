"""Constant-velocity Kalman filter over `[cx, cy, aspect, height]` bounding
box state, the standard SORT/DeepSORT/ByteTrack parameterization: it tracks
the box center, aspect ratio (width/height), and height directly rather
than width/height independently, since aspect ratio is far more stable
than width under partial occlusion (a person's silhouette narrows when
occluded from the side, but the box center and aspect vary more slowly
than raw width does frame-to-frame).

State vector (8,): [cx, cy, a, h, vcx, vcy, va, vh].
"""

from __future__ import annotations

import numpy as np

# Motion/observation noise scales, following the original SORT/DeepSORT
# convention of tying uncertainty to the current box height -- a box that's
# already large (close to camera) is expected to move more in pixel terms
# than a small, distant one for the same real-world motion.
STD_WEIGHT_POSITION = 1.0 / 20
STD_WEIGHT_VELOCITY = 1.0 / 160


def xyxy_to_state(xyxy: np.ndarray) -> np.ndarray:
    left, top, right, bottom = xyxy
    width = right - left
    height = bottom - top
    cx = left + width / 2
    cy = top + height / 2
    aspect = width / height if height > 1e-6 else 1e-6
    return np.array([cx, cy, aspect, height], dtype=float)


def state_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, aspect, height = state[:4]
    width = aspect * height
    return np.array(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], dtype=float
    )


class KalmanBoxTracker:
    """One Kalman filter instance for one track's `[cx, cy, a, h]` box state."""

    _F = np.eye(8)
    for _i in range(4):
        _F[_i, _i + 4] = 1.0

    _H = np.eye(4, 8)

    def __init__(self, initial_xyxy: np.ndarray):
        self.mean = np.zeros(8)
        self.mean[:4] = xyxy_to_state(initial_xyxy)
        height = self.mean[3]
        std = [
            2 * STD_WEIGHT_POSITION * height,
            2 * STD_WEIGHT_POSITION * height,
            1e-2,
            2 * STD_WEIGHT_POSITION * height,
            10 * STD_WEIGHT_VELOCITY * height,
            10 * STD_WEIGHT_VELOCITY * height,
            1e-5,
            10 * STD_WEIGHT_VELOCITY * height,
        ]
        self.covariance = np.diag(np.square(std))

    def predict(self) -> None:
        height = self.mean[3]
        process_std = [
            STD_WEIGHT_POSITION * height,
            STD_WEIGHT_POSITION * height,
            1e-2,
            STD_WEIGHT_POSITION * height,
            STD_WEIGHT_VELOCITY * height,
            STD_WEIGHT_VELOCITY * height,
            1e-5,
            STD_WEIGHT_VELOCITY * height,
        ]
        process_cov = np.diag(np.square(process_std))
        self.mean = self._F @ self.mean
        self.covariance = self._F @ self.covariance @ self._F.T + process_cov

    def update(self, xyxy: np.ndarray) -> None:
        measurement = xyxy_to_state(xyxy)
        height = self.mean[3]
        obs_std = [
            STD_WEIGHT_POSITION * height,
            STD_WEIGHT_POSITION * height,
            1e-1,
            STD_WEIGHT_POSITION * height,
        ]
        obs_cov = np.diag(np.square(obs_std))

        projected_mean = self._H @ self.mean
        projected_cov = self._H @ self.covariance @ self._H.T + obs_cov

        kalman_gain = self.covariance @ self._H.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean

        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ self._H @ self.covariance

    @property
    def xyxy(self) -> np.ndarray:
        return state_to_xyxy(self.mean)

    def mahalanobis_gate(self, xyxy: np.ndarray) -> float:
        """Squared Mahalanobis distance between this track's predicted
        position and a candidate detection box, used by DeepSORT to reject
        motion-implausible matches before appearance is even considered."""
        measurement = xyxy_to_state(xyxy)
        height = self.mean[3]
        obs_std = [
            STD_WEIGHT_POSITION * height,
            STD_WEIGHT_POSITION * height,
            1e-1,
            STD_WEIGHT_POSITION * height,
        ]
        obs_cov = np.diag(np.square(obs_std))
        projected_mean = self._H @ self.mean
        projected_cov = self._H @ self.covariance @ self._H.T + obs_cov
        diff = measurement - projected_mean
        return float(diff.T @ np.linalg.inv(projected_cov) @ diff)
