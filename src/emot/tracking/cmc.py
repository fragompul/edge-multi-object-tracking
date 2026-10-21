"""Camera-motion compensation (CMC) via sparse optical flow.

When the camera pans or the operator walks (4 of the 7 MOT17 training
sequences are shot hand-held or from a moving platform), *every* box on
screen shifts at once. A Kalman filter has no way to know that: it reads the
shift as a sudden, simultaneous acceleration of every pedestrian, the
predicted boxes land in the wrong place, IoU collapses, and identities
break. The fix (Aharon et al., BoT-SORT, 2022) is to estimate the global
image motion between consecutive frames and warp every track's state by it
before association.

Estimator: Shi-Tomasi corners -> pyramidal Lucas-Kanade flow ->
RANSAC-fitted 4-DoF similarity transform. Pedestrians are outliers to the
(rigid) background motion, which RANSAC rejects.
"""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class CameraMotionEstimator:
    def __init__(self, downscale: int = 2, max_corners: int = 1000):
        self.downscale = downscale
        self.max_corners = max_corners
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_points = None

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns the 2x3 affine mapping the *previous* frame's pixel
        coordinates onto this frame's (identity for the first frame or when
        too few reliable features are found)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(
                gray,
                None,
                fx=1 / self.downscale,
                fy=1 / self.downscale,
                interpolation=cv2.INTER_AREA,
            )
        h, w = gray.shape
        mask = np.zeros_like(gray)
        mask[int(0.02 * h) : int(0.98 * h), int(0.02 * w) : int(0.98 * w)] = 255

        warp = IDENTITY.copy()
        if self._prev_gray is not None and self._prev_points is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                self._prev_gray, gray, self._prev_points, None
            )
            ok = status.ravel() == 1
            if ok.sum() >= 10:
                matrix, _ = cv2.estimateAffinePartial2D(
                    self._prev_points[ok], next_pts[ok], method=cv2.RANSAC, ransacReprojThreshold=3
                )
                if matrix is not None:
                    warp = matrix.astype(float)
                    warp[:, 2] *= self.downscale  # back to full-resolution pixels

        self._prev_gray = gray
        self._prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            mask=mask,
        )
        return warp
