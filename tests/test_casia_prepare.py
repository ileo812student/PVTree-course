# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PVTree-course contributors.
# Course-project addition; see LICENSE for terms and README for upstream attribution.
import numpy as np

from casia_prepare import extract_roi


def _synthetic_hand():
    import cv2
    image = np.zeros((576, 768), dtype=np.uint8)
    cv2.ellipse(image, (430, 290), (135, 125), 0, 0, 360, 130, -1)
    cv2.rectangle(image, (520, 245), (767, 335), 130, -1)
    for y in (180, 225, 270, 315):
        cv2.rectangle(image, (100, y), (400, y + 28), 130, -1)
    return image


def test_extract_roi_shape_and_metadata():
    roi, metadata, _ = extract_roi(_synthetic_hand(), "left")
    assert roi.shape == (128, 128)
    assert roi.dtype == np.uint8
    assert metadata["horizontal_flip"] is False
    assert metadata["vertical_flip_for_right_hand"] is False
