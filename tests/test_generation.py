from argparse import Namespace

import cv2
import numpy as np
import pytest

from crop_veinline import crop
from get_bezier import generate_parameters, get_bezier
from main import (
    create_main_tree_c1,
    create_main_tree_c2,
    create_main_tree_c3,
    create_main_tree_c4,
)
from tools import norm


def tree_args():
    return Namespace(step=1.0, r_ori=0.5, ratioE=0.065)


def test_norm_returns_unit_vector():
    result = norm(3, 4, 0)
    np.testing.assert_allclose(result, [0.6, 0.8, 0.0])


@pytest.mark.parametrize(
    "builder",
    [create_main_tree_c1, create_main_tree_c2, create_main_tree_c3, create_main_tree_c4],
)
def test_each_tree_template_builds_valid_segments(builder):
    segments = []
    builder(tree_args(), segments)

    assert segments
    for segment in segments:
        assert np.isfinite(segment.point_in.position).all()
        assert np.isfinite(segment.point_out.position).all()
        assert segment.point_out.radius > 0


def test_bezier_curve_generation():
    for nodes in generate_parameters():
        curve = get_bezier(nodes[0], nodes[2], s=0.0)
        points = curve.evaluate_multi(np.linspace(0.0, 1.0, 11))
        assert points.shape == (2, 11)
        assert np.isfinite(points).all()


@pytest.mark.parametrize("case", ["c1", "c2", "c3", "c4"])
def test_crop_produces_expected_image_size(tmp_path, case):
    source = np.full((616, 542, 3), 255, dtype=np.uint8)
    cv2.line(source, (50, 50), (490, 560), (0, 0, 0), 5)
    source_path = tmp_path / "source.png"
    output_path = tmp_path / f"{case}.png"
    assert cv2.imwrite(str(source_path), source)

    crop(str(source_path), case, str(output_path))

    result = cv2.imread(str(output_path))
    assert result is not None
    assert result.shape == (400, 400, 3)
