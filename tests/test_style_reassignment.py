from pathlib import Path

import numpy as np

from style_reassignment import (
    audit_assignments,
    create_assignments,
    sample_latent_pool,
    scan_explicit_identities,
)


def test_latent_sampling_matches_legacy_formula():
    centers, vectors = sample_latent_pool(2, 7, 8, 2026, 0.5)
    rng = np.random.RandomState(2026)
    expected_center = rng.normal(size=(1, 8))
    expected_center /= np.linalg.norm(expected_center)
    directions = rng.normal(size=(7, 8))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    expected = expected_center + 0.5 * directions
    np.testing.assert_array_equal(centers[0], expected_center[0].astype(np.float32))
    np.testing.assert_array_equal(vectors[:7], expected.astype(np.float32))


def test_balanced_reassignment_constraints(tmp_path):
    input_root = tmp_path / "patterns"
    grouped = {}
    identities = [f"c{case}_{index}" for case in range(1, 5) for index in range(4)]
    for identity in identities:
        grouped[identity] = [input_root / identity / f"{identity}_sample{sample}.png" for sample in range(7)]
    _, vectors = sample_latent_pool(len(identities), 7, 8, 2026, 0.5)
    a0, c1, _ = create_assignments(grouped, input_root, vectors, 2027)
    audit = audit_assignments(a0, c1, len(vectors))
    assert audit["constraints_passed"]
    assert audit["C1"]["distinct_source_centers_per_target"] == [7]
    assert audit["C1"]["distinct_targets_per_source_center"] == [7]
    assert {row["z_sha256"] for row in a0} == {row["z_sha256"] for row in c1}


def test_scan_uses_explicit_identity_directories(tmp_path):
    for identity in ("c1_0", "c2_0"):
        directory = tmp_path / identity
        directory.mkdir()
        for sample in range(7):
            (directory / f"{identity}_sample{sample}.png").write_bytes(b"not decoded here")
    grouped = scan_explicit_identities(tmp_path)
    assert list(grouped) == ["c1_0", "c2_0"]
    assert all(len(paths) == 7 for paths in grouped.values())
