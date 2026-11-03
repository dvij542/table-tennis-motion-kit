"""Focused tests for the forehand reference generator."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_table_tennis_forehand_bundle import RIGHT, smooth_curve, natural_racket_rotations, load_robot, generate
from preview_table_tennis_forehand import table_meshes, racket_meshes


def test_only_right_arm_is_optimized():
    assert len(RIGHT) == 7
    assert all(name.startswith("right_") for name in RIGHT)
    assert RIGHT[0] == "right_shoulder_pitch_joint"
    assert RIGHT[-1] == "right_wrist_yaw_joint"


def test_contact_is_not_a_stopping_waypoint():
    curve = smooth_curve([.7, 1., 1.3], [[.12, -.43, .94], [.43, -.2, 1.02], [.4, .08, 1.16]],
                         [[0, 0, 0], [.9, .6, .3], [0, 0, 0]])
    np.testing.assert_allclose(curve(1.), [.43, -.2, 1.02])
    np.testing.assert_allclose(curve.derivative()(1.), [.9, .6, .3])
    np.testing.assert_allclose(curve.derivative(2)(1.), [0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(curve.derivative()(.7), [0, 0, 0])
    np.testing.assert_allclose(curve.derivative()(1.3), [0, 0, 0])


def test_table_top_and_center_alignment():
    table = dict(length_m=2.74, width_m=1.525, height_m=.76, near_edge_x_m=.36,
                 center_y_m=0., thickness_m=.025, net_height_m=.1525)
    parts = table_meshes(table)
    top = parts[0][1]
    np.testing.assert_allclose(top.bounds, [[.36, -.7625, .735], [3.1, .7625, .76]])


def test_racket_plane_is_perpendicular_to_local_x():
    racket = dict(center_offset_local=[.16, 0, .11], blade_half_width_m=.075,
                  blade_half_length_m=.085)
    blade = racket_meshes(racket)[0][1]
    np.testing.assert_allclose(blade.extents, [.012, .15, .17], atol=1e-10)


def test_rotated_racket_mount_applies_to_blade_and_handle():
    mount = Rotation.from_euler("z", -30, degrees=True).as_matrix()
    center = np.array([.19, 0., .11])
    parts = racket_meshes(dict(center_offset_local=center, blade_half_width_m=.075,
                              blade_half_length_m=.085, blade_rotation_local=mount))
    blade = parts[0][1]
    local_vertices = (blade.vertices-center) @ mount
    np.testing.assert_allclose(np.ptp(local_vertices, axis=0), [.012, .15, .17], atol=1e-10)
    np.testing.assert_allclose(parts[-1][1].centroid, center-mount @ [0, 0, .12], atol=1e-10)
    red, black = parts[1][1], parts[2][1]
    np.testing.assert_allclose(red.centroid-black.centroid, .013*mount[:, 0], atol=1e-10)


def test_forehand_head_is_outward_and_legacy_head_is_inward():
    knots = [0., .45, .70, 1., 1.3, 1.5, 2.5]
    for forehand, mount_angle, expected_y_sign in ((True, 30, -1), (False, -30, 1)):
        mount = Rotation.from_euler("z", mount_angle, degrees=True).as_matrix()
        wrist = natural_racket_rotations(np.array([1.]), knots, np.array([0.]), mount, forehand)[0]
        np.testing.assert_allclose(wrist @ mount[:, 0], [1, 0, 0], atol=1e-10)
        head_from_grip = wrist @ (.11*mount[:, 2])
        assert expected_y_sign*head_from_grip[1] > .09
        assert head_from_grip[2] > 0


def test_integrated_racket_is_not_drawn_twice():
    assert racket_meshes({"embedded_in_urdf": True}) == []


def test_robot_loader_resolves_meshes_relative_to_urdf(tmp_path):
    import trimesh

    (tmp_path / "meshes").mkdir()
    trimesh.creation.box().export(tmp_path / "meshes/box.stl")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"><visual><geometry>'
                    '<mesh filename="meshes/box.stl"/></geometry></visual></link></robot>')
    robot = load_robot(urdf)
    assert len(robot.scene.geometry) == 1
    np.testing.assert_allclose(robot.scene.bounds, [[-.5]*3, [.5]*3])


@pytest.mark.parametrize("offset", [(0., 0.), (0., float("nan"), 0.)])
def test_target_offset_validation_precedes_asset_loading(tmp_path, offset):
    with pytest.raises(ValueError, match="finite XYZ"):
        generate(tmp_path / "output", tmp_path / "missing.urdf", target_offset=offset)


def test_grid_packs_unique_keys_and_complete_offset_range(tmp_path, monkeypatch):
    import json
    import joblib
    import build_table_tennis_forehand_grid as grid

    urdf = tmp_path / "robot.urdf"
    urdf.write_text("test fixture")
    seen = []

    def fake_generate(directory, urdf, **kwargs):
        offset = kwargs["target_offset"]
        seen.append(offset)
        directory.mkdir()
        meta = {"target_hit_position_world": (np.array([.43, -.38, .86])+offset).tolist()}
        joblib.dump({grid.KEY: {"offset": offset}}, directory / "processed_robot_motions.pkl")
        joblib.dump({grid.KEY: meta}, directory / "motion_meta_bundle.pkl")

    monkeypatch.setattr(grid, "generate", fake_generate)
    output = tmp_path / "grid"
    grid.build(output, urdf)
    manifest = json.loads((output / "motion_grid.json").read_text())
    assert manifest["complete"]
    assert len(set(seen)) == 20
    assert seen[0] == (0., 0., 0.)
    assert set(y for _, y, _ in seen) == {-.15, -.10, -.05, 0., .05}
    assert set(z for _, _, z in seen) == {0., .05, .10, .15}
    assert len(joblib.load(output / "processed_robot_motions.pkl")) == 20
