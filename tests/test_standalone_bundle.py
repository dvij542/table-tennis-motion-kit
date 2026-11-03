import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_table_tennis_forehand_bundle import DEFAULT_URDF, load_robot
from preview_table_tennis_forehand import load
from validate_bundle import validate


def test_assets_and_examples_match_manifest():
    provenance = json.loads((ROOT/'ASSET_PROVENANCE.json').read_text())
    for group in ('asset_files', 'example_files'):
        for name, expected in provenance[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name


def test_real_racket_mount():
    robot = load_robot(DEFAULT_URDF)
    assert len(robot.actuated_joint_names) == 29
    transform = robot.get_transform('right_racket', 'right_wrist_yaw_link')
    np.testing.assert_allclose(transform[:3, 3], [.2365, 0, 0])
    np.testing.assert_allclose(transform[:3, :3], np.eye(3))
    assert len(robot.scene.geometry) > 20


def test_examples_load_outside_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    meta, data, robot = load(ROOT/'examples/forehand_grid/table_tennis_forehand_y+000_z000')
    assert not Path(meta['urdf']).is_absolute()
    assert len(data['time']) == 126
    data.close()


def test_all_motion_arrays_and_forward_kinematics():
    assert validate(ROOT/'examples/forehand_grid')['motions'] == 20
