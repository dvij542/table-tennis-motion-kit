"""Recompute FK and frozen-body checks for a trusted saved bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

from build_table_tennis_forehand_bundle import ROOT, NAMES, WRIST
from motion_format import body_joint_axes, compute_pose_aa
from preview_table_tennis_forehand import load


def validate(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'motion_grid.json').read_text())
    assert manifest['complete'] and not manifest['failures']
    motions = joblib.load(bundle / 'processed_robot_motions.pkl')
    metas = joblib.load(bundle / 'motion_meta_bundle.pkl')
    assert set(motions) == set(metas) == {e['key'] for e in manifest['entries']}
    assert len(motions) == 20
    assert {tuple(e['offset_m']) for e in manifest['entries']} == {
        (0., y, z) for y in [-.15, -.10, -.05, 0., .05] for z in [0., .05, .10, .15]}
    worst_pos, worst_rot = 0., 0.
    for entry in manifest['entries']:
        directory = bundle / entry['directory']
        meta, data, robot = load(directory)
        asset = (directory / meta['urdf']).resolve()
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == meta['urdf_sha256']
        assert (bundle / metas[entry['key']]['urdf']).resolve() == asset
        for name in data.files:
            if np.issubdtype(data[name].dtype, np.number):
                assert np.isfinite(data[name]).all(), name
        np.testing.assert_allclose(data['time'], np.arange(126)/50, atol=1e-12)
        assert meta['hit_idx'] == 50 and meta['hit_time_s'] == 1.
        dof = data['dof']
        np.testing.assert_array_equal(dof, motions[entry['key']]['dof'])
        np.testing.assert_array_equal(data['root_pos'], motions[entry['key']]['root_trans_offset'])
        np.testing.assert_array_equal(data['root_wxyz'], motions[entry['key']]['root_rot'])
        np.testing.assert_allclose(motions[entry['key']]['pose_aa'],
                                   compute_pose_aa(dof, data['root_wxyz'], body_joint_axes(robot)))
        for key in ('root_pos', 'root_wxyz'):
            assert np.max(np.ptp(data[key], axis=0)) == 0
        assert np.max(np.ptp(dof[:, :12], axis=0)) == 0
        assert np.max(np.ptp(dof[:, 15:22], axis=0)) == 0
        np.testing.assert_array_equal(dof[0], dof[-1])
        assert dof[:, NAMES.index('right_shoulder_roll_joint')].max() <= -.3+1e-7
        for i, name in enumerate(NAMES):
            limit = robot.joint_map[name].limit
            assert np.all((dof[:, i] >= limit.lower) & (dof[:, i] <= limit.upper))
        contacts, rotations = [], []
        for f, cfg in enumerate(data['cfg_urdf']):
            robot.update_cfg(cfg)
            tf = robot.get_transform(WRIST)
            contacts.append(data['root_pos'][f]+tf[:3, 3]+tf[:3, :3] @ meta['racket']['contact_offset_local'])
            rotations.append(tf[:3, :3])
        contacts, rotations = np.array(contacts), np.array(rotations)
        np.testing.assert_allclose(contacts, data['actual_contact_pos'], atol=1e-7)
        pos_error = float(np.linalg.norm(contacts-data['target_contact_pos'], axis=1).max())
        rot_error = float(np.degrees(Rotation.from_matrix(
            rotations @ data['target_wrist_rotation'].transpose(0, 2, 1)).magnitude()).max())
        assert pos_error < .002 and rot_error < .5, (entry['key'], pos_error, rot_error)
        target = np.array([.43, -.38, .86])+entry['offset_m']
        np.testing.assert_allclose(data['target_contact_pos'][50], target, atol=1e-10)
        assert abs(contacts[50, 2]-target[2]) < .001
        normal = rotations[50] @ meta['racket']['face_normal_local']
        assert np.degrees(np.arcsin(abs(normal[2]))) < .5
        worst_pos, worst_rot = max(worst_pos, pos_error), max(worst_rot, rot_error)
        data.close()
    result = dict(motions=len(motions), frames=len(motions)*126,
                  max_position_error_m=worst_pos, max_orientation_error_deg=worst_rot)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, default=ROOT/'examples/forehand_grid')
    args = parser.parse_args()
    validate(args.bundle_dir)
