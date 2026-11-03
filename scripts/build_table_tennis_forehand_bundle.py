"""Generate a stationary G1 right-hand forehand reference (not dynamics)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import least_squares
from scipy.sparse import block_diag, csr_matrix, diags, eye, kron, vstack
from scipy.spatial.transform import Rotation
import yourdfpy

import motion_format as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "assets/vicar/g1.urdf"
NAMES = base.BM_BODY_JOINT_NAMES
RIGHT = [name for name in NAMES if name.startswith("right_")][6:]
WRIST = "right_wrist_yaw_link"
# Nominal striking/grip center, not a calibrated racket attachment.
HAND_OFFSET = np.array([0.16, 0.0, 0.0])
RACKET_CENTER_OFFSET = HAND_OFFSET + np.array([0., 0., .11])
RACKET_FACE_OFFSET = RACKET_CENTER_OFFSET + np.array([.006, 0., 0.])
KEY = "table_tennis_right_forehand_0000"


def load_robot(path, load_meshes=True):
    path = Path(path).resolve()

    def filename_handler(fname):
        if fname.startswith("package://"):
            raise ValueError("Use URDF-relative mesh filenames in this standalone package")
        resolved = Path(fname)
        return str(resolved if resolved.is_absolute() else path.parent / resolved)

    return yourdfpy.URDF.load(path, load_meshes=load_meshes, filename_handler=filename_handler)


def smooth_curve(times, values, derivatives=None):
    values = np.asarray(values, dtype=float)
    if derivatives is None:
        derivatives = np.zeros_like(values)
    return BPoly.from_derivatives(times, [[v, d, np.zeros_like(v)] for v, d in zip(values, derivatives)])


def natural_racket_rotations(times, knots, yaw, mount, forehand=False):
    blade_roll = (smooth_curve(knots, np.radians([65, 65, 65, 65, 50, 45, 65]))(times)
                  if forehand else np.full(len(times), np.radians(-80)))
    return Rotation.from_euler("xyz", np.column_stack([
        blade_roll, np.zeros(len(times)), yaw
    ])).as_matrix() @ mount.T


def seed_configuration(urdf):
    cfg = {name: 0.0 for name in urdf.actuated_joint_names}
    for side, sign in (("left", 1), ("right", -1)):
        cfg.update({
            f"{side}_hip_pitch_joint": -0.18,
            f"{side}_knee_joint": 0.36,
            f"{side}_ankle_pitch_joint": -0.18,
            f"{side}_shoulder_pitch_joint": -0.1,
            f"{side}_shoulder_roll_joint": sign * 0.22,
            f"{side}_elbow_joint": 1.15,
        })
        for finger in (1, 2):
            name = f"{side}_yam_Slider_{finger}"
            if name in urdf.joint_map:
                cfg[name] = urdf.joint_map[name].limit.lower
    if "neck_link3_joint" in cfg:
        cfg["neck_link3_joint"] = np.pi / 4
    return cfg


def refine_racket_sequence(urdf, configurations, roots, targets, rotations, offset, lower, upper,
                          full_orientation=False, orientation_weight=10., max_nfev=120, position_weight=100.):
    """Jointly smooth the constrained IK sequence without moving its targets."""
    frames = len(configurations)
    indices = [urdf.actuated_joint_names.index(n) for n in RIGHT]
    configs = np.array(configurations)
    initial = configs[:, indices].copy()
    variables = frames-1 if full_orientation else frames
    # Share the ready-pose variables at both ends, rather than accepting an IK
    # null-space jump when this reference is looped in the viewer.
    lift = csr_matrix((np.ones(frames*7), (np.arange(frames*7),
                      np.arange(frames*7) % (variables*7))), shape=(frames*7, variables*7))
    velocity_jac = .3*kron(diags([-1., 1.], [0, 1], shape=(frames-1, frames)), eye(7))
    acceleration_weight = 6. if full_orientation else 2.
    acceleration_jac = acceleration_weight*kron(diags([1., -2., 1.], [0, 1, 2], shape=(frames-2, frames)), eye(7))

    def residual(flat):
        angles = (lift @ flat).reshape(frames, 7)
        result = []
        for f in range(frames):
            configs[f, indices] = angles[f]
            urdf.update_cfg(configs[f])
            tf = urdf.get_transform(WRIST)
            point = roots + tf[:3, 3] + tf[:3, :3] @ offset
            result.extend(position_weight*(point-targets[f]))
            if full_orientation:
                result.extend((orientation_weight*(tf[:3, :3]-rotations[f])).T.ravel())
            else:
                result.extend(orientation_weight*(tf[:3, 0]-rotations[f, :, 0]))
            result.extend(.015*(angles[f]-initial[f]))
        return np.r_[result, (.3*np.diff(angles, axis=0)).ravel(),
                     (acceleration_weight*np.diff(angles, n=2, axis=0)).ravel()]

    def jacobian(flat):
        angles = (lift @ flat).reshape(frames, 7)
        blocks = []
        for f in range(frames):
            configs[f, indices] = angles[f]
            urdf.update_cfg(configs[f])
            tf = urdf.get_transform(WRIST)
            point = roots + tf[:3, 3] + tf[:3, :3] @ offset
            origins, axes = [], []
            for name in RIGHT:
                joint = urdf.joint_map[name]
                joint_tf = urdf.get_transform(joint.child)
                origins.append(roots + joint_tf[:3, 3])
                axes.append(joint_tf[:3, :3] @ joint.axis)
            orientation_jac = np.vstack([orientation_weight*np.cross(axes, tf[:3, k]).T
                                         for k in range(3 if full_orientation else 1)])
            blocks.append(np.vstack([position_weight*np.cross(axes, point-np.array(origins)).T,
                                     orientation_jac, .015*np.eye(7)]))
        return vstack([block_diag(blocks), velocity_jac, acceleration_jac], format="csr") @ lift

    initial_flat = initial[:variables].ravel()
    direction = np.random.default_rng(0).normal(size=initial_flat.size)
    eps = 1e-5
    numeric = (residual(initial_flat+eps*direction)-residual(initial_flat-eps*direction))/(2*eps)
    np.testing.assert_allclose(jacobian(initial_flat)@direction, numeric, atol=1e-5, rtol=1e-5)

    fit = least_squares(residual, initial_flat, bounds=(np.tile(lower, variables), np.tile(upper, variables)),
                        jac=jacobian, max_nfev=max_nfev,
                        ftol=1e-5, xtol=1e-5, gtol=1e-5, verbose=1)
    if not fit.success:
        raise RuntimeError(f"Sequence smoothing failed: {fit.message}")
    return (lift @ fit.x).reshape(frames, 7)


def generate(output: Path, urdf_path: Path, fps=50, table_scene=False, right_shoulder_roll_max=None,
             racket_normal_only=False, natural=False, forehand=False, target_offset=(0., 0., 0.)):
    natural = natural or forehand
    target_offset = np.asarray(target_offset, dtype=float)
    if target_offset.shape != (3,) or not np.isfinite(target_offset).all():
        raise ValueError("Target offset must be a finite XYZ vector in metres")
    position_weight = 250. if np.any(target_offset) else 100.
    if target_offset[1] < -.149 and abs(target_offset[2]) < .001:
        position_weight = 300.
    if fps <= 0 or fps % 2:
        raise ValueError("Use a positive even integer frame rate")
    if racket_normal_only and not table_scene:
        raise ValueError("Racket face constraints require --table-scene")
    if natural and (not table_scene or racket_normal_only):
        raise ValueError("Natural forehand requires --table-scene and full racket orientation")
    hand_offset = np.array([.19, 0., 0.]) if natural else HAND_OFFSET
    mount = Rotation.from_euler("z", -30, degrees=True).as_matrix() if natural else np.eye(3)
    if forehand:
        mount = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    racket_center = hand_offset + .11*mount[:, 2]
    racket_contact = racket_center + .006*mount[:, 0]
    urdf = load_robot(urdf_path)
    embedded_racket = "right_racket" in urdf.link_map
    if embedded_racket:
        if not (forehand and table_scene):
            raise ValueError("The integrated VICAR racket currently requires --forehand --table-scene")
        # VICAR's racket shaft is +X and its face is in XZ. Preserve its
        # fixed joint exactly, rather than treating it as the YAM attachment.
        attachment = urdf.get_transform("right_racket", WRIST)
        mount = attachment[:3, :3] @ np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        racket_center = attachment[:3, 3]
        racket_contact = racket_center + .0062*mount[:, 0]
        hand_offset = racket_center - .12*mount[:, 2]
    cfg = seed_configuration(urdf)
    grip_width = 0.
    if natural and not embedded_racket:
        # The visual pads close along wrist Y. Leave room for the rigid wooden
        # handle; the hand-action channel still commands a closed holding grip.
        grip_width = float(np.abs(mount[1]) @ np.array([.023, .028, .10]))
        for finger in (1, 2):
            name = f"right_yam_Slider_{finger}"
            cfg[name] = urdf.joint_map[name].limit.lower + grip_width/2
    urdf.update_cfg(cfg)
    # Put the actual visual sole on z=0, without changing root/legs during motion.
    sole_z = []
    for node in urdf.scene.graph.nodes_geometry:
        if "ankle_roll" in node:
            transform, mesh = urdf.scene.graph[node]
            vertices = urdf.scene.geometry[mesh].vertices
            sole_z.append((vertices @ transform[:3, :3].T + transform[:3, 3])[:, 2].min())
    if len(sole_z) != 2:
        raise ValueError(f"Expected two foot visuals, got {len(sole_z)}")
    root = np.array([0., 0., -min(sole_z)])
    # Meshes were needed for sole-ground alignment, not for subsequent IK.
    urdf = load_robot(urdf_path, load_meshes=False)
    times = np.arange(round(2.5 * fps) + 1) / fps
    knots = [0., .45, .70, 1., 1.30, 1.55, 2.5]
    points = np.array([[.28, -.28, 1.00], [.12, -.43, .94], [.12, -.43, .94],
                       [.43, -.20, 1.02], [.40, .08, 1.16], [.40, .08, 1.16], [.28, -.28, 1.00]])
    if table_scene:
        points[:, 2] = [.92, .88, .88, .86, 1.02, 1.02, .92]
    velocities = np.zeros_like(points)
    velocities[3] = [.9, .6, .3]
    if natural:
        # Compact backswing and continuous loading avoid the wide, straight-arm
        # sweep forced by the original Cartesian targets.
        points = np.array([[.32, -.18, .95], [.26, -.27, .92], [.26, -.30, .87],
                           [.43, -.20, .86], [.44, -.03, .99], [.40, -.02, 1.02],
                           [.32, -.18, .95]])
        velocities[[1, 2, 4, 5]] = [[-.15, -.15, -.12], [.08, 0., 0.],
                                    [0., .10, .20], [-.10, -.05, 0.]]
    if forehand:
        points = np.array([[.32, -.35, .97], [.22, -.45, .93], [.22, -.48, .90],
                           [.43, -.38, .86], [.46, -.22, .97], [.44, -.22, .99],
                           [.32, -.35, .97]])
        knots[5] = 1.5
        velocities[5] = 0.
        velocities[3, 2] = .15
    if embedded_racket and np.any(target_offset):
        # Vary contact location while retaining the ready/settled poses. A
        # far-outside contact needs a more lateral, less forward swing to keep
        # its post-impact arc inside the stationary arm's workspace.
        points += np.array([0., .5, .8, 1., .5, 0., 0.])[:, None]*target_offset
        reach = np.clip(-target_offset[1]/.15, 0., 1.)
        velocities[3, 0] -= .2*reach
        velocities[3, 1] += .15*reach
    else:
        points += target_offset
    targets = smooth_curve(knots, points, velocities)(times)
    waist = smooth_curve(knots, np.radians([0, -8, -8, 2, 9, 9, 0]))(times)
    if forehand:
        waist = smooth_curve(knots, np.radians([0, -12, -12, 6, 18, 16, 0]))(times)
    wrist_yaw = smooth_curve(knots, np.radians([-10, -35, -35, 10, 40, 40, -10]))(times)
    rotations = Rotation.from_euler("xyz", np.column_stack([
        np.full(len(times), np.radians(-15)), np.full(len(times), np.radians(5)), wrist_yaw
    ])).as_matrix()
    contact_offset = HAND_OFFSET
    if table_scene:
        contact_offset = racket_contact
        yaw = smooth_curve(knots, np.radians([-10, -25, -25, 0, 30, 30, -10]))(times)
        # Blade lies in wrist YZ, with its handle along wrist Z. Roll 90 degrees
        # puts the handle sideways: the gripper stays above the table at impact.
        rotations = Rotation.from_euler("xyz", np.column_stack([
            np.full(len(times), np.pi/2), np.zeros(len(times)), yaw
        ])).as_matrix()
        if natural:
            rotations = natural_racket_rotations(times, knots, yaw, mount, forehand)
        if embedded_racket:
            # The inline racket needs its shaft outward and forward, keeping
            # the wrist within reach while the face remains vertical at impact.
            yaw = smooth_curve(knots, np.radians([45, 35, 35, 45, 60, 60, 45]))(times)
            reach_tilt = 20.*np.clip(-target_offset[1]/.15, 0., 1.)
            blade_roll = np.radians(90. + smooth_curve(knots, [0, 0, .5, 1, 0, 0, 0])(times)*reach_tilt)
            rotations = Rotation.from_euler("xyz", np.column_stack([
                blade_roll, np.zeros(len(times)), yaw
            ])).as_matrix() @ mount.T
    lower = np.array([urdf.joint_map[n].limit.lower for n in RIGHT]) + .01
    upper = np.array([urdf.joint_map[n].limit.upper for n in RIGHT]) - .01
    lower[3], upper[3] = -.9, 1.85
    upper[1] = .15
    lower[4:], upper[4:] = [-.5, -.4, -.6], [.5, .4, .6]
    if table_scene:
        lower[4:], upper[4:] = [-1.85, -1.4, -1.4], [1.85, 1.4, 1.4]
        upper[1] = .5
    if natural:
        lower[1], upper[1] = -1.1, -.3
        lower[4:], upper[4:] = [-1.1, -1., -1.1], [1.1, 1., 1.1]
    if forehand:
        lower[4:], upper[4:] = [-1.85, -1.3, -1.3], [1.85, 1.3, 1.3]
    if embedded_racket and np.any(target_offset):
        lower[1] = urdf.joint_map[RIGHT[1]].limit.lower+.01
        lower[4:] = [urdf.joint_map[n].limit.lower+.01 for n in RIGHT[4:]]
        upper[4:] = [urdf.joint_map[n].limit.upper-.01 for n in RIGHT[4:]]
    if right_shoulder_roll_max is not None:
        if not np.isfinite(right_shoulder_roll_max) or right_shoulder_roll_max <= lower[1]:
            raise ValueError("Shoulder-roll upper limit must be finite and exceed its lower limit")
        upper[1] = min(upper[1], right_shoulder_roll_max)
    lower = np.maximum(lower, [urdf.joint_map[n].limit.lower+.01 for n in RIGHT])
    upper = np.minimum(upper, [urdf.joint_map[n].limit.upper-.01 for n in RIGHT])
    q = np.array([-.6, -.3, 0., -.2, 0., 0., 0.])
    if table_scene:
        q = np.array([-.6, -.3, 0., .7, 1.4, 0., 0.])
    if racket_normal_only:
        q = np.array([-.9, -1.12, 1.36, .055, -1.84, .428, .678])
    if natural:
        q = np.array([-.41, -.87, .17, .75, -.55, -.35, .37])
    if forehand:
        q = np.array([-.31, -.31, -.28, .38, 1.34, .22, -.42])
    if embedded_racket:
        q = np.array([-.25, -.49, .22, .62, .58, -.39, -.83])
    q = np.clip(q, lower + 1e-6, upper - 1e-6)
    nominal = q.copy()
    dof, actual, left, wrists, full_cfg, wrist_pos, hand_pos = [[None]*len(times) for _ in range(7)]
    orientation_error = [None]*len(times)
    frame_order = list(range(len(times)))
    if racket_normal_only or natural:
        # Solve the critical contact first, then continue in both directions
        # to avoid choosing an unreachable branch at the initial ready pose.
        frame_order = list(range(fps, -1, -1)) + list(range(fps+1, len(times)))
    recovery_start = round(1.5*fps)
    if forehand:
        frame_order = [frame for frame in frame_order if frame <= recovery_start]
    hit_solution = None
    for frame in frame_order:
        target, rotation = targets[frame], rotations[frame]
        if (racket_normal_only or natural) and frame == fps+1:
            q = hit_solution.copy()
        cfg["waist_yaw_joint"] = waist[frame]
        previous = q.copy()

        def residual(candidate):
            cfg.update(zip(RIGHT, candidate))
            urdf.update_cfg(cfg)
            tf = urdf.get_transform(WRIST)
            point = root + tf[:3, 3] + tf[:3, :3] @ contact_offset
            error_r = Rotation.from_matrix(rotation.T @ tf[:3, :3]).as_rotvec()
            orientation_residual = (10 if table_scene else .02) * error_r
            if racket_normal_only:
                orientation_residual = np.r_[10*(tf[:3, 0]-rotation[:, 0]), .02*error_r]
            return np.concatenate([(100 if table_scene else 10) * (point - target),
                                   orientation_residual,
                                   (.015 if table_scene else .06) * np.array([1, 1, 1, 1, 4, 4, 4]) * (candidate - nominal),
                                   (.4 if racket_normal_only else .15) * (candidate - previous)])

        fit = least_squares(residual, q, bounds=(lower, upper), max_nfev=400,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        if not fit.success:
            raise RuntimeError(f"IK did not converge at frame {frame}: {fit.message}")
        q = fit.x
        if frame == fps:
            hit_solution = q.copy()
        residual(q)
        tf = urdf.get_transform(WRIST)
        actual[frame] = root + tf[:3, 3] + tf[:3, :3] @ contact_offset
        wrist_pos[frame] = root + tf[:3, 3]
        hand_pos[frame] = root + tf[:3, 3] + tf[:3, :3] @ hand_offset
        wrists[frame] = tf[:3, :3].copy()
        lt = urdf.get_transform("left_wrist_yaw_link")
        left[frame] = root + lt[:3, 3] + lt[:3, :3] @ HAND_OFFSET
        orientation_error[frame] = np.linalg.norm(Rotation.from_matrix(rotation.T @ tf[:3, :3]).as_rotvec())
        dof[frame] = [cfg[n] for n in NAMES]
        full_cfg[frame] = [cfg[n] for n in urdf.actuated_joint_names]
    if forehand:
        # Return in joint space after settling. A straight Cartesian return with
        # a fixed paddle attitude needlessly drives the elbow through its bound.
        indices = [urdf.actuated_joint_names.index(n) for n in RIGHT]
        ready = np.array(full_cfg[0])[indices]
        finish = np.array(full_cfg[recovery_start])[indices]
        for frame in range(recovery_start+1, len(times)):
            u = (frame-recovery_start)/(len(times)-1-recovery_start)
            blend = u**3*(10-15*u+6*u*u)
            cfg.update(zip(RIGHT, finish+(ready-finish)*blend))
            cfg["waist_yaw_joint"] = waist[frame]
            urdf.update_cfg(cfg)
            tf = urdf.get_transform(WRIST)
            targets[frame] = root+tf[:3, 3]+tf[:3, :3]@contact_offset
            rotations[frame] = tf[:3, :3]
            lt = urdf.get_transform("left_wrist_yaw_link")
            left[frame] = root+lt[:3, 3]+lt[:3, :3]@HAND_OFFSET
            full_cfg[frame] = [cfg[n] for n in urdf.actuated_joint_names]
    if racket_normal_only or natural:
        print("Refining constrained sequence jointly...", flush=True)
        refined = refine_racket_sequence(urdf, full_cfg, root, targets, rotations, contact_offset, lower, upper,
                                        full_orientation=natural, orientation_weight=20. if embedded_racket else 10.,
                                        max_nfev=400 if np.any(target_offset) else 120,
                                        position_weight=position_weight)
        for frame in range(len(times)):
            cfg = dict(zip(urdf.actuated_joint_names, full_cfg[frame]))
            cfg.update(zip(RIGHT, refined[frame]))
            urdf.update_cfg(cfg)
            tf = urdf.get_transform(WRIST)
            actual[frame] = root + tf[:3, 3] + tf[:3, :3] @ contact_offset
            wrist_pos[frame] = root + tf[:3, 3]
            hand_pos[frame] = root + tf[:3, 3] + tf[:3, :3] @ hand_offset
            wrists[frame] = tf[:3, :3].copy()
            orientation_error[frame] = np.linalg.norm(Rotation.from_matrix(rotations[frame].T @ tf[:3, :3]).as_rotvec())
            dof[frame] = [cfg[n] for n in NAMES]
            full_cfg[frame] = [cfg[n] for n in urdf.actuated_joint_names]
    dof = np.asarray(dof, dtype=np.float32)
    actual = np.asarray(actual)
    roots = np.tile(root, (len(times), 1)).astype(np.float32)
    quats = np.tile([1., 0., 0., 0.], (len(times), 1)).astype(np.float32)
    robot = dict(dof=dof, root_trans_offset=roots, root_rot=quats,
                 pose_aa=base.compute_pose_aa(dof, quats, base.body_joint_axes(urdf)),
                 smpl_joints=np.zeros((len(times), 24, 3), dtype=np.float32),
                 hand_action_left=np.ones(len(times), dtype=np.float32),
                 hand_action_right=np.ones(len(times), dtype=np.float32), fps=float(fps))
    # Existing synthetic bundles use +1 for closed, -1 for open.
    meta = dict(task="table_tennis_right_forehand", hit_idx=fps, hit_time_s=1.0,
                duration_s=2.5, fps=fps, joint_names=NAMES,
                urdf=os.path.relpath(urdf_path.resolve(), output.resolve()),
                urdf_path_base="metadata_directory",
                urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                axes="+X forward, +Y left, +Z up", wrist_link=WRIST,
                hand_center_offset_local=hand_offset.tolist(),
                natural_forehand=natural,
                stroke_style="right_forehand_outward_head" if forehand else "legacy_right_stroke",
                recovery_mode="joint_space_quintic_return" if forehand else "cartesian_targets",
                visual_right_gripper_gap_m=grip_width,
                target_knots_s=knots, target_knots_xyz=points.tolist(),
                target_offset_world_m=target_offset.tolist(),
                refinement_position_weight=position_weight,
                target_knot_velocities=velocities.tolist(),
                hit_position_world=actual[fps].tolist(), target_hit_position_world=points[3].tolist(),
                nominal_hit_velocity_m_s=velocities[3].tolist(),
                lower_body_frozen=True, pelvis_frozen=True, left_arm_joints_frozen=True,
                right_shoulder_roll_upper_limit_rad=float(upper[1]),
                optimized_joint_bounds_rad={name: [float(lo), float(hi)] for name, lo, hi in zip(RIGHT, lower, upper)},
                racket_orientation_constraint="face_normal_with_free_in_plane_rotation" if racket_normal_only else "full_rotation",
                phases_s={"backswing": [0, .45], "loaded": [.45, .70],
                          "forward_swing": [.70, 1.0], "follow_through": [1., 1.3],
                          "settle": [1.3, 1.55], "recover": [1.55, 2.5]},
                limitations="Kinematic hand reference only: no racket calibration, ball dynamics or dynamic balance certification. smpl_joints is an unused zero placeholder.")
    if forehand:
        meta["phases_s"]["settle"] = [1.3, 1.5]
        meta["phases_s"]["recover"] = [1.5, 2.5]
    checks = dict(
        frames=len(times), hit_frame=fps, max_position_error_m=float(np.linalg.norm(actual-targets, axis=1).max()),
        max_position_error_frame=int(np.argmax(np.linalg.norm(actual-targets, axis=1))),
        max_orientation_error_deg=float(np.degrees(max(orientation_error))),
        max_orientation_error_frame=int(np.argmax(orientation_error)),
        leg_joint_max_variation_rad=float(np.ptp(dof[:, :12], axis=0).max()),
        left_joint_max_variation_rad=float(np.ptp(dof[:, 15:22], axis=0).max()),
        left_hand_max_displacement_m=float(np.linalg.norm(np.array(left)-left[0], axis=1).max()),
        hand_velocity_at_hit_m_s=np.gradient(actual, 1/fps, axis=0)[fps].tolist(),
        max_joint_speed_rad_s=float(np.abs(np.gradient(dof, 1/fps, axis=0)).max()),
        max_joint_acceleration_rad_s2=float(np.abs(np.gradient(np.gradient(dof, 1/fps, axis=0), 1/fps, axis=0)).max()),
        right_shoulder_roll_range_rad=[float(dof[:, NAMES.index("right_shoulder_roll_joint")].min()),
                                       float(dof[:, NAMES.index("right_shoulder_roll_joint")].max())],
        loop_joint_difference_rad=float(np.max(np.abs(dof[-1]-dof[0]))),
    )
    if table_scene:
        normals = np.array(wrists) @ mount[:, 0]
        target_normals = rotations @ mount[:, 0]
        normal = normals[fps]
        checks.update(hit_height_above_table_m=float(actual[fps, 2]-.76),
                      racket_vertical_error_deg=float(np.degrees(np.arcsin(abs(normal[2])))),
                      max_face_normal_error_deg=float(np.degrees(np.arccos(np.clip(
                          np.sum(normals*target_normals, axis=1), -1., 1.))).max()),
                      racket_normal_at_hit=normal.tolist())
        assert abs(checks["hit_height_above_table_m"]-(.1+target_offset[2])) < .001, checks
        assert checks["racket_vertical_error_deg"] < .5, checks
        # Impact must be vertical; away from impact the face is a soft target.
        assert checks["max_face_normal_error_deg"] < (3. if racket_normal_only else .5), checks
        if natural:
            assert checks["loop_joint_difference_rad"] == 0., checks
            assert checks["max_orientation_error_deg"] < .5, checks
            assert checks["max_position_error_m"] < .002, checks
        meta.update(table=dict(length_m=2.74, width_m=1.525, height_m=.76, near_edge_x_m=.36,
                               center_y_m=0., thickness_m=.025, net_height_m=.1525),
                    racket=dict(center_offset_local=racket_center.tolist(),
                                contact_offset_local=racket_contact.tolist(),
                                blade_rotation_local=mount.tolist(),
                                face_normal_local=mount[:, 0].tolist(), blade_half_width_m=.075,
                                blade_half_length_m=.085, blade_thickness_m=.012,
                                approximate_mount=True),
                    contact_point="center of racket front face", pelvis_distance_to_table_end_m=.36,
                    limitations="Kinematic reference with approximate rigid racket mount, not calibrated hardware or ball dynamics. smpl_joints is an unused zero placeholder.")
        if embedded_racket:
            meta["racket_attitude"] = dict(yaw_knots_deg=[45, 35, 35, 45, 60, 60, 45],
                                           roll_knots_deg=(90.+np.array([0, 0, .5, 1, 0, 0, 0])*reach_tilt).tolist())
            meta["racket"].update(embedded_in_urdf=True, link="right_racket", approximate_mount=False,
                                  contact_surface_offset_m=.0062,
                                  contact_definition="VICAR right_racket origin offset 6.2 mm along local +Y onto the mesh face")
            meta["limitations"] = ("VICAR website visual URDF: simplified meshes, original joint geometry, no inertial/collision elements. "
                                   "Kinematic preview only; not certified for dynamics or hardware. Face offset estimated from mesh. "
                                   "smpl_joints is an unused zero placeholder.")
    assert np.isfinite(dof).all()
    assert checks["right_shoulder_roll_range_rad"][1] <= upper[1] + 1e-7
    assert checks["leg_joint_max_variation_rad"] == checks["left_joint_max_variation_rad"] == 0
    assert checks["max_position_error_m"] < .01, checks
    for i, name in enumerate(NAMES):
        limit = urdf.joint_map[name].limit
        assert np.all((dof[:, i] >= limit.lower) & (dof[:, i] <= limit.upper)), name
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({KEY: robot}, output / "processed_robot_motions.pkl", compress=3)
    joblib.dump({KEY: meta}, output / "motion_meta_bundle.pkl", compress=3)
    np.savez_compressed(output / "reference.npz", time=times, dof=dof, root_pos=roots,
                        root_wxyz=quats, target_hand_pos=targets, actual_hand_pos=np.array(hand_pos),
                        target_contact_pos=targets, actual_contact_pos=actual,
                        target_wrist_rotation=rotations,
                        wrist_pos=np.array(wrist_pos), wrist_rotation=np.array(wrists), cfg_urdf=np.array(full_cfg),
                        joint_names=np.array(urdf.actuated_joint_names))
    (output / "metadata.json").write_text(json.dumps({**meta, "validation": checks}, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--table-scene", action="store_true")
    parser.add_argument("--right-shoulder-roll-max", type=float)
    parser.add_argument("--racket-normal-only", action="store_true")
    parser.add_argument("--natural", action="store_true")
    parser.add_argument("--forehand", action="store_true", help="Outward racket head, right-side loading, full pose constraints")
    parser.add_argument("--target-offset", nargs=3, type=float, default=(0., 0., 0.), metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    output = args.output_dir or ROOT / "outputs/forehand"
    generate(output, args.urdf, args.fps, args.table_scene, args.right_shoulder_roll_max,
             args.racket_normal_only, args.natural, args.forehand, args.target_offset)
