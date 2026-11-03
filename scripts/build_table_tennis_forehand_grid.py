"""Solve a Y/Z grid of stationary forehands and pack them into one motion bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import build_table_tennis_forehand_bundle as generator
from build_table_tennis_forehand_bundle import generate, KEY


def build(output, urdf, pack_only=False):
    output.mkdir(parents=True, exist_ok=True)
    y_values = [-.15, -.10, -.05, 0., .05]
    z_values = [0., .05, .10, .15]
    motions, metadata, entries, failures = {}, {}, [], []
    # Solve the nominal motion first and preserve successful work on a retry.
    pairs = sorted([(y, z) for y in y_values for z in z_values], key=lambda yz: abs(yz[0])+yz[1])
    generator_hash = hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest()
    urdf_hash = hashlib.sha256(urdf.read_bytes()).hexdigest()
    for i, (y, z) in enumerate(pairs):
        key = f"table_tennis_forehand_y{round(y*100):+04d}_z{round(z*100):03d}"
        directory = output / key
        print(f"GRID {i+1}/{len(pairs)} offset=(0, {y}, {z})", flush=True)
        try:
            cache = directory / "metadata.json"
            existing = json.loads(cache.read_text()) if cache.exists() else {}
            valid_cache = ((pack_only or existing.get("generator_sha256") == generator_hash)
                           and existing.get("urdf_sha256") == urdf_hash
                           and existing.get("target_offset_world_m") == [0., y, z]
                           and all((directory / name).exists() for name in
                                   ["processed_robot_motions.pkl", "motion_meta_bundle.pkl", "reference.npz"]))
            if not valid_cache:
                if pack_only:
                    raise RuntimeError("Missing or incompatible validated motion; generate it before packing")
                generate(directory, urdf, table_scene=True, right_shoulder_roll_max=-.3,
                         forehand=True, target_offset=(0., y, z))
            checks = json.loads(cache.read_text())["validation"] if cache.exists() else None
            if checks is not None:
                assert checks["max_position_error_m"] < .002
                assert checks["max_orientation_error_deg"] < .5
                assert checks["leg_joint_max_variation_rad"] == 0.
                assert checks["left_joint_max_variation_rad"] == 0.
                assert checks["right_shoulder_roll_range_rad"][1] <= -.3+1e-7
            motions[key] = joblib.load(directory / "processed_robot_motions.pkl")[KEY]
            metadata[key] = joblib.load(directory / "motion_meta_bundle.pkl")[KEY]
            if "urdf" in metadata[key]:
                metadata[key] = dict(metadata[key],
                                     urdf=os.path.relpath(urdf.resolve(), output.resolve()))
            entries.append(dict(key=key, directory=key, offset_m=[0., y, z],
                                generator_sha256=metadata[key].get("generator_sha256"),
                                target_xyz_m=metadata[key]["target_hit_position_world"]))
        except (AssertionError, RuntimeError) as exc:
            failures.append(dict(key=key, offset_m=[0., y, z], error=str(exc)))
            print(f"REJECTED {key}: {exc}", flush=True)
    manifest = dict(axes=["y", "z"], y_offsets_m=y_values, z_offsets_m=z_values,
                    nominal_target_xyz_m=[.43, -.38, .86], entries=entries, failures=failures,
                    complete=not failures, fps=50, hit_time_s=1., duration_s=2.5)
    (output / "motion_grid.json").write_text(json.dumps(manifest, indent=2)+"\n")
    joblib.dump(motions, output / "processed_robot_motions.pkl", compress=3)
    joblib.dump(metadata, output / "motion_meta_bundle.pkl", compress=3)
    print(f"Saved {len(motions)} motions; {len(failures)} rejected", flush=True)
    if failures:
        raise RuntimeError("Grid incomplete; inspect motion_grid.json. Rejected targets were not substituted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=generator.ROOT / "outputs/forehand_grid")
    parser.add_argument("--urdf", type=Path, default=generator.DEFAULT_URDF)
    parser.add_argument("--pack-only", action="store_true", help="Pack already validated variants, retaining their generation provenance")
    args = parser.parse_args()
    build(args.output_dir, args.urdf, args.pack_only)
