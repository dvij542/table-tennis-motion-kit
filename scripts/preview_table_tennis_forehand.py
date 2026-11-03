"""Render a kinematic forehand preview, or serve it in Viser with a free camera."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy

from build_table_tennis_forehand_bundle import load_robot, ROOT


def table_meshes(table):
    import trimesh

    length, width, height = table["length_m"], table["width_m"], table["height_m"]
    near, thick = table["near_edge_x_m"], table["thickness_m"]
    cx, cy = near + length/2, table["center_y_m"]
    parts = []

    def box(name, size, position, color):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(position)
        parts.append((name, mesh, color))

    box("top", (length, width, thick), (cx, cy, height-thick/2), (30, 105, 155))
    for side in (-1, 1):
        box(f"sideline{side}", (length, .02, .001), (cx, cy+side*(width/2-.01), height+.0005), (240, 240, 240))
        box(f"endline{side}", (.02, width, .001), (cx+side*(length/2-.01), cy, height+.0005), (240, 240, 240))
    box("centerline", (length, .003, .001), (cx, cy, height+.0005), (240, 240, 240))
    for i, x in enumerate((near+.3, near+length-.3)):
        for j, y in enumerate((cy-width/2+.18, cy+width/2-.18)):
            box(f"leg{i}{j}", (.045, .045, height-thick), (x, y, (height-thick)/2), (65, 70, 75))
    net_width = width + .305
    for i, y in enumerate(np.linspace(cy-net_width/2, cy+net_width/2, 100)):
        box(f"net_vertical{i}", (.0015, .0015, table["net_height_m"]), (cx, y, height+table["net_height_m"]/2), (215, 215, 215))
    for i, z in enumerate(np.linspace(height, height+table["net_height_m"], 10)):
        box(f"net_horizontal{i}", (.0015, net_width, .0015), (cx, cy, z), (215, 215, 215))
    box("net_tape", (.008, net_width, .012), (cx, cy, height+table["net_height_m"]), (250,250,250))
    for side in (-1, 1):
        box(f"net_post{side}", (.02, .02, .19), (cx, cy+side*net_width/2, height+.075), (65, 70, 75))
    return parts


def racket_meshes(racket):
    import trimesh

    if racket.get("embedded_in_urdf", False):
        return []
    center = np.array(racket["center_offset_local"])
    mount = np.array(racket.get("blade_rotation_local", np.eye(3)))
    meshes = []
    for name, thickness, x_offset, color in (("wood", .012, 0., (180, 145, 90)),
                                             ("rubber_red", .001, .0065, (205, 35, 40)),
                                             ("rubber_black", .001, -.0065, (30, 30, 32))):
        mesh = trimesh.creation.cylinder(radius=1, height=thickness, sections=64)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        mesh.vertices[:, 1] *= racket["blade_half_width_m"]
        mesh.vertices[:, 2] *= racket["blade_half_length_m"]
        mesh.vertices = mesh.vertices @ mount.T
        mesh.apply_translation(center + mount @ [x_offset, 0, 0])
        meshes.append((name, mesh, color))
    handle = trimesh.creation.box(extents=[.023, .028, .10])
    handle.vertices = handle.vertices @ mount.T
    handle.apply_translation(center - mount @ [0, 0, .12])
    meshes.append(("handle", handle, (170, 125, 75)))
    return meshes


def quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[[3, 0, 1, 2]]


def load(bundle):
    meta = json.loads((bundle / "metadata.json").read_text())
    data = np.load(bundle / "reference.npz")
    asset = Path(meta["urdf"])
    robot = load_robot(asset if asset.is_absolute() else bundle / asset)
    assert list(data["joint_names"]) == list(robot.actuated_joint_names)
    return meta, data, robot


def serve(bundle, port, compare_bundle=None):
    import viser
    from viser.extras import ViserUrdf

    grid = None
    grid_path = bundle / "motion_grid.json"
    if grid_path.exists():
        if compare_bundle is not None:
            raise ValueError("Grid and legacy comparison modes are separate")
        grid = json.loads(grid_path.read_text())
        if not grid["complete"]:
            raise ValueError("Do not visualize an incomplete grid as a complete target range")
        initial = min(grid["entries"], key=lambda entry: np.linalg.norm(entry["offset_m"]))
        meta, data, robot = load(bundle / initial["directory"])
        grid_variants = {}
        for entry in grid["entries"]:
            directory = bundle / entry["directory"]
            entry_meta = json.loads((directory / "metadata.json").read_text())
            with np.load(directory / "reference.npz") as archive:
                entry_data = {key: archive[key] for key in archive.files}
            assert entry_meta["urdf_sha256"] == meta["urdf_sha256"]
            np.testing.assert_array_equal(entry_data["joint_names"], data["joint_names"])
            grid_variants[tuple(round(value*100) for value in entry["offset_m"][1:])] = (entry_meta, entry_data)
    else:
        meta, data, robot = load(bundle)
    variants = {"Forehand": (meta, data)}
    if compare_bundle is not None:
        other_meta = json.loads((compare_bundle / "metadata.json").read_text())
        other_data = np.load(compare_bundle / "reference.npz")
        if other_meta["urdf_sha256"] != meta["urdf_sha256"] or other_meta.get("table") != meta.get("table"):
            raise ValueError("Comparison requires the same robot and table")
        np.testing.assert_array_equal(other_data["joint_names"], data["joint_names"])
        variants["Previous (backhand)"] = (other_meta, other_data)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    selection = server.gui.add_dropdown("Motion", options=tuple(variants), initial_value="Forehand") if compare_bundle else None
    selected = "Forehand"
    y_slider = z_slider = None
    selected_offsets = None
    if grid is not None:
        y_slider = server.gui.add_slider("Target Y offset (cm)", min=-15, max=5, step=5, initial_value=0)
        z_slider = server.gui.add_slider("Target Z offset (cm)", min=0, max=15, step=5, initial_value=0)
        selected_offsets = (0, 0)
    root = server.scene.add_frame("/robot", show_axes=False, position=data["root_pos"][0])
    visual = ViserUrdf(server, robot, root_node_name="/robot")
    server.scene.add_grid("/ground", width=8, height=6, plane="xy")
    path = server.scene.add_point_cloud("/path", points=data["actual_contact_pos"] if "actual_contact_pos" in data else data["actual_hand_pos"],
                                 colors=np.tile([245, 155, 40], (len(data["time"]), 1)), point_size=.008)
    racket_frame = None
    racket_handles = []
    if "table" in meta:
        for name, mesh, color in table_meshes(meta["table"]):
            server.scene.add_mesh_simple(f"/table/{name}", vertices=mesh.vertices, faces=mesh.faces, color=color)
        racket_frame = server.scene.add_frame("/racket", show_axes=False)
        for name, mesh, color in racket_meshes(meta["racket"]):
            racket_handles.append(server.scene.add_mesh_simple(f"/racket/{name}", vertices=mesh.vertices, faces=mesh.faces, color=color))
    contact = server.scene.add_icosphere("/contact", radius=.01, color=(60, 220, 90), position=tuple(meta["target_hit_position_world"]))
    frame = server.gui.add_slider("Frame", min=0, max=len(data["time"])-1, step=1, initial_value=0)
    play = server.gui.add_checkbox("Play", initial_value=True)
    rate = server.gui.add_slider("Playback speed", min=.25, max=1.5, step=.25, initial_value=1.)
    hit = server.gui.add_button("Hit pose (1.00 s)")
    show_path = server.gui.add_checkbox("Racket path", initial_value=True)

    @hit.on_click
    def jump_to_hit(_):
        play.value = False
        frame.value = meta["hit_idx"]

    @show_path.on_update
    def toggle_path(_):
        path.visible = show_path.value

    @server.on_client_connect
    def connected(client):
        client.camera.position = (-1.8, -3.5, 2.5) if "table" in meta else (2., -2., 1.6)
        client.camera.look_at = (1.2, 0., .65) if "table" in meta else (.15, 0., .8)
        client.camera.up_direction = (0., 0., 1.)
        if "table" in meta:
            client.camera.fov = np.radians(50)
            center = np.array([1.2, 0., .65])
            distance_scale = max(1., .95/client.camera.aspect)
            client.camera.position = center + (np.array([-1.8, -3.5, 2.5])-center)*distance_scale

    while True:
        if grid is not None and (y_slider.value, z_slider.value) != selected_offsets:
            selected_offsets = (y_slider.value, z_slider.value)
            meta, data = grid_variants[selected_offsets]
            # Preserve playback phase and the user's camera while changing targets.
            path.points = data["actual_contact_pos"]
            contact.position = tuple(meta["target_hit_position_world"])
        if selection is not None and selection.value != selected:
            selected = selection.value
            meta, data = variants[selected]
            frame.max = len(data["time"])-1
            frame.value = 0
            path.points = data["actual_contact_pos"]
            path.colors = np.tile([245, 155, 40], (len(data["time"]), 1))
            contact.position = tuple(meta["target_hit_position_world"])
            for handle in racket_handles:
                handle.remove()
            racket_handles = [server.scene.add_mesh_simple(f"/racket/{name}", vertices=mesh.vertices,
                              faces=mesh.faces, color=color)
                              for name, mesh, color in racket_meshes(meta["racket"])]
        i = frame.value
        root.position = data["root_pos"][i]
        visual.update_cfg(data["cfg_urdf"][i])
        if racket_frame is not None:
            racket_frame.position = data["wrist_pos"][i]
            racket_frame.wxyz = quat(data["wrist_rotation"][i])
        time.sleep(1 / (meta["fps"] * rate.value))
        if play.value:
            frame.value = (i + 1) % len(data["time"])


def render(bundle, closeup=False):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import imageio.v2 as imageio
    import mujoco

    meta, data, robot = load(bundle)
    model_xml = ET.Element("mujoco")
    ET.SubElement(model_xml, "compiler", angle="radian")
    visual = ET.SubElement(model_xml, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="800")
    assets = ET.SubElement(model_xml, "asset")
    ET.SubElement(assets, "texture", name="grid", type="2d", builtin="checker", rgb1=".23 .25 .28", rgb2=".29 .31 .34", width="512", height="512")
    ET.SubElement(assets, "material", name="ground", texture="grid", texrepeat="8 8", reflectance="0")
    world = ET.SubElement(model_xml, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 4", diffuse=".8 .8 .8", ambient=".4 .4 .4", castshadow="true")
    ET.SubElement(world, "geom", type="plane", size="5 5 .1", material="ground")
    nodes = list(robot.scene.graph.nodes_geometry)
    for index, node in enumerate(nodes):
        _, geometry = robot.scene.graph[node]
        mesh = robot.scene.geometry[geometry]
        name = f"visual{index}"
        ET.SubElement(assets, "mesh", name=name, vertex=" ".join(map(str, mesh.vertices.ravel())), face=" ".join(map(str, mesh.faces.ravel())))
        body = ET.SubElement(world, "body", name=name, mocap="true")
        color = ".75 .77 .80 1"
        if "yam" in node:
            color = ".18 .40 .65 1"
        if "racket" in node:
            color = ".7 .025 .025 1"
        ET.SubElement(body, "geom", type="mesh", mesh=name, rgba=color, contype="0", conaffinity="0")
    if "table" in meta:
        racket_body = ET.SubElement(world, "body", name="racket", mocap="true")
        for prefix, parts, parent in (("table", table_meshes(meta["table"]), world),
                                       ("racket", racket_meshes(meta["racket"]), racket_body)):
            for name, mesh, color in parts:
                mesh_name = f"{prefix}_{name}"
                ET.SubElement(assets, "mesh", name=mesh_name, vertex=" ".join(map(str, mesh.vertices.ravel())), face=" ".join(map(str, mesh.faces.ravel())))
                ET.SubElement(parent, "geom", type="mesh", mesh=mesh_name,
                              rgba=" ".join(str(c/255) for c in color)+" 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", type="sphere", size=".014", pos=" ".join(map(str, meta["target_hit_position_world"])), rgba=".1 .9 .25 .8", contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))
    sim = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=800, width=640)
    cameras = []
    for azimuth in (140, 70):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.15, 0., .72]
        cam.distance, cam.azimuth, cam.elevation = 2.5, azimuth, -12
        if "table" in meta:
            cam.lookat[:] = [1.1, 0., .65]
            cam.distance, cam.azimuth, cam.elevation = 4.3, azimuth, -20
        if closeup:
            cam.lookat[:] = [.2, -.1, .95]
            cam.distance, cam.azimuth, cam.elevation = 1.65, azimuth, -10
        cameras.append(cam)
    frames = []
    for i in range(0, len(data["time"]), 2):
        robot.update_cfg(data["cfg_urdf"][i])
        for j, node in enumerate(nodes):
            transform, _ = robot.scene.graph[node]
            sim.mocap_pos[j] = transform[:3, 3] + data["root_pos"][i]
            sim.mocap_quat[j] = quat(transform[:3, :3])
        if "table" in meta:
            sim.mocap_pos[len(nodes)] = data["wrist_pos"][i]
            sim.mocap_quat[len(nodes)] = quat(data["wrist_rotation"][i])
        mujoco.mj_forward(model, sim)
        views = []
        for cam, label in zip(cameras, ("Front-right", "Right-side")):
            renderer.update_scene(sim, camera=cam)
            image = renderer.render().copy()
            cv2.putText(image, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (245,245,245), 2)
            views.append(image)
        image = np.concatenate(views, axis=1)
        label = f"RIGHT FOREHAND | t={data['time'][i]:.2f}s | hit at 1.00s | kinematic reference"
        cv2.putText(image, label, (20, 760), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        if i == meta["hit_idx"]:
            cv2.putText(image, "CONTACT", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1., (70,255,110), 2)
        frames.append(image)
        if i in (0, 34, 50, 64, 124):
            imageio.imwrite(bundle / f"frame_{i:03d}.png", image)
    renderer.close()
    path = bundle / "forehand_preview.mp4"
    with imageio.get_writer(path, fps=25, codec="libx264", quality=8, macro_block_size=1) as writer:
        for label, repeat in (("1x speed", 1), ("0.5x slow motion", 2)):
            for frame in frames:
                image = frame.copy()
                cv2.putText(image, label, (20, 710), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
                for _ in range(repeat):
                    writer.append_data(image)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "examples/forehand_grid")
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--closeup", action="store_true")
    parser.add_argument("--compare-bundle", type=Path)
    args = parser.parse_args()
    serve(args.bundle_dir, args.port, args.compare_bundle) if args.viser else render(args.bundle_dir, args.closeup)
