"""MuJoCo utilities for synthesizing stable tabletop clutter scenes."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time

import mujoco
import numpy as np


@dataclass
class ClutterSimCfg:
    timestep: float = 0.002
    table_size: tuple[float, float] = (0.4, 0.4)
    table_height: float = 0.0
    table_thickness: float = 0.04
    friction: float = 0.8
    table_friction: float = 1000.0
    impratio: float = 10.0
    noslip_iterations: int = 3
    object_mass: float = 0.1
    spawn_z: tuple[float, float] = (0.18, 0.35)
    spawn_xy_scale: float = 1.0
    spawn_gap: float = 0.02
    spawn_position_attempts: int = 100
    max_object_speed: float = 0.5
    freejoint_damping: float = 0.001
    containment_walls: bool = True
    containment_height: float = 0.6
    containment_thickness: float = 0.02
    bound_mass: float = 1e-4
    bound_inertia: float = 1e-6
    debug_view: bool = False
    debug_realtime: bool = True
    debug_spawn_pause: float = 0.5
    debug_result_pause: float = 1.0
    settle_timeout: float = 15.0
    stable_time: float = 0.5
    linear_velocity: float = 0.0001
    angular_velocity: float = 0.0
    penetration_tolerance: float = 1e-4
    table_penetration_tolerance: float = 0.001


class MuJoCoClutterEnv:
    """Drop rigid-object assets on a finite table and return stable poses."""

    _OBJECT_PREFIX = "clutter-"
    _INSTABILITY_WARNINGS = (
        (mujoco.mjtWarning.mjWARN_BADQPOS, "BADQPOS"),
        (mujoco.mjtWarning.mjWARN_BADQVEL, "BADQVEL"),
        (mujoco.mjtWarning.mjWARN_BADQACC, "BADQACC"),
    )

    def __init__(self, objects: list[dict], cfg: ClutterSimCfg):
        self.objects = objects
        self.cfg = cfg
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.getcwd()
        self.spec.compiler.boundmass = cfg.bound_mass
        self.spec.compiler.boundinertia = cfg.bound_inertia
        self.spec.compiler.balanceinertia = True
        self.spec.option.timestep = cfg.timestep
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.impratio = cfg.impratio
        self.spec.option.noslip_iterations = cfg.noslip_iterations
        self.spec.option.gravity[:] = [0.0, 0.0, -9.81]

        half_x, half_y = np.asarray(cfg.table_size, dtype=float) / 2
        self.spec.worldbody.add_geom(
            name="clutter-table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, cfg.table_height - cfg.table_thickness / 2],
            size=[half_x, half_y, cfg.table_thickness / 2],
            friction=[cfg.table_friction, 0.005, 0.0001],
            rgba=[0.35, 0.32, 0.28, 1.0],
        )
        if cfg.containment_walls:
            self._add_containment_walls(half_x, half_y)

        self._body_names: list[str] = []
        self._joint_names: list[str] = []
        for obj in objects:
            self._add_object(obj)

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._body_ids = [self.model.body(name).id for name in self._body_names]
        self._joint_ids = [self.model.joint(name).id for name in self._joint_names]
        self._qpos_adrs = [int(self.model.jnt_qposadr[j]) for j in self._joint_ids]
        self._dof_adrs = [int(self.model.jnt_dofadr[j]) for j in self._joint_ids]
        self._table_geom_id = self.model.geom("clutter-table").id
        body_objects = dict(zip(self._body_ids, (obj["name"] for obj in objects)))
        self._geom_object_names: dict[int, str] = {}
        self._object_geom_ids = {obj["name"]: [] for obj in objects}
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            while body_id != 0 and body_id not in body_objects:
                body_id = self.model.body_parentid[body_id]
            if body_id in body_objects:
                name = body_objects[body_id]
                self._geom_object_names[geom_id] = name
                self._object_geom_ids[name].append(geom_id)
        self._debug_view = None
        self._debug_paused = False
        self._debug_single_step = False
        self._debug_wall_start = time.perf_counter()
        if cfg.debug_view:
            self._open_debug_view()

    def _add_containment_walls(self, half_x: float, half_y: float) -> None:
        """Add invisible temporary walls used only while settling objects."""
        thickness = self.cfg.containment_thickness
        half_height = self.cfg.containment_height / 2
        z = self.cfg.table_height + half_height
        common = {
            "type": mujoco.mjtGeom.mjGEOM_BOX,
            "friction": [self.cfg.friction, 0.005, 0.0001],
            "rgba": [0.0, 0.0, 0.0, 0.0],
            "group": 5,
        }
        for side, x in (("left", -half_x), ("right", half_x)):
            self.spec.worldbody.add_geom(
                name=f"clutter-wall-{side}",
                pos=[x + np.sign(x) * thickness / 2, 0.0, z],
                size=[thickness / 2, half_y + thickness, half_height],
                **common,
            )
        for side, y in (("front", -half_y), ("back", half_y)):
            self.spec.worldbody.add_geom(
                name=f"clutter-wall-{side}",
                pos=[0.0, y + np.sign(y) * thickness / 2, z],
                size=[half_x + thickness, thickness / 2, half_height],
                **common,
            )

    def _add_object(self, obj: dict) -> None:
        name = obj["name"]
        child_spec = mujoco.MjSpec.from_file(obj["xml_path"])
        for mesh in child_spec.meshes:
            mesh.file = os.path.join(
                os.path.dirname(obj["xml_path"]), child_spec.meshdir, mesh.file
            )
            mesh.scale *= np.asarray(obj["scale"], dtype=float)
        child_spec.meshdir = self.spec.meshdir

        for geom in child_spec.geoms:
            if geom.contype != 0:
                geom.density = float(obj["density"])
                geom.friction[:] = [self.cfg.friction, 0.005, 0.0001]
                geom.condim = 4

        prefix = self._OBJECT_PREFIX + name
        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(child_spec.worldbody, prefix, "")
        body_name = prefix + "world"
        joint_name = prefix + "freejoint"
        freejoint = child_world.add_freejoint(name=joint_name)
        freejoint.damping = self.cfg.freejoint_damping
        self._body_names.append(body_name)
        self._joint_names.append(joint_name)

    def set_spawn_poses(self, poses: np.ndarray) -> bool:
        """Reset objects at spawn poses and reject initial penetration."""
        if poses.shape != (len(self.objects), 7):
            raise ValueError(
                f"Expected spawn poses {(len(self.objects), 7)}, got {poses.shape}"
            )
        mujoco.mj_resetData(self.model, self.data)
        for qpos_adr, pose in zip(self._qpos_adrs, poses):
            self.data.qpos[qpos_adr : qpos_adr + 7] = pose
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        valid = not any(
            contact.dist < -self.cfg.penetration_tolerance
            for contact in self.data.contact
        )
        self._debug_sync(force=True)
        self._debug_hold(self.cfg.debug_spawn_pause)
        self._debug_wall_start = time.perf_counter() - self.data.time
        return valid

    def settle(self) -> tuple[bool, np.ndarray, dict]:
        """Run gravity until every free object remains below velocity limits."""
        stable_steps = 0
        required_stable_steps = max(1, round(self.cfg.stable_time / self.cfg.timestep))
        max_steps = max(1, round(self.cfg.settle_timeout / self.cfg.timestep))
        max_linear = float("inf")
        max_angular = float("inf")
        velocity_clamps = 0

        for step in range(max_steps):
            mujoco.mj_step(self.model, self.data)
            self._debug_sync()
            warning = self._get_instability_warning()
            if warning is not None:
                warning_name, warning_info = warning
                return self._settle_result(False, {
                    "reason": "simulation_instability",
                    "warning": warning_name,
                    "dof": warning_info,
                    "simulated_seconds": (step + 1) * self.cfg.timestep,
                    "velocity_clamps": velocity_clamps,
                })
            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
                and np.all(np.isfinite(self.data.qacc))
            ):
                return self._settle_result(False, {
                    "reason": "nonfinite_state",
                    "simulated_seconds": (step + 1) * self.cfg.timestep,
                    "velocity_clamps": velocity_clamps,
                })
            velocity_clamps += self._clamp_object_velocities()
            # Normal impacts briefly compress soft contacts. During the drop,
            # reject objects wholly below the tabletop, even without contacts.
            penetration = self.get_table_penetration(check_contacts=False)
            if penetration is not None:
                return self._settle_result(False, {
                    **penetration,
                    "simulated_seconds": (step + 1) * self.cfg.timestep,
                    "velocity_clamps": velocity_clamps,
                })
            if step % 25 == 0 and self._escaped_workspace(self.get_poses()):
                return self._settle_result(False, {
                    "reason": "escaped_workspace",
                    "simulated_seconds": (step + 1) * self.cfg.timestep,
                    "velocity_clamps": velocity_clamps,
                })
            max_linear = max(
                np.linalg.norm(self.data.qvel[adr : adr + 3])
                for adr in self._dof_adrs
            )
            max_angular = max(
                np.linalg.norm(self.data.qvel[adr + 3 : adr + 6])
                for adr in self._dof_adrs
            )
            if (
                max_linear < self.cfg.linear_velocity
                and (
                    self.cfg.angular_velocity <= 0
                    or max_angular <= self.cfg.angular_velocity
                )
            ):
                stable_steps += 1
                if stable_steps >= required_stable_steps:
                    break
            else:
                stable_steps = 0
        else:
            return self._settle_result(False, {
                "reason": "settle_timeout",
                "max_linear_velocity": max_linear,
                "max_angular_velocity": max_angular,
                "velocity_clamps": velocity_clamps,
            })

        # Refresh contacts at the final integrated pose before accepting it.
        mujoco.mj_forward(self.model, self.data)
        penetration = self.get_table_penetration()
        if penetration is not None:
            return self._settle_result(False, {
                **penetration,
                "simulated_seconds": (step + 1) * self.cfg.timestep,
                "velocity_clamps": velocity_clamps,
            })
        poses = self.get_poses()
        if not self._inside_workspace(poses):
            return self._settle_result(False, {
                "reason": "outside_workspace",
                "max_linear_velocity": max_linear,
                "max_angular_velocity": max_angular,
                "velocity_clamps": velocity_clamps,
            })
        return self._settle_result(True, {
            "reason": "stable",
            "simulated_seconds": (step + 1) * self.cfg.timestep,
            "max_linear_velocity": max_linear,
            "max_angular_velocity": max_angular,
            "velocity_clamps": velocity_clamps,
        })

    def get_table_penetration(self, check_contacts: bool = True) -> dict | None:
        """Detect deep table contacts or objects wholly below the tabletop."""
        tolerance = self.cfg.table_penetration_tolerance
        if check_contacts:
            for contact in self.data.contact:
                if self._table_geom_id not in (contact.geom1, contact.geom2):
                    continue
                other_geom = (
                    contact.geom2
                    if contact.geom1 == self._table_geom_id
                    else contact.geom1
                )
                name = self._geom_object_names.get(other_geom)
                if name is not None and contact.dist < -tolerance:
                    return {
                        "reason": "table_penetration",
                        "object_name": name,
                        "penetration_depth": float(-contact.dist),
                        "fully_below_table": False,
                    }

        for name, geom_ids in self._object_geom_ids.items():
            if not geom_ids:
                continue
            # Transform each object's local bounds to a conservative
            # world-space upper bound, accounting for rotation and offsets.
            z_axes = self.data.geom_xmat[geom_ids].reshape(-1, 3, 3)[:, 2, :]
            bounds = self.model.geom_aabb[geom_ids]
            upper_z = (
                self.data.geom_xpos[geom_ids, 2]
                + np.sum(z_axes * bounds[:, :3], axis=1)
                + np.sum(np.abs(z_axes) * bounds[:, 3:], axis=1)
            )
            top_z = float(upper_z.max())
            if top_z < self.cfg.table_height - tolerance:
                return {
                    "reason": "table_penetration",
                    "object_name": name,
                    "penetration_depth": self.cfg.table_height - top_z,
                    "fully_below_table": True,
                }
            if check_contacts:
                lower_z = (
                    self.data.geom_xpos[geom_ids, 2]
                    + np.sum(z_axes * bounds[:, :3], axis=1)
                    - np.sum(np.abs(z_axes) * bounds[:, 3:], axis=1)
                )
                for geom_id, bottom_z in zip(geom_ids, lower_z):
                    if bottom_z >= self.cfg.table_height - tolerance:
                        continue
                    bottom_z = self._geom_bottom_z(geom_id)
                    if bottom_z < self.cfg.table_height - tolerance:
                        return {
                            "reason": "table_penetration",
                            "object_name": name,
                            "penetration_depth": self.cfg.table_height - bottom_z,
                            "fully_below_table": False,
                        }
        return None

    def _geom_bottom_z(self, geom_id: int) -> float:
        """Compute geometry's lowest world point, avoiding loose bounds."""
        z_axis = self.data.geom_xmat[geom_id].reshape(3, 3)[2]
        z = float(self.data.geom_xpos[geom_id, 2])
        size = self.model.geom_size[geom_id]
        geom_type = self.model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[geom_id]
            start = self.model.mesh_vertadr[mesh_id]
            count = self.model.mesh_vertnum[mesh_id]
            vertices = self.model.mesh_vert[start : start + count]
            return z + float(np.min(vertices @ z_axis))
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0] + size[1] * abs(z_axis[2])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = size[0] * np.linalg.norm(z_axis[:2]) + size[1] * abs(z_axis[2])
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radius = np.linalg.norm(z_axis * size)
        else:
            bounds = self.model.geom_aabb[geom_id]
            z += float(np.dot(z_axis, bounds[:3]))
            radius = np.dot(np.abs(z_axis), bounds[3:])
        return z - float(radius)

    def _settle_result(self, stable: bool, info: dict) -> tuple[bool, np.ndarray, dict]:
        self._debug_sync(force=True)
        if self._debug_view is not None:
            logging.info("Debug settle result: %s", info)
        self._debug_hold(self.cfg.debug_result_pause)
        return stable, self.get_poses(), info

    def _open_debug_view(self) -> None:
        import glfw
        from mujoco import viewer as mujoco_viewer

        def on_key(key: int) -> None:
            if key == glfw.KEY_SPACE:
                self._debug_paused = not self._debug_paused
            elif key == glfw.KEY_N:
                self._debug_single_step = True

        self._debug_view = mujoco_viewer.launch_passive(
            self.model,
            self.data,
            key_callback=on_key,
            show_left_ui=True,
            show_right_ui=False,
        )
        self._debug_view.cam.lookat[:] = [0.0, 0.0, self.cfg.table_height + 0.08]
        self._debug_view.cam.distance = max(self.cfg.table_size) * 1.8
        self._debug_view.cam.azimuth = 135.0
        self._debug_view.cam.elevation = -30.0
        self._debug_view.opt.geomgroup[5] = False
        logging.info("MuJoCo clutter debug keys: Space pause/resume, N single step")
        self._debug_sync(force=True)

    def _debug_sync(self, force: bool = False) -> None:
        if self._debug_view is None:
            return
        if not self._debug_view.is_running():
            self.close()
            return

        waited = False
        while self._debug_paused and not self._debug_single_step:
            if not self._debug_view.is_running():
                self.close()
                return
            self._debug_view.sync()
            time.sleep(0.02)
            waited = True
        if waited:
            self._debug_wall_start = time.perf_counter() - self.data.time
        self._debug_single_step = False

        if self.cfg.debug_realtime and not force:
            delay = self._debug_wall_start + self.data.time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self._debug_view.sync()

    def _debug_hold(self, seconds: float) -> None:
        if self._debug_view is None or seconds <= 0:
            return
        deadline = time.perf_counter() + seconds
        while self._debug_view is not None and time.perf_counter() < deadline:
            if not self._debug_view.is_running():
                self.close()
                return
            self._debug_view.sync()
            time.sleep(0.02)

    def close(self) -> None:
        if self._debug_view is not None:
            self._debug_view.close()
            self._debug_view = None

    def _clamp_object_velocities(self) -> int:
        clamp_count = 0
        max_speed = self.cfg.max_object_speed
        if max_speed <= 0:
            return clamp_count
        for dof_adr in self._dof_adrs:
            # Match vmf-contact2: clamp only the linear base velocity after
            # every physics step, preserving its direction and angular speed.
            velocity = self.data.qvel[dof_adr : dof_adr + 3]
            speed = float(np.linalg.norm(velocity))
            if speed > max_speed and speed > 1e-12:
                velocity *= max_speed / speed
                clamp_count += 1
        return clamp_count

    def _get_instability_warning(self) -> tuple[str, int] | None:
        for warning_type, warning_name in self._INSTABILITY_WARNINGS:
            warning = self.data.warning[int(warning_type)]
            if warning.number:
                return warning_name, int(warning.lastinfo)
        return None

    def get_poses(self) -> np.ndarray:
        return np.asarray(
            [
                np.concatenate([self.data.xpos[body_id], self.data.xquat[body_id]])
                for body_id in self._body_ids
            ],
            dtype=np.float32,
        )

    def _inside_workspace(self, poses: np.ndarray) -> bool:
        half_x, half_y = np.asarray(self.cfg.table_size, dtype=float) / 2
        for pose, obj in zip(poses, self.objects):
            extent = np.asarray(obj["extent"], dtype=float)
            margin_x = min(extent.max() / 2, half_x * 0.9)
            margin_y = min(extent.max() / 2, half_y * 0.9)
            if abs(pose[0]) > half_x - margin_x:
                return False
            if abs(pose[1]) > half_y - margin_y:
                return False
            if pose[2] < self.cfg.table_height - 0.02 or pose[2] > 1.0:
                return False
        return True

    def _escaped_workspace(self, poses: np.ndarray) -> bool:
        half_x, half_y = np.asarray(self.cfg.table_size, dtype=float) / 2
        return any(
            abs(pose[0]) > half_x + 0.1
            or abs(pose[1]) > half_y + 0.1
            or pose[2] < self.cfg.table_height - 0.1
            or pose[2] > 2.0
            for pose in poses
        )


def sample_spawn_poses(
    objects: list[dict], cfg: ClutterSimCfg, rng: np.random.Generator
) -> np.ndarray:
    """Sample separated vertical drop poses over the tabletop."""
    half_x, half_y = np.asarray(cfg.table_size, dtype=float) / 2
    if not 0 < cfg.spawn_xy_scale <= 1:
        raise ValueError(
            f"spawn_xy_scale must be in (0, 1], got {cfg.spawn_xy_scale}"
        )
    spawn_half_x = half_x * cfg.spawn_xy_scale
    spawn_half_y = half_y * cfg.spawn_xy_scale
    order = rng.permutation(len(objects))
    poses = np.zeros((len(objects), 7), dtype=np.float64)
    placed: list[tuple[np.ndarray, float]] = []
    for obj_index in order:
        extent = np.asarray(objects[obj_index]["extent"], dtype=float)
        bounding_radius = float(np.linalg.norm(extent) / 2)
        spawn_radius = min(
            bounding_radius, min(spawn_half_x, spawn_half_y) * 0.8
        )
        x_limit = max(0.01, spawn_half_x - spawn_radius)
        y_limit = max(0.01, spawn_half_y - spawn_radius)
        z_min = max(cfg.spawn_z[0], cfg.table_height + bounding_radius + 0.01)
        z_max = max(z_min, cfg.spawn_z[1])

        position = None
        for _ in range(cfg.spawn_position_attempts):
            candidate = np.asarray(
                [
                    rng.uniform(-x_limit, x_limit),
                    rng.uniform(-y_limit, y_limit),
                    rng.uniform(z_min, z_max),
                ]
            )
            if all(
                np.linalg.norm(candidate - other_position)
                >= bounding_radius + other_radius + cfg.spawn_gap
                for other_position, other_radius in placed
            ):
                position = candidate
                break

        if position is None:
            # If the random volume is crowded, keep the sampled XY position and
            # raise the object just enough to clear every previously placed sphere.
            position = np.asarray(
                [
                    rng.uniform(-x_limit, x_limit),
                    rng.uniform(-y_limit, y_limit),
                    z_min,
                ]
            )
            for other_position, other_radius in placed:
                required = bounding_radius + other_radius + cfg.spawn_gap
                horizontal = float(np.linalg.norm(position[:2] - other_position[:2]))
                if horizontal < required:
                    vertical = np.sqrt(max(required**2 - horizontal**2, 0.0))
                    position[2] = max(position[2], other_position[2] + vertical)

        poses[obj_index, :3] = position
        placed.append((position, bounding_radius))
        # Shoemake's uniform random unit quaternion, returned as MuJoCo WXYZ.
        u1, u2, u3 = rng.random(3)
        x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        z = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        w = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        poses[obj_index, 3:] = [w, x, y, z]
    return poses
