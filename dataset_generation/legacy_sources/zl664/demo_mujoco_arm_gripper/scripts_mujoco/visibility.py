from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ProjectionRecord:
    frame_index: int
    time_s: float
    ball_position: list[float]
    camera_position: list[float]
    in_front: bool
    visible: bool
    comfortably_visible: bool
    trackable: bool
    pixel_xy: list[float] | None
    pixel_radius: float
    depth: float


def _name2id(model: mujoco.MjModel, objtype, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return idx


def _event_frame(time_s: float | None, fps: int, frame_count: int) -> int | None:
    if time_s is None:
        return None
    return int(np.clip(round(float(time_s) * float(fps)), 0, frame_count - 1))


def _window_indices(center: int, frame_count: int, radius: int) -> list[int]:
    start = max(0, int(center) - int(radius))
    stop = min(frame_count, int(center) + int(radius) + 1)
    return list(range(start, stop))


def camera_projection(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_name: str,
    point: np.ndarray,
    sphere_radius: float,
    width: int,
    height: int,
    frame_index: int,
    time_s: float,
) -> ProjectionRecord:
    camera_id = _name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    camera_pos = data.cam_xpos[camera_id].copy()
    camera_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()
    local = camera_mat.T @ (np.asarray(point, dtype=np.float64) - camera_pos)
    depth = float(-local[2])
    in_front = bool(depth > 1e-6)

    pixel_xy: list[float] | None = None
    pixel_radius = 0.0
    visible = False
    comfortably_visible = False
    trackable = False
    if in_front:
        fovy = math.radians(float(model.cam_fovy[camera_id]))
        fy = float(height) / (2.0 * math.tan(0.5 * fovy))
        fx = fy
        cx = 0.5 * (float(width) - 1.0)
        cy = 0.5 * (float(height) - 1.0)
        x = cx + fx * float(local[0]) / depth
        y = cy - fy * float(local[1]) / depth
        pixel_radius = max(0.0, float(sphere_radius) * fy / depth)
        pixel_xy = [float(x), float(y)]
        visible = bool(
            -pixel_radius <= x <= float(width - 1) + pixel_radius
            and -pixel_radius <= y <= float(height - 1) + pixel_radius
        )
        margin_x = max(8.0, 0.06 * float(width))
        margin_y = max(8.0, 0.06 * float(height))
        comfortably_visible = bool(
            margin_x <= x <= float(width) - margin_x
            and margin_y <= y <= float(height) - margin_y
        )
        trackable = bool(visible and pixel_radius >= 3.0)

    return ProjectionRecord(
        frame_index=int(frame_index),
        time_s=float(time_s),
        ball_position=[float(v) for v in np.asarray(point, dtype=np.float64)],
        camera_position=[float(v) for v in camera_pos],
        in_front=in_front,
        visible=visible,
        comfortably_visible=comfortably_visible,
        trackable=trackable,
        pixel_xy=pixel_xy,
        pixel_radius=float(pixel_radius),
        depth=float(depth),
    )


class BallVisibilityRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_names: tuple[str, ...] | list[str],
        width: int,
        height: int,
        ball_body_name: str,
        ball_radius: float,
    ):
        self.model = model
        self.camera_names = tuple(camera_names)
        self.width = int(width)
        self.height = int(height)
        self.ball_body_id = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball_body_name)
        self.ball_radius = float(ball_radius)
        self.records: dict[str, list[ProjectionRecord]] = {
            camera_name: [] for camera_name in self.camera_names
        }
        self.ball_positions: list[list[float]] = []

    @property
    def frame_count(self) -> int:
        if not self.records:
            return 0
        return len(next(iter(self.records.values())))

    def add_frame(self, data: mujoco.MjData, *, frame_index: int, time_s: float) -> None:
        point = data.xpos[self.ball_body_id].copy()
        self.ball_positions.append([float(v) for v in point])
        for camera_name in self.camera_names:
            self.records[camera_name].append(
                camera_projection(
                    self.model,
                    data,
                    camera_name=camera_name,
                    point=point,
                    sphere_radius=self.ball_radius,
                    width=self.width,
                    height=self.height,
                    frame_index=frame_index,
                    time_s=time_s,
                )
            )

    def _event_indices(self, controller_result: dict) -> dict[str, int]:
        frame_count = self.frame_count
        fps = 1.0
        if frame_count >= 2:
            # The records are synchronized and evenly sampled. The first two
            # timestamps are enough to recover fps without plumbing it through.
            first = next(iter(self.records.values()))
            dt = first[1].time_s - first[0].time_s
            if dt > 0:
                fps = 1.0 / dt
        candidates = {
            "launch": 0,
            "gripper_close": _event_frame(controller_result.get("gripper_close_time"), fps, frame_count),
            "first_contact": _event_frame(controller_result.get("first_contact_time_s"), fps, frame_count),
            "first_pad_contact": _event_frame(controller_result.get("first_pad_contact_time_s"), fps, frame_count),
            "first_two_pad_contact": _event_frame(controller_result.get("first_two_pad_contact_time_s"), fps, frame_count),
            "grasp": _event_frame(controller_result.get("grasp_time_s"), fps, frame_count),
            "final": frame_count - 1,
        }
        return {
            name: int(value)
            for name, value in candidates.items()
            if value is not None and 0 <= int(value) < frame_count
        }

    def summarize(
        self,
        *,
        controller_result: dict,
        view_roles: dict[str, str],
    ) -> dict:
        frame_count = self.frame_count
        event_indices = self._event_indices(controller_result)
        ball_positions = np.asarray(self.ball_positions, dtype=np.float64)
        envelope = {
            "min": [float(v) for v in ball_positions.min(axis=0)],
            "max": [float(v) for v in ball_positions.max(axis=0)],
        } if len(ball_positions) else {"min": [], "max": []}
        views = {}
        all_passed = True
        for camera_name, records in self.records.items():
            role = view_roles.get(camera_name, "side")
            visible = np.asarray([record.visible for record in records], dtype=bool)
            comfortable = np.asarray([record.comfortably_visible for record in records], dtype=bool)
            trackable = np.asarray([record.trackable for record in records], dtype=bool)
            pixel_radius = np.asarray([record.pixel_radius for record in records], dtype=np.float64)

            if role == "on_device":
                decisive_centers = [
                    event_indices[name]
                    for name in ("gripper_close", "first_pad_contact", "first_two_pad_contact", "grasp", "final")
                    if name in event_indices
                ]
                if decisive_centers:
                    decisive_indices = sorted(
                        {
                            idx
                            for center in decisive_centers
                            for idx in _window_indices(center, frame_count, radius=6)
                        }
                    )
                else:
                    decisive_indices = list(range(max(0, frame_count - 16), frame_count))
                important_indices = decisive_indices
                min_visible_ratio = 0.45
                min_trackable_ratio = 0.35
                require_all_events_comfortable = False
            else:
                important_indices = list(range(frame_count))
                min_visible_ratio = 0.82
                min_trackable_ratio = 0.70
                require_all_events_comfortable = True

            if important_indices:
                important_visible_ratio = float(visible[important_indices].mean())
                important_trackable_ratio = float(trackable[important_indices].mean())
            else:
                important_visible_ratio = 0.0
                important_trackable_ratio = 0.0

            event_visibility = {}
            for name, idx in event_indices.items():
                record = records[idx]
                event_visibility[name] = {
                    "frame_index": int(idx),
                    "visible": bool(record.visible),
                    "comfortably_visible": bool(record.comfortably_visible),
                    "trackable": bool(record.trackable),
                    "pixel_xy": record.pixel_xy,
                    "pixel_radius": float(record.pixel_radius),
                    "depth": float(record.depth),
                }
            if require_all_events_comfortable:
                events_ok = all(
                    item["visible"] and item["comfortably_visible"] and item["trackable"]
                    for item in event_visibility.values()
                )
            else:
                decisive_events = [
                    event_visibility[name]
                    for name in ("gripper_close", "first_pad_contact", "first_two_pad_contact", "grasp", "final")
                    if name in event_visibility
                ]
                events_ok = any(item["visible"] and item["trackable"] for item in decisive_events)

            passed = bool(
                important_visible_ratio >= min_visible_ratio
                and important_trackable_ratio >= min_trackable_ratio
                and events_ok
            )
            all_passed = all_passed and passed
            views[camera_name] = {
                "role": role,
                "important_frame_count": int(len(important_indices)),
                "important_visible_ratio": important_visible_ratio,
                "important_trackable_ratio": important_trackable_ratio,
                "all_frame_visible_ratio": float(visible.mean()) if len(visible) else 0.0,
                "all_frame_comfortable_ratio": float(comfortable.mean()) if len(comfortable) else 0.0,
                "all_frame_trackable_ratio": float(trackable.mean()) if len(trackable) else 0.0,
                "min_pixel_radius": float(pixel_radius.min()) if len(pixel_radius) else 0.0,
                "median_pixel_radius": float(np.median(pixel_radius)) if len(pixel_radius) else 0.0,
                "max_pixel_radius": float(pixel_radius.max()) if len(pixel_radius) else 0.0,
                "event_visibility": event_visibility,
                "passed": passed,
            }

        return {
            "passed": bool(all_passed),
            "frame_count": int(frame_count),
            "ball_world_envelope": envelope,
            "event_indices": event_indices,
            "views": views,
            "requirements": {
                "side_views": "ball visible and trackable for most important frames; launch/contact/interception/final events not near frame edge",
                "on_device": "decisive interaction phase visible and trackable when full trajectory is not feasible",
            },
        }
