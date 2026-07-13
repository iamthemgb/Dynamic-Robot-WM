"""Small deterministic rigid-sphere solver used by smoke adapters.

This is not a scripted trajectory generator.  Free flight is integrated under
constant gravity and impacts are resolved at their continuous collision time
with a normal restitution impulse and a bounded Coulomb tangential impulse.
There are no post-contact pose/velocity keyframes or outcome-conditioned resets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence


Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, value: float) -> Vec3:
    return (a[0] * value, a[1] * value, a[2] * value)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class RigidTrace:
    timestamps_s: tuple[float, ...]
    positions_m: tuple[Vec3, ...]
    velocities_mps: tuple[Vec3, ...]
    angular_velocities_radps: tuple[Vec3, ...]
    contacts: tuple[Mapping[str, object], ...]


def simulate_ballistic_sphere_with_paddle(
    *,
    initial_position_m: Vec3,
    initial_velocity_mps: Vec3,
    radius_m: float,
    mass_kg: float,
    gravity_mps2: Vec3,
    floor_restitution: float,
    floor_friction: float,
    paddle_restitution: float,
    paddle_friction: float,
    paddle_center: Callable[[float], Vec3],
    paddle_normal: Vec3,
    paddle_radius_m: float,
    duration_s: float,
    sim_hz: int,
    floor_z_m: float = 0.0,
) -> RigidTrace:
    """Integrate a sphere against a floor and a finite kinematic disk.

    Ground impacts use continuous collision detection. The paddle is represented
    by an oriented, zero-thickness disk; collision and impulse use the same
    configured normal. Each disk can be hit once, matching the interception
    tasks and preventing numerical chatter.
    """

    normal_length = norm(paddle_normal)
    if normal_length <= 0:
        raise ValueError("paddle normal must be non-zero")
    paddle_normal = scale(paddle_normal, 1.0 / normal_length)
    dt = 1.0 / sim_hz
    steps = int(round(duration_s * sim_hz))
    floor_center_z = floor_z_m + radius_m
    p = initial_position_m
    v = initial_velocity_mps
    omega = (0.0, 0.0, 0.0)
    times: list[float] = [0.0]
    positions: list[Vec3] = [p]
    velocities: list[Vec3] = [v]
    angular: list[Vec3] = [omega]
    contacts: list[Mapping[str, object]] = []
    paddle_hit = False

    def ball_position_at(time_offset_s: float, acceleration: Vec3) -> Vec3:
        return add(
            add(p, scale(v, time_offset_s)),
            scale(acceleration, 0.5 * time_offset_s * time_offset_s),
        )

    def paddle_velocity(time_s: float) -> Vec3:
        """Differentiate the supplied kinematic paddle-position curve."""

        width = min(1e-5, 0.1 * dt)
        if time_s <= width:
            return scale(
                sub(paddle_center(time_s + width), paddle_center(time_s)),
                1.0 / width,
            )
        return scale(
            sub(paddle_center(time_s + width), paddle_center(time_s - width)),
            0.5 / width,
        )

    def paddle_collision_time(
        time_s: float,
        maximum_offset_s: float,
        acceleration: Vec3,
    ) -> tuple[float, Vec3] | None:
        """Find the first sphere/oriented-disk face crossing in this substep."""

        signed_distance = dot(sub(p, paddle_center(time_s)), paddle_normal)
        face_normal = paddle_normal if signed_distance >= 0.0 else scale(paddle_normal, -1.0)

        def clearance(offset_s: float) -> float:
            position = ball_position_at(offset_s, acceleration)
            center = paddle_center(time_s + offset_s)
            return dot(sub(position, center), face_normal) - radius_m

        previous_time = 0.0
        previous_clearance = clearance(0.0)
        if previous_clearance <= 0.0:
            return None
        # Minimum-jerk paddle motion is smooth but nonlinear. Subdivision gives
        # a bracket without assuming constant surface velocity.
        for sample in range(1, 9):
            sample_time = maximum_offset_s * sample / 8.0
            sample_clearance = clearance(sample_time)
            if sample_clearance <= 0.0 < previous_clearance:
                left, right = previous_time, sample_time
                for _ in range(48):
                    midpoint = 0.5 * (left + right)
                    if clearance(midpoint) > 0.0:
                        left = midpoint
                    else:
                        right = midpoint
                candidate = right
                position = ball_position_at(candidate, acceleration)
                center = paddle_center(time_s + candidate)
                sphere_contact = sub(position, scale(face_normal, radius_m))
                radial = sub(sphere_contact, center)
                radial = sub(radial, scale(face_normal, dot(radial, face_normal)))
                relative_velocity = sub(
                    add(v, scale(acceleration, candidate)),
                    paddle_velocity(time_s + candidate),
                )
                if (
                    norm(radial) <= paddle_radius_m + 1e-9
                    and dot(relative_velocity, face_normal) < 0.0
                ):
                    return candidate, face_normal
            previous_time = sample_time
            previous_clearance = sample_clearance
        return None

    def apply_impulse(pre: Vec3, normal: Vec3, restitution: float, friction: float) -> tuple[Vec3, Vec3]:
        vn = dot(pre, normal)
        if vn >= 0:
            return pre, (0.0, 0.0, 0.0)
        normal_delta = scale(normal, -(1.0 + restitution) * vn)
        tangent = sub(pre, scale(normal, vn))
        tangent_speed = norm(tangent)
        max_tangent_delta = friction * (1.0 + restitution) * (-vn)
        tangent_delta = scale(tangent, -min(1.0, max_tangent_delta / max(tangent_speed, 1e-12)))
        after = add(pre, add(normal_delta, tangent_delta))
        return after, scale(sub(after, pre), mass_kg)

    for step in range(steps):
        remaining = dt
        local_time = step * dt
        for _ in range(6):
            resting = p[2] <= floor_center_z + 1e-8 and abs(v[2]) < 2e-3
            acceleration = gravity_mps2
            if resting:
                p = (p[0], p[1], floor_center_z)
                v = (v[0], v[1], 0.0)
                acceleration = (gravity_mps2[0], gravity_mps2[1], 0.0)
            ground_time = None if resting else _impact_time(
                p[2], v[2], acceleration[2], floor_center_z, remaining
            )
            paddle_collision = (
                None
                if paddle_hit
                else paddle_collision_time(local_time, remaining, acceleration)
            )
            paddle_time = paddle_collision[0] if paddle_collision is not None else None
            candidates = [
                (time, kind)
                for time, kind in ((ground_time, "ground"), (paddle_time, "paddle"))
                if time is not None
            ]
            if not candidates:
                p = add(add(p, scale(v, remaining)), scale(acceleration, 0.5 * remaining * remaining))
                v = add(v, scale(acceleration, remaining))
                local_time += remaining
                remaining = 0.0
                break
            event_time, kind = min(candidates)
            p = add(add(p, scale(v, event_time)), scale(acceleration, 0.5 * event_time * event_time))
            v = add(v, scale(acceleration, event_time))
            local_time += event_time
            remaining -= event_time
            pre = v
            if kind == "ground":
                p = (p[0], p[1], floor_center_z)
                v, impulse = apply_impulse(pre, (0.0, 0.0, 1.0), floor_restitution, floor_friction)
                if v[2] < 2e-3:
                    v = (v[0], v[1], 0.0)
                contact_point = [p[0], p[1], floor_z_m]
                object_b = "floor"
                normal = (0.0, 0.0, 1.0)
            else:
                if paddle_collision is None:
                    raise RuntimeError("Paddle collision lost before resolution")
                normal = paddle_collision[1]
                surface_velocity = paddle_velocity(local_time)
                relative_pre = sub(pre, surface_velocity)
                relative_post, _ = apply_impulse(
                    relative_pre, normal, paddle_restitution, paddle_friction
                )
                v = add(surface_velocity, relative_post)
                impulse = scale(sub(v, pre), mass_kg)
                contact_point = list(sub(p, scale(normal, radius_m)))
                object_b = "robot_paddle"
                paddle_hit = True
            if kind == "ground":
                relative_pre = pre
                relative_post = v
            contacts.append(
                {
                    "timestamp": local_time,
                    "object_a": "ball",
                    "object_b": object_b,
                    "contact_point_m": contact_point,
                    "contact_normal": list(normal),
                    "penetration_depth_m": 0.0,
                    "normal_impulse_Ns": abs(dot(impulse, normal)),
                    "impulse_Ns": list(impulse),
                    "relative_velocity_pre_mps": list(relative_pre),
                    "relative_velocity_post_mps": list(relative_post),
                    "object_velocity_pre_mps": list(pre),
                    "object_velocity_post_mps": list(v),
                }
            )
            if remaining <= 1e-12:
                break
            # Advance a tiny amount after an event to avoid detecting the same root.
            epsilon = min(1e-9, remaining)
            p = add(add(p, scale(v, epsilon)), scale(acceleration, 0.5 * epsilon * epsilon))
            v = add(v, scale(acceleration, epsilon))
            local_time += epsilon
            remaining -= epsilon
        times.append((step + 1) * dt)
        positions.append(p)
        velocities.append(v)
        angular.append(omega)
    return RigidTrace(tuple(times), tuple(positions), tuple(velocities), tuple(angular), tuple(contacts))


def _impact_time(z: float, vz: float, gz: float, radius: float, dt: float) -> float | None:
    c = z - radius
    if c <= 1e-10 and vz <= 0.0:
        return 0.0
    if abs(gz) < 1e-12:
        if vz >= 0.0:
            return None
        candidate = -c / vz
        return candidate if 0.0 <= candidate <= dt else None
    discriminant = vz * vz - 2.0 * gz * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    candidates = [(-vz + root) / gz, (-vz - root) / gz]
    valid = [value for value in candidates if -1e-12 <= value <= dt + 1e-12]
    if c <= 1e-10 and vz > 0.0:
        valid = [value for value in valid if value > 1e-10]
    return max(0.0, min(valid)) if valid else None


def simulate_ballistic_sphere(
    *,
    initial_position_m: Vec3,
    initial_velocity_mps: Vec3,
    radius_m: float,
    mass_kg: float,
    gravity_mps2: Vec3,
    restitution: float,
    friction: float,
    duration_s: float,
    sim_hz: int,
    floor_z_m: float = 0.0,
    rolling_resistance_mps2: float = 0.0,
    object_id: str = "ball",
    surface_id: str = "floor",
) -> RigidTrace:
    if radius_m <= 0 or mass_kg <= 0:
        raise ValueError("radius and mass must be positive")
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must be in [0, 1]")
    if friction < 0.0:
        raise ValueError("friction must be non-negative")
    dt = 1.0 / sim_hz
    steps = int(round(duration_s * sim_hz))
    p = initial_position_m
    v = initial_velocity_mps
    omega = (0.0, 0.0, 0.0)
    times: list[float] = [0.0]
    positions: list[Vec3] = [p]
    velocities: list[Vec3] = [v]
    angular: list[Vec3] = [omega]
    contacts: list[Mapping[str, object]] = []
    floor_center_z = floor_z_m + radius_m

    for step in range(steps):
        remaining = dt
        local_time = step * dt
        # At most four contacts in one small step; normally there is zero or one.
        for _ in range(4):
            resting = p[2] <= floor_center_z + 1e-8 and abs(v[2]) < 2e-3
            acceleration = gravity_mps2
            if resting:
                acceleration = (gravity_mps2[0], gravity_mps2[1], 0.0)
                p = (p[0], p[1], floor_center_z)
                v = (v[0], v[1], 0.0)
                horizontal_speed = math.hypot(v[0], v[1])
                if horizontal_speed > 0 and rolling_resistance_mps2 > 0:
                    drop = min(horizontal_speed, rolling_resistance_mps2 * remaining)
                    ratio = (horizontal_speed - drop) / horizontal_speed
                    v = (v[0] * ratio, v[1] * ratio, 0.0)
                    omega = (-v[1] / radius_m, v[0] / radius_m, 0.0)

            impact = None if resting else _impact_time(
                p[2], v[2], acceleration[2], floor_center_z, remaining
            )
            advance = remaining if impact is None else impact
            p = add(add(p, scale(v, advance)), scale(acceleration, 0.5 * advance * advance))
            v = add(v, scale(acceleration, advance))
            local_time += advance
            remaining -= advance
            if impact is None:
                break

            pre = v
            normal_speed = pre[2]
            tangential = (pre[0], pre[1], 0.0)
            tangent_speed = norm(tangential)
            normal_impulse = mass_kg * (1.0 + restitution) * max(0.0, -normal_speed)
            max_tangent_delta = friction * normal_impulse / mass_kg
            if tangent_speed > 0.0:
                remaining_tangent_speed = max(0.0, tangent_speed - max_tangent_delta)
                ratio = remaining_tangent_speed / tangent_speed
                tangent_after = scale(tangential, ratio)
            else:
                tangent_after = tangential
            v = (tangent_after[0], tangent_after[1], -restitution * normal_speed)
            p = (p[0], p[1], floor_center_z)
            impulse = (mass_kg * (v[0] - pre[0]), mass_kg * (v[1] - pre[1]), normal_impulse)
            contacts.append(
                {
                    "timestamp": local_time,
                    "object_a": object_id,
                    "object_b": surface_id,
                    "contact_point_m": [p[0], p[1], floor_z_m],
                    "contact_normal": [0.0, 0.0, 1.0],
                    "penetration_depth_m": 0.0,
                    "normal_impulse_Ns": normal_impulse,
                    "impulse_Ns": list(impulse),
                    "relative_velocity_pre_mps": list(pre),
                    "relative_velocity_post_mps": list(v),
                }
            )
            if v[2] < 2e-3:
                v = (v[0], v[1], 0.0)
            if remaining <= 1e-12:
                break
        times.append((step + 1) * dt)
        positions.append(p)
        velocities.append(v)
        angular.append(omega)

    return RigidTrace(tuple(times), tuple(positions), tuple(velocities), tuple(angular), tuple(contacts))


def sample_trace(trace: RigidTrace, sim_hz: int, output_hz: int) -> list[int]:
    if sim_hz % output_hz:
        raise ValueError("sim_hz must be divisible by output_hz")
    stride = sim_hz // output_hz
    return list(range(0, len(trace.timestamps_s), stride))


def finite_difference_qc(
    trace: RigidTrace,
    *,
    gravity_mps2: Vec3,
    contact_guard_s: float,
    velocity_abs_tolerance_mps: float = 0.08,
    acceleration_abs_tolerance_mps2: float = 0.8,
) -> dict[str, object]:
    contact_times = [float(event["timestamp"]) for event in trace.contacts]
    max_velocity_error = 0.0
    max_acceleration_error = 0.0
    checked_acceleration = 0
    for index in range(len(trace.timestamps_s) - 1):
        dt = trace.timestamps_s[index + 1] - trace.timestamps_s[index]
        dp = sub(trace.positions_m[index + 1], trace.positions_m[index])
        estimated_velocity = scale(dp, 1.0 / dt)
        midpoint_velocity = scale(
            add(trace.velocities_mps[index], trace.velocities_mps[index + 1]), 0.5
        )
        near_contact = any(
            abs(trace.timestamps_s[index] - time) <= contact_guard_s for time in contact_times
        )
        floor_height = min(point[2] for point in trace.positions_m)
        grounded = (
            trace.positions_m[index][2] <= floor_height + 1e-7
            and abs(trace.velocities_mps[index][2]) < 2e-3
        )
        if not near_contact and not grounded:
            max_velocity_error = max(
                max_velocity_error,
                norm(sub(estimated_velocity, midpoint_velocity)),
            )
            dv = sub(trace.velocities_mps[index + 1], trace.velocities_mps[index])
            estimated_acceleration = scale(dv, 1.0 / dt)
            max_acceleration_error = max(
                max_acceleration_error, norm(sub(estimated_acceleration, gravity_mps2))
            )
            checked_acceleration += 1
    finite = all(
        math.isfinite(value)
        for row in (*trace.positions_m, *trace.velocities_mps)
        for value in row
    )
    return {
        "finite_state": finite,
        "position_velocity_consistent": max_velocity_error <= velocity_abs_tolerance_mps,
        "free_flight_acceleration_consistent": checked_acceleration == 0
        or max_acceleration_error <= acceleration_abs_tolerance_mps2,
        "max_velocity_fd_error_mps": max_velocity_error,
        "max_free_flight_acceleration_error_mps2": max_acceleration_error,
        "free_flight_intervals_checked": checked_acceleration,
    }


def analytic_first_impact(
    *,
    initial_height_m: float,
    initial_vertical_velocity_mps: float,
    floor_center_height_m: float,
    gravity_z_mps2: float,
) -> tuple[float, float] | None:
    """Validation oracle returning impact time and vertical pre-impact speed."""

    impact = _impact_time(
        initial_height_m,
        initial_vertical_velocity_mps,
        gravity_z_mps2,
        floor_center_height_m,
        1e6,
    )
    if impact is None:
        return None
    return impact, initial_vertical_velocity_mps + gravity_z_mps2 * impact


def moving_target_position(
    time_s: float,
    *,
    start: Vec3,
    target: Vec3,
    start_time_s: float,
    travel_time_s: float,
) -> Vec3:
    if time_s <= start_time_s:
        return start
    alpha = min(1.0, (time_s - start_time_s) / max(travel_time_s, 1e-9))
    # Minimum-jerk interpolation avoids discontinuous action velocity.
    blend = alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))
    return add(start, scale(sub(target, start), blend))
