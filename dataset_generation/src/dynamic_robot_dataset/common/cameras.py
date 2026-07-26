"""Camera calibration serialization and right-handed WXYZ coordinate helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import SchemaValidationError


def _finite(values: Iterable[float], expected: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected or not all(math.isfinite(value) for value in result):
        raise SchemaValidationError(f"{name} must contain {expected} finite values")
    return result


def normalize_quaternion_wxyz(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    """Normalize a WXYZ quaternion, choosing a deterministic sign."""

    w, x, y, z = _finite(quaternion, 4, "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise SchemaValidationError("Quaternion norm is zero")
    normalized = tuple(value / norm for value in (w, x, y, z))
    if normalized[0] < 0:
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def quaternion_xyzw_to_wxyz(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert XYZW input ordering to canonical WXYZ ordering."""

    x, y, z, w = _finite(quaternion, 4, "quaternion")
    return normalize_quaternion_wxyz((w, x, y, z))


def quaternion_wxyz_to_xyzw(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert canonical WXYZ ordering to XYZW ordering."""

    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return x, y, z, w


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> tuple[float, ...]:
    """Return a row-major 3×3 rotation matrix from a WXYZ quaternion."""

    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return (
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )


def pose_to_matrix(position_m: Sequence[float], quaternion_wxyz: Sequence[float]) -> tuple[float, ...]:
    """Create a row-major homogeneous transform from position and WXYZ pose."""

    px, py, pz = _finite(position_m, 3, "position_m")
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    return (
        rotation[0], rotation[1], rotation[2], px,
        rotation[3], rotation[4], rotation[5], py,
        rotation[6], rotation[7], rotation[8], pz,
        0.0, 0.0, 0.0, 1.0,
    )


def multiply_matrix4(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    """Multiply two row-major 4×4 matrices."""

    a = _finite(left, 16, "left matrix")
    b = _finite(right, 16, "right matrix")
    return tuple(sum(a[row * 4 + k] * b[k * 4 + column] for k in range(4)) for row in range(4) for column in range(4))


def invert_rigid_transform(matrix: Sequence[float]) -> tuple[float, ...]:
    """Invert a row-major homogeneous rigid transform."""

    m = _finite(matrix, 16, "transform")
    if any(abs(m[index] - expected) > 1e-6 for index, expected in zip((12, 13, 14, 15), (0, 0, 0, 1))):
        raise SchemaValidationError("Transform bottom row must be [0, 0, 0, 1]")
    rotation_t = (m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10])
    translation = (m[3], m[7], m[11])
    inverse_translation = tuple(
        -sum(rotation_t[row * 3 + column] * translation[column] for column in range(3))
        for row in range(3)
    )
    return (
        rotation_t[0], rotation_t[1], rotation_t[2], inverse_translation[0],
        rotation_t[3], rotation_t[4], rotation_t[5], inverse_translation[1],
        rotation_t[6], rotation_t[7], rotation_t[8], inverse_translation[2],
        0.0, 0.0, 0.0, 1.0,
    )


def transform_point(matrix: Sequence[float], point: Sequence[float]) -> tuple[float, float, float]:
    """Apply a homogeneous transform to a 3D point."""

    m = _finite(matrix, 16, "transform")
    x, y, z = _finite(point, 3, "point")
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )


@dataclass(slots=True, frozen=True)
class CameraCalibration:
    """Complete calibration for one synchronized observation stream."""

    camera_name: str
    intrinsic_matrix: tuple[float, ...]
    world_to_camera: tuple[float, ...]
    camera_to_world: tuple[float, ...]
    width: int = 832
    height: int = 480
    fps: float = 30.0
    distortion_model: str = "none"
    distortion_coefficients: tuple[float, ...] = ()
    near_m: float = 0.01
    far_m: float = 100.0
    renderer: str = "mujoco"

    def validate(self, *, inverse_tolerance: float = 1e-5) -> None:
        if not self.camera_name:
            raise SchemaValidationError("camera_name is required")
        intrinsic = _finite(self.intrinsic_matrix, 9, "intrinsic_matrix")
        world_to_camera = _finite(self.world_to_camera, 16, "world_to_camera")
        camera_to_world = _finite(self.camera_to_world, 16, "camera_to_world")
        if self.width <= 0 or self.height <= 0 or not math.isfinite(self.fps) or self.fps <= 0:
            raise SchemaValidationError("Camera resolution and FPS must be positive")
        if intrinsic[0] <= 0 or intrinsic[4] <= 0:
            raise SchemaValidationError("Camera focal lengths fx/fy must be positive")
        if any(abs(intrinsic[index] - expected) > 1e-8 for index, expected in ((3, 0.0), (6, 0.0), (7, 0.0), (8, 1.0))):
            raise SchemaValidationError("Intrinsic matrix must use canonical pinhole form")
        if abs(intrinsic[1]) > 1e-8:
            raise SchemaValidationError("Canonical intrinsic skew must be zero")
        if not (0.0 <= intrinsic[2] <= self.width and 0.0 <= intrinsic[5] <= self.height):
            raise SchemaValidationError("Camera principal point lies outside the image")
        if not all(math.isfinite(value) for value in self.distortion_coefficients):
            raise SchemaValidationError("Distortion coefficients must be finite")
        if self.distortion_model == "none" and self.distortion_coefficients:
            raise SchemaValidationError("distortion_model=none requires no coefficients")
        if not 0 < self.near_m < self.far_m:
            raise SchemaValidationError("Camera clipping planes must satisfy 0 < near < far")

        def validate_rigid_transform(matrix: tuple[float, ...], name: str) -> None:
            if any(
                abs(matrix[index] - expected) > inverse_tolerance
                for index, expected in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0))
            ):
                raise SchemaValidationError(f"{name} bottom row is not homogeneous rigid form")
            rotation = (
                matrix[0], matrix[1], matrix[2],
                matrix[4], matrix[5], matrix[6],
                matrix[8], matrix[9], matrix[10],
            )
            gram = tuple(
                sum(rotation[row * 3 + axis] * rotation[column * 3 + axis] for axis in range(3))
                for row in range(3)
                for column in range(3)
            )
            identity3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            if max(abs(value - expected) for value, expected in zip(gram, identity3)) > inverse_tolerance:
                raise SchemaValidationError(f"{name} rotation is not orthonormal")
            determinant = (
                rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
                - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
                + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
            )
            if abs(determinant - 1.0) > inverse_tolerance:
                raise SchemaValidationError(f"{name} rotation determinant must be +1")

        validate_rigid_transform(world_to_camera, "world_to_camera")
        validate_rigid_transform(camera_to_world, "camera_to_world")
        identity = multiply_matrix4(self.world_to_camera, self.camera_to_world)
        expected = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        if max(abs(value - target) for value, target in zip(identity, expected)) > inverse_tolerance:
            raise SchemaValidationError("world_to_camera and camera_to_world are not inverses")

    def project_world(self, point_world_m: Sequence[float]) -> tuple[float, float, float]:
        """Project a world point to ``(pixel_x, pixel_y, camera_depth)``."""

        x, y, z = transform_point(self.world_to_camera, point_world_m)
        if z <= 0:
            raise ValueError("Point lies behind the camera")
        k = self.intrinsic_matrix
        pixel_x = (k[0] * x + k[1] * y) / z + k[2]
        pixel_y = (k[3] * x + k[4] * y) / z + k[5]
        return pixel_x, pixel_y, z

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraCalibration":
        fields = dict(value)
        for key in (
            "intrinsic_matrix",
            "world_to_camera",
            "camera_to_world",
            "distortion_coefficients",
        ):
            fields[key] = tuple(fields.get(key, ()))
        calibration = cls(**fields)
        calibration.validate()
        return calibration
