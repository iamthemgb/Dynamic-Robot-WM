"""Canonical H.264 video encoding, probing, and frame extraction via FFmpeg."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator

from .paths import ExistingOutputError, ensure_not_source_path


class VideoEncodingError(RuntimeError):
    """FFmpeg could not encode or validate a canonical observation stream."""


@dataclass(slots=True, frozen=True)
class VideoSpec:
    """Encoding contract for a canonical or derived observation video."""

    width: int = 832
    height: int = 480
    fps_num: int = 30
    fps_den: int = 1
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    input_pixel_format: str = "rgb24"
    crf: int = 18
    preset: str = "medium"

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("H.264/yuv420p dimensions must be positive and even")
        if self.fps_num <= 0 or self.fps_den <= 0:
            raise ValueError("FPS must be a positive rational")
        if self.pixel_format != "yuv420p":
            raise ValueError("Canonical output pixel format is yuv420p")


@dataclass(slots=True, frozen=True)
class VideoProbe:
    """Decoded FFprobe facts for one video stream."""

    path: str
    width: int
    height: int
    codec_name: str
    pixel_format: str
    fps_num: int
    fps_den: int
    time_base_num: int
    time_base_den: int
    frame_count: int
    duration_s: float

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None and name == "ffmpeg":
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError, OSError):
            executable = None
    if executable is None:
        raise VideoEncodingError(f"Required executable is not on PATH: {name}")
    return executable


def _parse_fraction(value: str | None, default: str = "0/1") -> tuple[int, int]:
    fraction = Fraction(value or default)
    return fraction.numerator, fraction.denominator


def probe_video(path: str | Path, *, count_frames: bool = True) -> VideoProbe:
    """Probe the first video stream using FFprobe JSON output."""

    video = Path(path).resolve(strict=True)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return _probe_video_av(video, count_frames=count_frames)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command.extend(["-show_entries", "stream=width,height,codec_name,pix_fmt,avg_frame_rate,time_base,nb_frames,nb_read_frames,duration", "-of", "json", str(video)])
    process = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise VideoEncodingError(f"FFprobe failed for {video}: {process.stderr.strip()}")
    payload = json.loads(process.stdout)
    if not payload.get("streams"):
        raise VideoEncodingError(f"No video stream found: {video}")
    stream = payload["streams"][0]
    fps_num, fps_den = _parse_fraction(stream.get("avg_frame_rate"))
    time_num, time_den = _parse_fraction(stream.get("time_base"), "1/1")
    frame_count_raw = stream.get("nb_read_frames") or stream.get("nb_frames") or 0
    duration_raw = stream.get("duration")
    frame_count = int(frame_count_raw) if str(frame_count_raw).isdigit() else 0
    duration = float(duration_raw) if duration_raw not in (None, "N/A") else (
        frame_count * fps_den / fps_num if fps_num else 0.0
    )
    return VideoProbe(
        path=str(video),
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec_name=str(stream.get("codec_name") or ""),
        pixel_format=str(stream.get("pix_fmt") or ""),
        fps_num=fps_num,
        fps_den=fps_den,
        time_base_num=time_num,
        time_base_den=time_den,
        frame_count=frame_count,
        duration_s=duration,
    )


def _probe_video_av(video: Path, *, count_frames: bool) -> VideoProbe:
    """PyAV fallback for clusters that provide FFmpeg but not FFprobe."""

    try:
        import av
    except ImportError as exc:
        raise VideoEncodingError(
            "Video probing requires ffprobe or PyAV; neither is available"
        ) from exc
    with av.open(str(video), mode="r") as container:
        if not container.streams.video:
            raise VideoEncodingError(f"No video stream found: {video}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(0, 1)
        time_base = stream.time_base or Fraction(1, 1)
        frame_count = int(stream.frames or 0)
        decoded_pts: list[float] = []
        if count_frames and frame_count <= 0:
            for frame in container.decode(stream):
                if frame.pts is not None:
                    decoded_pts.append(float(frame.pts * time_base))
            frame_count = len(decoded_pts)
        if stream.duration is not None:
            duration = float(stream.duration * time_base)
        elif decoded_pts and rate:
            duration = decoded_pts[-1] - decoded_pts[0] + float(1 / rate)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        elif frame_count and rate:
            duration = float(frame_count / rate)
        else:
            duration = 0.0
        pixel_format = stream.codec_context.format.name if stream.codec_context.format is not None else ""
        return VideoProbe(
            path=str(video),
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            codec_name=str(stream.codec_context.name or ""),
            pixel_format=str(pixel_format),
            fps_num=int(rate.numerator),
            fps_den=int(rate.denominator),
            time_base_num=int(time_base.numerator),
            time_base_den=int(time_base.denominator),
            frame_count=frame_count,
            duration_s=duration,
        )


def probe_frame_timestamps(path: str | Path) -> list[float]:
    """Return decoded presentation timestamps reported by FFprobe."""

    video = Path(path).resolve(strict=True)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        try:
            import av
        except ImportError as exc:
            raise VideoEncodingError(
                "Frame timestamp probing requires ffprobe or PyAV"
            ) from exc
        with av.open(str(video), mode="r") as container:
            if not container.streams.video:
                raise VideoEncodingError(f"No video stream found: {video}")
            stream = container.streams.video[0]
            time_base = stream.time_base
            result = [
                float(frame.pts * time_base)
                for frame in container.decode(stream)
                if frame.pts is not None and time_base is not None
            ]
        if not result:
            raise VideoEncodingError(f"No presentation timestamps found: {video}")
        return result
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ]
    process = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise VideoEncodingError(f"Could not read PTS from {video}: {process.stderr.strip()}")
    frames = json.loads(process.stdout).get("frames", [])
    result = [float(frame["best_effort_timestamp_time"]) for frame in frames if "best_effort_timestamp_time" in frame]
    if not result:
        raise VideoEncodingError(f"No presentation timestamps found: {video}")
    return result


class FrameVideoWriter:
    """Stream RGB frames to FFmpeg and atomically publish a validated MP4."""

    def __init__(self, destination: str | Path, spec: VideoSpec = VideoSpec()):
        self.destination = ensure_not_source_path(destination)
        self.spec = spec
        self.process: subprocess.Popen[bytes] | None = None
        self.temporary: Path | None = None
        self.frame_count = 0

    def __enter__(self) -> "FrameVideoWriter":
        self.spec.validate()
        if self.destination.exists():
            raise ExistingOutputError(f"Video already exists: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.destination.name}.", suffix=".mp4", dir=self.destination.parent)
        os.close(fd)
        self.temporary = Path(name)
        command = [
            _require_executable("ffmpeg"), "-v", "error", "-y",
            "-f", "rawvideo", "-pixel_format", self.spec.input_pixel_format,
            "-video_size", f"{self.spec.width}x{self.spec.height}",
            "-framerate", f"{self.spec.fps_num}/{self.spec.fps_den}", "-i", "pipe:0",
            "-an", "-c:v", self.spec.codec, "-preset", self.spec.preset,
            "-crf", str(self.spec.crf), "-pix_fmt", self.spec.pixel_format,
            "-movflags", "+faststart", str(self.temporary),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return self

    def write(self, frame: Any) -> None:
        """Write one RGB24 frame from bytes or an array-like object."""

        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Video writer is not open")
        expected = self.spec.width * self.spec.height * 3
        shape = getattr(frame, "shape", None)
        if shape is not None and tuple(shape) != (self.spec.height, self.spec.width, 3):
            raise VideoEncodingError(f"Frame shape {shape} != {(self.spec.height, self.spec.width, 3)}")
        data = frame.tobytes(order="C") if hasattr(frame, "tobytes") else bytes(frame)
        if len(data) != expected:
            raise VideoEncodingError(f"RGB frame has {len(data)} bytes, expected {expected}")
        self.process.stdin.write(data)
        self.frame_count += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        assert self.process is not None and self.temporary is not None
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if exc_type is not None or return_code != 0 or self.frame_count == 0:
            self.temporary.unlink(missing_ok=True)
            if exc_type is None:
                raise VideoEncodingError(stderr.strip() or "FFmpeg failed or no frames were written")
            return False
        try:
            probe = probe_video(self.temporary)
            validate_video_probe(probe, self.spec, expected_frames=self.frame_count)
            os.link(self.temporary, self.destination)
        except FileExistsError as error:
            raise ExistingOutputError(f"Video already exists: {self.destination}") from error
        finally:
            self.temporary.unlink(missing_ok=True)
        return False


def encode_video(frames: Iterable[Any], destination: str | Path, spec: VideoSpec = VideoSpec()) -> VideoProbe:
    """Encode an iterable of RGB frames and return validated probe metadata."""

    with FrameVideoWriter(destination, spec) as writer:
        for frame in frames:
            writer.write(frame)
    return probe_video(destination)


def validate_video_probe(
    probe: VideoProbe,
    spec: VideoSpec,
    *,
    expected_frames: int | None = None,
    fps_tolerance: float = 1e-3,
) -> None:
    """Validate resolution, codec, pixel format, FPS, and optional frame count."""

    failures: list[str] = []
    if (probe.width, probe.height) != (spec.width, spec.height):
        failures.append(f"resolution {probe.width}x{probe.height}")
    if probe.codec_name != "h264":
        failures.append(f"codec {probe.codec_name}")
    if probe.pixel_format != spec.pixel_format:
        failures.append(f"pixel format {probe.pixel_format}")
    if abs(probe.fps - spec.fps) > fps_tolerance:
        failures.append(f"FPS {probe.fps}")
    if expected_frames is not None and probe.frame_count != expected_frames:
        failures.append(f"frame count {probe.frame_count}, expected {expected_frames}")
    if failures:
        raise VideoEncodingError(f"Non-canonical video {probe.path}: {', '.join(failures)}")


def iter_rgb_frames(path: str | Path, width: int, height: int) -> Iterator[bytes]:
    """Decode RGB24 frames as bytes without requiring OpenCV or NumPy."""

    command = [
        _require_executable("ffmpeg"), "-v", "error", "-i", str(Path(path).resolve(strict=True)),
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    size = width * height * 3
    completed = False
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                completed = True
                break
            if len(data) != size:
                process.kill()
                raise VideoEncodingError(f"Truncated decoded frame in {path}")
            yield data
    finally:
        if not completed and process.poll() is None:
            process.stdout.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if not completed:
        return
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise VideoEncodingError(f"FFmpeg decode failed for {path}: {stderr.strip()}")
