"""Shared wire format, geometry and control for the closed-loop agent and model server (numpy + stdlib only)."""
from __future__ import annotations

import json
import math
import socket
import struct
from collections import deque

import numpy as np

N_HISTORY = 16
N_FUTURE = 64
DT = 0.1                         # s
SIM_HZ = 20
TICKS_PER_STEP = int(round(SIM_HZ * DT))
N_FRAMES = 4
STUDENT_HW = (360, 640)
IMG_W, IMG_H = 1600, 900
FOCAL_PX = 1142.5184053936916
TELE_FOV_DEG = 30.0
REAR_AXLE_OFFSET_M = 1.25

TELE_SLOT = "camera_front_tele_30fov"
SLOT_SENSOR = {
    "camera_cross_left_120fov": "CAM_FRONT_LEFT",
    "camera_front_wide_120fov": "CAM_FRONT",
    "camera_cross_right_120fov": "CAM_FRONT_RIGHT",
    TELE_SLOT: "CAM_FRONT",
}
CAMERA_RIG = {
    "CAM_FRONT":       dict(x=0.80, y=0.0, z=1.60, yaw=0.0),
    "CAM_FRONT_LEFT":  dict(x=0.27, y=-0.55, z=1.60, yaw=-55.0),
    "CAM_FRONT_RIGHT": dict(x=0.27, y=0.55, z=1.60, yaw=55.0),
}
CAMERA_FOV = 70
GPS_X = -1.4

LANEFOLLOW = 4
HINT_LEFT, HINT_RIGHT = 1, 2
EARTH_RADIUS_EQUA = 6378137.0


def send_msg(sock: socket.socket, fields: dict) -> None:
    meta, arrays, blobs = {}, [], []
    for k, v in fields.items():
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            arrays.append([k, a.dtype.str, list(a.shape)])
            blobs.append(a.tobytes())
        else:
            meta[k] = v
    head = json.dumps({"meta": meta, "arrays": arrays}).encode()
    sock.sendall(struct.pack(">I", len(head)) + head + b"".join(blobs))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict:
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    head = json.loads(_recv_exact(sock, n))
    out = dict(head["meta"])
    for name, dtype, shape in head["arrays"]:
        dt = np.dtype(dtype)
        size = int(np.prod(shape)) * dt.itemsize
        out[name] = np.frombuffer(_recv_exact(sock, size), dtype=dt).reshape(shape).copy()
    return out


def connect(path: str, timeout: float | None = None) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    return s


def gps_to_location(lat: float, lon: float, lat_ref: float, lon_ref: float) -> np.ndarray:
    """GNSS reading -> CARLA world (x, y)."""
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = lon * (math.pi * EARTH_RADIUS_EQUA * scale) / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    return np.array([x, y])


def solve_latlon_ref(lat: float, lon: float, x: float, y: float) -> tuple[float, float]:
    """Solve the map's geo-reference from one route point known in both frames."""
    def g(lat_ref):
        s = math.cos(math.radians(lat_ref))
        return (s * EARTH_RADIUS_EQUA * (math.log(math.tan(math.radians(90.0 + lat_ref) / 2.0))
                                         - math.log(math.tan(math.radians(90.0 + lat) / 2.0))) - y)
    lo, hi = lat - 1.0, lat + 1.0
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        raise ValueError(f"no geo-reference within 1 deg of lat {lat} for y={y}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    lat_ref = 0.5 * (lo + hi)
    s = math.cos(math.radians(lat_ref))
    lon_ref = lon - x * 180.0 / (math.pi * EARTH_RADIUS_EQUA * s)
    return lat_ref, lon_ref


class RoutePlanner:
    """Route waypoint popper matching Bench2DriveZoo's RoutePlanner."""

    def __init__(self, min_distance: float = 4.0, max_distance: float = 50.0):
        self.route: deque = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

    def set_route(self, points) -> None:
        self.route.clear()
        for xy, cmd in points:
            self.route.append((np.asarray(xy, dtype=np.float64), int(cmd)))

    def run_step(self, pos: np.ndarray):
        if len(self.route) == 1:
            return self.route[0], self.route[0]
        to_pop, farthest, cum = 0, -np.inf, 0.0
        for i in range(1, len(self.route)):
            if cum > self.max_distance:
                break
            cum += np.linalg.norm(self.route[i][0] - self.route[i - 1][0])
            d = np.linalg.norm(self.route[i][0] - pos)
            if d <= self.min_distance and d > farthest:
                farthest, to_pop = d, i
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        return self.route[0], self.route[1]

    def command_ahead(self, lookahead_m: float) -> int:
        """First non-LANEFOLLOW command within `lookahead_m`, else the near command."""
        near = self.route[1] if len(self.route) > 1 else self.route[0]
        if near[1] not in (LANEFOLLOW, -1):
            return near[1]
        cum = 0.0
        for i in range(2, len(self.route)):
            cum += float(np.linalg.norm(self.route[i][0] - self.route[i - 1][0]))
            if cum > lookahead_m:
                break
            if self.route[i][1] not in (LANEFOLLOW, -1):
                return self.route[i][1]
        return near[1]


def hint_command(cmd: int) -> str:
    """RoadOption -> route hint key."""
    return {HINT_LEFT: "turn_left", HINT_RIGHT: "turn_right"}.get(int(cmd), "straight")


def compass_to_yaw(compass: float) -> float:
    """IMU compass -> CARLA yaw (rad)."""
    return float(compass) - math.pi / 2.0


class PoseTracker:
    """Dead-reckoned rear-axle pose from speed and compass, one entry per sim tick."""

    def __init__(self, sim_hz: int = SIM_HZ, keep_s: float = 12.0):
        self.dt = 1.0 / sim_hz
        self.stride = int(round(sim_hz * DT))
        self.poses: deque = deque(maxlen=int(keep_s * sim_hz))
        self._xy = np.zeros(2)
        self._last = None

    def update(self, speed: float, compass: float) -> None:
        yaw = compass_to_yaw(compass)
        if self._last is not None:
            v0, y0 = self._last
            # trapezoid: mean speed along mean heading
            dyaw = math.atan2(math.sin(yaw - y0), math.cos(yaw - y0))
            ym = y0 + 0.5 * dyaw
            self._xy = self._xy + 0.5 * (v0 + speed) * self.dt * np.array([math.cos(ym), math.sin(ym)])
        self._last = (float(speed), yaw)
        self.poses.append((self._xy[0], self._xy[1], yaw))

    def pose(self, ticks_ago: int = 0):
        return self.poses[max(len(self.poses) - 1 - ticks_ago, 0)]

    def history(self) -> tuple[np.ndarray, np.ndarray]:
        """Ego history (xyz (16,3), rot (16,3,3)) in the current ego frame."""
        x0, y0, yaw0 = self.pose(0)
        c, s = math.cos(yaw0), math.sin(yaw0)
        xyz = np.zeros((N_HISTORY, 3), dtype=np.float32)
        rot = np.zeros((N_HISTORY, 3, 3), dtype=np.float32)
        for i in range(N_HISTORY):
            x, y, yaw = self.pose((N_HISTORY - 1 - i) * self.stride)
            dx, dy = x - x0, y - y0
            fwd = c * dx + s * dy
            right = -s * dx + c * dy                 # CARLA: +y is right
            xyz[i] = (fwd, -right, 0.0)
            dyaw = -(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0)))   # + = left
            cy, sy = math.cos(dyaw), math.sin(dyaw)
            rot[i] = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
        xyz[-1] = 0.0
        rot[-1] = np.eye(3)
        return xyz, rot

    def to_current(self, xy_ego: np.ndarray, ticks_ago: int) -> np.ndarray:
        """Ego-frame points from `ticks_ago` -> current ego frame."""
        if ticks_ago <= 0:
            return np.asarray(xy_ego, dtype=np.float64)
        xp, yp, yawp = self.pose(ticks_ago)
        x0, y0, yaw0 = self.pose(0)
        p = np.asarray(xy_ego, dtype=np.float64)
        cp, sp = math.cos(yawp), math.sin(yawp)
        # ego(past) -> local CARLA frame (y right = -left)
        wx = xp + cp * p[:, 0] - sp * (-p[:, 1])
        wy = yp + sp * p[:, 0] + cp * (-p[:, 1])
        c, s = math.cos(yaw0), math.sin(yaw0)
        dx, dy = wx - x0, wy - y0
        return np.stack([c * dx + s * dy, -(-s * dx + c * dy)], axis=1)


def tele_crop_box(f_px: float = FOCAL_PX, fov_deg: float = TELE_FOV_DEG,
                  w: int = IMG_W, h: int = IMG_H) -> tuple[int, int, int, int]:
    """Centre crop box emulating the 30-deg tele camera."""
    half_w = f_px * math.tan(math.radians(fov_deg / 2.0))
    crop_w = int(round(2 * half_w))
    crop_h = int(round(crop_w * h / w))
    x0 = int(round(w / 2.0 - crop_w / 2)); y0 = int(round(h / 2.0 - crop_h / 2))
    return x0, y0, x0 + crop_w, y0 + crop_h


def student_frame(bgra: np.ndarray, slot: str, jpeg_quality: int | None = 92) -> np.ndarray:
    """BGRA camera image -> (360, 640, 3) RGB student frame for `slot`."""
    import cv2
    img = bgra[:, :, :3]
    if slot == TELE_SLOT:
        x0, y0, x1, y1 = tele_crop_box(w=img.shape[1], h=img.shape[0])
        img = img[y0:y1, x0:x1]
    h, w = STUDENT_HW
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


class FrameBuffer:
    """Rolling buffer of student frames; `window()` returns 4 frames per slot."""

    def __init__(self, slots, sim_hz: int = SIM_HZ, n_frames: int = N_FRAMES):
        self.slots = list(slots)
        self.stride = int(round(sim_hz * DT))
        self.n_frames = n_frames
        self.buf: deque = deque(maxlen=(n_frames - 1) * self.stride + 1)

    def push(self, frames: dict) -> None:
        self.buf.append(frames)

    def window(self) -> dict:
        out = {}
        for slot in self.slots:
            picks = []
            for i in range(self.n_frames):
                back = (self.n_frames - 1 - i) * self.stride
                picks.append(self.buf[max(len(self.buf) - 1 - back, 0)][slot])
            out[slot] = np.stack(picks)
        return out


def controller_waypoints(traj_xy: np.ndarray, elapsed_s: float = 0.0,
                         n: int = 6, step_s: float = 0.5) -> np.ndarray:
    """10 Hz plan -> 2 Hz (right, forward) waypoints for the PID controller."""
    t_plan = (np.arange(traj_xy.shape[0]) + 1) * DT
    t_plan = np.concatenate([[0.0], t_plan])
    pts = np.concatenate([np.zeros((1, 2)), np.asarray(traj_xy, dtype=np.float64)])
    t_q = np.clip(elapsed_s + step_s * (np.arange(n) + 1), 0.0, t_plan[-1])
    x = np.interp(t_q, t_plan, pts[:, 0])
    y = np.interp(t_q, t_plan, pts[:, 1])
    return np.stack([-y, x], axis=1)


class _PID:
    def __init__(self, K_P, K_I, K_D, n):
        self.K_P, self.K_I, self.K_D = K_P, K_I, K_D
        self.window = deque([0.0] * n, maxlen=n)

    def step(self, error: float) -> float:
        self.window.append(float(error))
        integral = float(np.mean(self.window))
        derivative = self.window[-1] - self.window[-2]
        return self.K_P * error + self.K_I * integral + self.K_D * derivative


class PIDController:
    """Bench2DriveZoo's PID controller."""

    def __init__(self, turn=(1.1, 0.5, 0.4, 40), speed=(5.0, 0.5, 1.0, 40),
                 max_throttle: float = 0.75, brake_speed: float = 0.05,
                 brake_ratio: float = 1.1, clip_delta: float = 0.25, aim_dist: float = 3.5):
        self.turn_controller = _PID(*turn)
        self.speed_controller = _PID(*speed)
        self.max_throttle, self.brake_speed = max_throttle, brake_speed
        self.brake_ratio, self.clip_delta, self.aim_dist = brake_ratio, clip_delta, aim_dist

    def control_pid(self, waypoints: np.ndarray, speed: float):
        """Returns (steer, throttle, brake, metadata)."""
        wp = np.asarray(waypoints, dtype=np.float64)
        desired_speed = (0.75 * np.linalg.norm(wp[0]) * 2.0
                         + 0.25 * np.linalg.norm(wp[1] - wp[0]) * 2.0)
        aim, best = wp[0], 1e5
        for i in range(len(wp) - 1):
            for cand in (wp[i], (wp[i + 1] + wp[i]) / 2.0):
                norm = np.linalg.norm(cand)
                if abs(self.aim_dist - best) > abs(self.aim_dist - norm):
                    aim, best = cand, norm
        angle = 0.0 if aim[1] <= 0.02 else math.degrees(math.pi / 2 - math.atan2(aim[1], aim[0])) / 90.0
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))

        brake = bool(desired_speed < self.brake_speed
                     or (speed / max(desired_speed, 1e-6)) > self.brake_ratio)
        delta = float(np.clip(desired_speed - speed, 0.0, self.clip_delta))
        throttle = float(np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0
        meta = dict(speed=float(speed), desired_speed=float(desired_speed), angle=float(angle),
                    aim=[float(aim[0]), float(aim[1])], delta=delta)
        return steer, throttle, float(brake), meta
