#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Deque, List

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # OpenCV 없으면 캘리브레이션/왜곡 보정 기능 제한


# -------------------------------
# 1) Camera model (K, D)
# -------------------------------
@dataclass
class CameraModel:
    K: np.ndarray                 # (3,3)
    D: Optional[np.ndarray] = None  # (k1,k2,p1,p2[,k3...]) or None


# -------------------------------
# 2) Extrinsic: Radar -> Camera
# -------------------------------
@dataclass
class ExtrinsicRT:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,1)

    def as_3x4(self) -> np.ndarray:
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return Rt


def project_radar_to_image(
    radar_xyz: np.ndarray,          # (N,3) in Radar frame (RCS)
    cam: CameraModel,
    extr: ExtrinsicRT,
    w: int,
    h: int,
    use_distortion: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radar frame 3D points -> Camera frame -> Pixel projection.
    Returns:
      uv_int: (M,2) valid pixel coords (int)
      z_cam : (M,)  corresponding depth in camera frame (z>0)
    """
    if radar_xyz.ndim != 2 or radar_xyz.shape[1] != 3:
        raise ValueError("radar_xyz must be (N,3)")

    # Radar -> Camera
    Xr = radar_xyz.astype(np.float64).T  # (3,N)
    Xc = (extr.R @ Xr) + extr.t.reshape(3, 1)  # (3,N)

    z = Xc[2, :]
    valid = z > 1e-6
    Xc = Xc[:, valid]
    z = z[valid]
    if Xc.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)

    # Camera -> Pixel
    if use_distortion and (cam.D is not None) and (cv2 is not None):
        # Use cv2.projectPoints for distortion-aware projection
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        pts = Xc.T.reshape(-1, 1, 3)  # (M,1,3)
        img_pts, _ = cv2.projectPoints(pts, rvec, tvec, cam.K, cam.D)
        uv = img_pts.reshape(-1, 2)
    else:
        # Pinhole
        uvw = cam.K @ Xc  # (3,M)
        uv = (uvw[:2, :] / uvw[2:3, :]).T  # (M,2)

    u = uv[:, 0]
    v = uv[:, 1]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = uv[in_img]
    z = z[in_img]

    uv_int = np.round(uv).astype(np.int32)
    return uv_int, z


# -------------------------------
# 3) (Optional) keep CARLA helpers
# -------------------------------
def get_camera_intrinsic_from_fov(fov_deg: float, w: int, h: int) -> np.ndarray:
    """
    CARLA처럼 fov만 있을 때 K 생성(실카메라에선 보통 캘리브레이션으로 K 획득)
    """
    f = w / (2.0 * math.tan((float(fov_deg) * math.pi / 180.0) / 2.0))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


# -------------------------------
# 4) Speed aggregation (Radar Doppler)
# -------------------------------
def estimate_speed_from_radar_points_in_bbox(
    radar_points_rsc,
    bxyxy,
    cam,
    extr,
    w,
    h,
    min_points: int = 1   # ★ 변경: 최소 1개
) -> float:

    if radar_points_rsc is None or len(radar_points_rsc) == 0:
        return float("nan")

    x1, y1, x2, y2 = bxyxy
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    xyz = np.array([[p[0], p[1], p[2]] for p in radar_points_rsc], dtype=np.float64)
    dop = np.array([p[3] for p in radar_points_rsc], dtype=np.float64)

    # === 투영 (왜곡 OFF → 인덱스 안정성 확보) ===
    uv, zcam = project_radar_to_image(
        xyz, cam, extr, w, h, use_distortion=False
    )

    if uv.shape[0] == 0:
        return float("nan")

    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    sel = dop[:uv.shape[0]][inside]

    if sel.shape[0] < min_points:
        return float("nan")

    # === Robust 집계 (median) ===
    v_mps = np.median(sel)
    return abs(v_mps) * 3.6  # kph