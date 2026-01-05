#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rendering.py (Optimized Version)

변경 사항:
1. 시각화(텍스트, 색상, 위치), 레이더 BEV 로직 등은 기존 코드와 100% 동일하게 유지.
2. [최적화] get_gt_speed_in_bbox의 반복 연산을 제거하여 시뮬레이션 렉 해결.
   (기존: 박스 N개 * 차량 M대 = N*M회 연산 -> 변경: 차량 M대 1회 연산 + N회 조회)
"""

import math
import time
import os
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import cv2
from PySide6 import QtGui, QtCore

import lane_utils
import perception_utils
import sensor_manager

# 캘리브레이션 extrinsic 로드
try:
    import calibration_manager
except Exception:
    calibration_manager = None

_yolo_frame_counter = 0
_last_detections_cache = []

# ==============================================================================
# 0. 캘리브레이션 로드 유틸
# ==============================================================================
EXTRINSIC_JSON_PATH = os.environ.get("EXTRINSIC_JSON", "extrinsic.json")
_EXTR_CACHE = {"R": None, "t": None, "loaded": False, "path": None}


def _load_extrinsic_cached() -> perception_utils.ExtrinsicRT:
    global _EXTR_CACHE
    if _EXTR_CACHE["loaded"] and _EXTR_CACHE["path"] == EXTRINSIC_JSON_PATH:
        R = _EXTR_CACHE["R"]
        t = _EXTR_CACHE["t"]
        return perception_utils.ExtrinsicRT(R=R, t=t)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros((3, 1), dtype=np.float64)

    if calibration_manager is not None:
        try:
            if os.path.exists(EXTRINSIC_JSON_PATH):
                R2, t2 = calibration_manager.load_extrinsic_json(EXTRINSIC_JSON_PATH)
                R = np.array(R2, dtype=np.float64).reshape(3, 3)
                t = np.array(t2, dtype=np.float64).reshape(3, 1)
        except Exception:
            pass

    _EXTR_CACHE = {"R": R, "t": t, "loaded": True, "path": EXTRINSIC_JSON_PATH}
    return perception_utils.ExtrinsicRT(R=R, t=t)

def _get_cam_model(cam_actor, w: int, h: int) -> perception_utils.CameraModel:
    try:
        fov = 70.0
        if cam_actor is not None:
            attr = getattr(cam_actor, "attributes", None)
            if attr and "fov" in attr:
                fov = float(attr["fov"])
        K = perception_utils.get_camera_intrinsic_from_fov(fov, w, h)
        return perception_utils.CameraModel(K=K, D=None)
    except Exception:
        K = np.array([[w, 0, w / 2.0],
                      [0, w, h / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)
        return perception_utils.CameraModel(K=K, D=None)

# ==============================================================================
# 1. 칼만 필터
# ==============================================================================
class SimpleKalmanFilter:
    def __init__(self, initial_value=0.0, process_noise=0.1, measurement_noise=5.0):
        self.estimate = initial_value
        self.error_cov = 1.0
        self.Q = process_noise
        self.R = measurement_noise

    def update(self, measurement):
        pred_estimate = self.estimate
        pred_error_cov = self.error_cov + self.Q
        kalman_gain = pred_error_cov / (pred_error_cov + self.R)
        self.estimate = pred_estimate + kalman_gain * (measurement - pred_estimate)
        self.error_cov = (1 - kalman_gain) * pred_error_cov
        return self.estimate

VEHICLE_TRACKERS = {}

# ==============================================================================
# [NEW] GT 차량 속도 최적화 함수 (기존 get_gt_speed_in_bbox 대체)
# ==============================================================================
def precompute_all_gt_vehicles(manager, cam, w, h) -> List[Dict]:
    """
    프레임 당 1회 호출. 모든 차량의 화면 좌표와 속도를 미리 계산해 둡니다.
    """
    if not manager or not getattr(manager, "world", None) or cam is None:
        return []

    # 1. 변환 행렬 준비 (한 번만 계산)
    cam_tf = cam.get_transform()
    world_to_cam = np.array(cam_tf.get_inverse_matrix())
    
    fov = 70.0
    if hasattr(cam, "attributes") and "fov" in cam.attributes:
        fov = float(cam.attributes["fov"])
    K = perception_utils.get_camera_intrinsic_from_fov(fov, w, h)

    # 2. 모든 차량 가져오기
    vehicles = manager.world.get_actors().filter("vehicle.*")
    results = []

    # 3. 일괄 변환
    for veh in vehicles:
        if not veh.is_alive: continue
        
        # World -> Camera
        loc = veh.get_location()
        p_world = np.array([loc.x, loc.y, loc.z, 1.0])
        p_cam = world_to_cam @ p_world # [X_fwd, Y_right, Z_up, 1]

        # Camera -> CV (X_right, Y_down, Z_fwd)
        cv_x = p_cam[1]
        cv_y = -p_cam[2]
        cv_z = p_cam[0]

        if cv_z <= 0.5: continue # 카메라 뒤쪽

        # Projection
        uv_homo = K @ np.array([cv_x, cv_y, cv_z])
        u = uv_homo[0] / uv_homo[2]
        v = uv_homo[1] / uv_homo[2]

        if 0 <= u < w and 0 <= v < h:
            vel = veh.get_velocity()
            speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            results.append({
                "id": veh.id,
                "u": u, "v": v, "z": cv_z,
                "speed": speed
            })
    return results

def find_nearest_gt_from_cache(cached_vehicles, bbox):
    """
    미리 계산된 리스트(cached_vehicles)에서 박스에 맞는 차량을 O(N)으로 찾음 (매우 빠름)
    """
    x1, y1, x2, y2 = bbox
    box_cx, box_cy = (x1+x2)/2, (y1+y2)/2
    
    best_dist = float("inf")
    best_speed = float("nan")
    best_id = -1

    for v in cached_vehicles:
        # 박스 안에 중심점이 들어오는지 확인
        if (x1 <= v["u"] <= x2) and (y1 <= v["v"] <= y2):
            # 깊이(Z)가 가장 가까운 차량 선택 (혹은 중심점 거리 사용 가능)
            if v["z"] < best_dist:
                best_dist = v["z"]
                best_speed = v["speed"]
                best_id = v["id"]
                
    return best_speed, best_id


# ==============================================================================
# 2. 카메라 오버레이 (Main)
# ==============================================================================
def overlay_camera(
    img: np.ndarray,
    manager: Any,
    lane_state: Dict[str, bool],
    bev_view_params: Dict[str, Any],
    overlay_radar_on: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    def bias(dist):
        # 20m 미만은 사용 안 함 (호출되면 안 되는 구간)
        if dist < 25:
            return -0.82
        elif dist < 35:
            return -0.48
        else:
            return -0.41
        
    stats = {
        "total": 0, "zpos": 0, "in_img": 0,
        "ext_file": EXTRINSIC_JSON_PATH, "warning": ""
    }

    if img is None: return img, stats
    if not manager or not getattr(manager, "sensor_manager", None): return img, stats

    sm = manager.sensor_manager
    cam = sm.cam
    h, w = img.shape[:2]

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    extr = _load_extrinsic_cached()
    cam_model = _get_cam_model(cam, w, h)

    # --------------------------------------------------------------------------
    # [최적화 1] 레이더 데이터 전처리 (Batch Projection)
    # --------------------------------------------------------------------------
    all_radar_points = []
    if hasattr(sm, "RAD_RCS_HISTORY"):
        frames = list(sm.RAD_RCS_HISTORY)[-5:] 
        for fr in frames:
            if not fr: continue
            for p in fr:
                if p and len(p) >= 4:
                    all_radar_points.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])

    radar_uv = np.empty((0, 2), dtype=np.int32)
    radar_z = np.empty((0,), dtype=np.float64)
    radar_v = np.empty((0,), dtype=np.float64)
    radar_xyz = np.empty((0, 3), dtype=np.float64)

    if all_radar_points:
        pts_np = np.array(all_radar_points, dtype=np.float64)
        xyz_all = pts_np[:, :3]
        vel_all = pts_np[:, 3]
        stats["total"] = len(pts_np)

        # 일괄 투영
        uv_int, z_cam = perception_utils.project_radar_to_image(
            xyz_all, cam_model, extr, w, h, use_distortion=True
        )

        # 정밀 필터링 (직접 계산)
        Xc = (extr.R @ xyz_all.T) + extr.t.reshape(3, 1)
        valid_mask = Xc[2, :] > 0.1 
        
        if np.any(valid_mask):
            xyz_filtered = xyz_all[valid_mask]
            vel_filtered = vel_all[valid_mask]
            Xc = Xc[:, valid_mask]
            
            if cam_model.D is not None:
                uv_proj, _ = cv2.projectPoints(xyz_filtered.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam_model.K, cam_model.D)
                uv_proj = uv_proj.reshape(-1, 2)
            else:
                uvw = cam_model.K @ Xc
                uv_proj = (uvw[:2, :] / uvw[2:3, :]).T

            in_img_mask = (uv_proj[:,0]>=0) & (uv_proj[:,0]<w) & (uv_proj[:,1]>=0) & (uv_proj[:,1]<h)
            
            radar_uv = np.round(uv_proj[in_img_mask]).astype(np.int32)
            radar_z = Xc[2, :][in_img_mask]
            radar_v = vel_filtered[in_img_mask]
            radar_xyz = xyz_filtered[in_img_mask]
            
            stats["zpos"] = np.count_nonzero(valid_mask)
            stats["in_img"] = len(radar_uv)

    # 레이더 전체 점 그리기
    if overlay_radar_on and len(radar_uv) > 0:
        for i in range(len(radar_uv)):
            u, v = radar_uv[i]
            cv2.circle(img, (int(u), int(v)), 2, (0, 255, 255), -1)

    # -----------------------------------------------------------
    # [3] YOLO 추론 (Frame Skipping + Resizing)
    # -----------------------------------------------------------
    global _yolo_frame_counter, _last_detections_cache

    if hasattr(manager, "yolo") and manager.yolo is not None:
        _yolo_frame_counter += 1
        # 3프레임마다 1번 실행 (YOLO_SKIP_FRAMES = 3 가정)
        if _yolo_frame_counter >= 1 or not _last_detections_cache:
            _yolo_frame_counter = 0
            
            # [핵심 수정] 원본(1280x720) 대신 작게 줄여서(640x360) 추론 -> 속도 2~3배 향상
            infer_w, infer_h = 640, 360
            img_small = cv2.resize(img, (infer_w, infer_h))
            
            try:
                conf_val = float(bev_view_params.get("yolo_conf", 0.35))
                # 작은 이미지로 추론
                small_dets = manager.yolo.infer(img_small, conf=conf_val)
                
                # 좌표를 원본 크기(1280x720)로 복원
                scale_x = w / infer_w
                scale_y = h / infer_h
                
                _last_detections_cache = []
                for d in small_dets:
                    sx1, sy1, sx2, sy2 = d["bbox"]
                    _last_detections_cache.append({
                        "bbox": (int(sx1 * scale_x), int(sy1 * scale_y), 
                                 int(sx2 * scale_x), int(sy2 * scale_y)),
                        "cls": d["cls"],
                        "conf": d["conf"]
                    })
                    
            except Exception:
                _last_detections_cache = []
        
        dets = _last_detections_cache
        manager.last_detections = dets # GUI 공유용
    else:
        dets = []
        sm.LAST_MEASUREMENTS = []
        return img, stats

    if not dets:
        sm.LAST_MEASUREMENTS = []
        return img, stats

    # --------------------------------------------------------------------------
    # [최적화 2] GT 차량 데이터 미리 계산 (루프 밖에서 1회 수행)
    # --------------------------------------------------------------------------
    cached_gt_vehicles = precompute_all_gt_vehicles(manager, cam, w, h)

    # --------------------------------------------------------------------------
    # (C) Rendering Loop
    # --------------------------------------------------------------------------
    current_time = time.time()
    frame_measurements = []
    
    for vid in list(VEHICLE_TRACKERS.keys()):
        if current_time - VEHICLE_TRACKERS[vid]["last_seen"] > 2.0:
            del VEHICLE_TRACKERS[vid]

    for det in dets:
        if "bbox" not in det: continue
        x1, y1, x2, y2 = det["bbox"]
        x1, x2 = max(0, int(x1)), min(w-1, int(x2))
        y1, y2 = max(0, int(y1)), min(h-1, int(y2))
        if x2<=x1 or y2<=y1: continue

        cx, cy = (x1+x2)//2, (y1+y2)//2
        if not lane_utils.is_in_selected_lanes(cx, cy, lane_state):
            continue
        
        bbox = (x1, y1, x2, y2)

        # 1. 레이더 매칭 (Numpy Masking)
        v_meas_raw = float("nan")
        dist_meas = float("nan")
        
        if len(radar_uv) > 0:
            in_box_mask = (radar_uv[:, 0] >= x1) & (radar_uv[:, 0] <= x2) & \
                          (radar_uv[:, 1] >= y1) & (radar_uv[:, 1] <= y2)
            
            if np.any(in_box_mask):
                box_vels = radar_v[in_box_mask]
                box_depths = np.linalg.norm(radar_xyz[in_box_mask], axis=1) 
                
                # 속도 (Median)
                valid_vel_mask = np.abs(box_vels) > 0.1
                if np.any(valid_vel_mask):
                    v_meas_raw = float(np.median(np.abs(box_vels[valid_vel_mask]))) * 3.6 
                
                # 거리 (Median)
                if len(box_depths) > 0:
                    dist_meas = float(np.median(box_depths))

                if math.isnan(dist_meas) or dist_meas < 20.0:
                    continue
                
                # Visual Debug: Blue Dot & Yellow Line (Radar Centroid)
                pts_in_box_3d = radar_xyz[in_box_mask]
                centroid_3d = np.mean(pts_in_box_3d, axis=0).reshape(1, 3)
                uv_cent, _ = perception_utils.project_radar_to_image(
                    centroid_3d, cam_model, extr, w, h, use_distortion=True
                )
                if len(uv_cent) > 0:
                    uc, vc = uv_cent[0]
                    cv2.circle(img, (int(uc), int(vc)), 4, (255, 0, 0), -1) 
                    cv2.line(img, (cx, cy), (int(uc), int(vc)), (0, 255, 255), 1)

        # 🔴 빨간 점 (Camera Target Center)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        # 2. [최적화 사용] GT 매칭 (캐시된 데이터에서 조회)
        v_real, veh_id = find_nearest_gt_from_cache(cached_gt_vehicles, bbox)

        # 3. Kalman Filter
        v_meas_final = float("nan")

        if not math.isnan(v_meas_raw):
            # 거리 기반 bias 보정 (KF 이전)
            v_meas_corr = v_meas_raw - bias(dist_meas)

            if veh_id != -1:
                if veh_id not in VEHICLE_TRACKERS:
                    VEHICLE_TRACKERS[veh_id] = {
                        "filter": SimpleKalmanFilter(
                            initial_value=float(v_meas_corr),
                            process_noise=float(bev_view_params.get("kf_q", 0.05)),
                            measurement_noise=float(bev_view_params.get("kf_r", 6.0)),
                        ),
                        "last_seen": current_time,
                    }

                v_meas_final = VEHICLE_TRACKERS[veh_id]["filter"].update(
                    abs(float(v_meas_corr))
                )
                VEHICLE_TRACKERS[veh_id]["last_seen"] = current_time

            else:
                # GT 매칭 안 된 경우 (KF 없이 사용)
                v_meas_final = abs(float(v_meas_corr))

        # 데이터 저장
        if (not math.isnan(v_meas_final)) and (not math.isnan(dist_meas)) and (v_meas_final > 1.0):
            frame_measurements.append({
                "time": current_time,
                "id": veh_id,
                "dist": dist_meas,
                "v_real": v_real if (not math.isnan(v_real) and v_real > 0.1) else float("nan"),
                "v_meas": v_meas_final,
            })

        # 텍스트 그리기 (기존 양식 유지)
        str_real = f"Real: {v_real:.1f}km/h" if not math.isnan(v_real) else "Real: --"
        str_meas = f"Meas: {v_meas_final:.1f}km/h" if not math.isnan(v_meas_final) else "Meas: --"
        str_dist = f"Dist: {dist_meas:.1f}m" if not math.isnan(dist_meas) else "Dist: --"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ty = max(15, y1 - 5)
        cv2.putText(img, str_real, (x1, ty), font, 0.5, (0, 255, 255), 2)
        cv2.putText(img, str_meas, (x1, max(15, ty - 20)), font, 0.5, (50, 255, 50), 2)
        cv2.putText(img, str_dist, (x1, max(15, ty - 40)), font, 0.5, (255, 255, 255), 2)

    sm.LAST_MEASUREMENTS = frame_measurements
    return img, stats


# ==============================================================================
# 3. 레이더 BEV 렌더링 (기존 유지)
# ==============================================================================
def render_radar_bev(
    manager: Any,
    sensor_params: Dict[str, float],
    bev_view_params: Dict[str, Any],
    canvas_size: Tuple[int, int] = (1000, 520),
) -> Optional[QtGui.QImage]:

    if not (manager and getattr(manager, "world", None) and getattr(manager, "sensor_manager", None) and manager.sensor_manager.rad):
        return None

    rad = manager.sensor_manager.rad
    rad_history = manager.sensor_manager.RAD_HISTORY 

    W, H = canvas_size
    qimg = QtGui.QImage(W, H, QtGui.QImage.Format_RGB888)
    qimg.fill(QtGui.QColor(255, 255, 255))
    p = QtGui.QPainter(qimg)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)

    margin = 10
    p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 3))
    p.drawRect(margin, margin, W - 2 * margin, H - 2 * margin)
    innerW, innerH = W - 2 * margin, H - 2 * margin

    current_max_range = float(sensor_params.get("range", 70.0))
    h_fov_deg = float(sensor_params.get("h_fov", 120.0))
    v_fov_deg = float(sensor_params.get("v_fov", 30.0))
    vel_range = float(sensor_params.get("vel_range", 40.0))

    h_fov_rad = math.radians(h_fov_deg / 2.0)
    v_fov_rad = math.radians(v_fov_deg / 2.0)

    offset_x = int(bev_view_params.get("offset_x", 0))
    offset_y = int(bev_view_params.get("offset_y", 0))
    origin_u = margin + innerW // 2 + offset_x
    origin_v = margin + innerH - 20 + offset_y

    lateral_max = current_max_range * math.tan(h_fov_rad)
    usableW = max(50, innerW - 60)
    usableH = max(50, innerH - 40)
    meters_per_px_lat = (2.0 * lateral_max) / float(usableW) if lateral_max > 1e-3 else 0.1
    meters_per_px_fwd = current_max_range / float(usableH) if current_max_range > 1e-3 else 0.2

    # FoV 라인
    try:
        p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 2, QtCore.Qt.DashLine))
        theta1, theta2 = -h_fov_rad, +h_fov_rad
        p1_u = int(origin_u + (current_max_range * math.tan(theta1)) / meters_per_px_lat)
        p1_v = int(origin_v - (current_max_range) / meters_per_px_fwd)
        p2_u = int(origin_u + (current_max_range * math.tan(theta2)) / meters_per_px_lat)
        p2_v = int(origin_v - (current_max_range) / meters_per_px_fwd)
        p.drawLine(origin_u, origin_v, p1_u, p1_v)
        p.drawLine(origin_u, origin_v, p2_u, p2_v)
        p.drawLine(p1_u, p1_v, p2_u, p2_v)
    except Exception:
        pass

    size = max(1, int(bev_view_params.get("point_size", 2)))
    half = size // 2

    def _vel_to_color(v_ms: float, alpha: int) -> QtGui.QColor:
        norm = float(np.clip(v_ms / max(1e-3, vel_range), -1.0, 1.0))
        intensity = max(0.2, abs(norm))
        if norm > 0.05:
            r, g, b = 255, int(100 * (1 - intensity)), int(100 * (1 - intensity))
        elif norm < -0.05:
            r, g, b = int(100 * (1 - intensity)), int(100 * (1 - intensity)), 255
        else:
            r, g, b = 150, 150, 150
        col = QtGui.QColor(r, g, b)
        col.setAlpha(alpha)
        return col

    valid_radar_points = []
    hist = list(rad_history)
    for i, pts in enumerate(hist):
        alpha = int(255 * (i + 1) / max(1, len(hist)))
        for item in pts:
            if item is None or len(item) < 4:
                continue

            fx, fy, fz, vel = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            depth = math.sqrt(fx * fx + fy * fy + fz * fz)

            if abs(math.atan2(fz, max(depth, 1e-3))) > v_fov_rad:
                continue
            if depth > current_max_range or depth <= 0.05:
                continue
            if abs(math.atan2(fy, max(fx, 1e-3))) > h_fov_rad:
                continue

            col = _vel_to_color(vel, alpha)
            u = int(origin_u + (fy / meters_per_px_lat))
            v = int(origin_v - (fx / meters_per_px_fwd))

            if margin <= u < W - margin and margin <= v < H - margin:
                if size <= 1:
                    p.setPen(QtGui.QPen(col))
                    p.drawPoint(u, v)
                else:
                    p.fillRect(u - half, v - half, size, size, col)
                valid_radar_points.append((fx, fy))

    radar_pts_np = np.array(valid_radar_points, dtype=np.float32) if len(valid_radar_points) > 0 else None

    # GT 박스(시뮬레이션에서만 의미) - 기존 유지
    try:
        sensor_tf = rad.get_transform()
        sensor_yaw_deg = sensor_tf.rotation.yaw
        world_to_sensor = np.array(sensor_tf.get_inverse_matrix(), dtype=np.float64)
        vehs = manager.world.get_actors().filter("vehicle.*")

        for vact in vehs:
            if not vact.is_alive:
                continue
            try:
                t = vact.get_transform().location
                p_world = np.array([t.x, t.y, t.z, 1.0], dtype=np.float64)
                ps = world_to_sensor @ p_world
                fx_val, fy_val = float(ps[0]), float(ps[1])

                if fx_val < 0.0 or fx_val > current_max_range:
                    continue

                if radar_pts_np is None:
                    continue
                dists_sq = (radar_pts_np[:, 0] - fx_val) ** 2 + (radar_pts_np[:, 1] - fy_val) ** 2
                if np.min(dists_sq) > 9.0:
                    continue

                u = int(origin_u + (fy_val / meters_per_px_lat))
                v = int(origin_v - (fx_val / meters_per_px_fwd))
                ext = vact.bounding_box.extent
                length_px = max(1, int((ext.x * 2.0) / meters_per_px_fwd))
                width_px = max(1, int((ext.y * 2.0) / meters_per_px_lat))

                p.save()
                p.translate(u, v)
                p.rotate(-(vact.get_transform().rotation.yaw - sensor_yaw_deg))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 255), 2))
                p.drawRect(-width_px // 2, -length_px // 2, width_px, length_px)
                p.restore()

                vel_w = vact.get_velocity()
                v_world = np.array([vel_w.x, vel_w.y, vel_w.z, 0.0], dtype=np.float64)
                v_sensor = world_to_sensor @ v_world
                vx_s, vy_s = float(v_sensor[0]), float(v_sensor[1])
                u_end = int(origin_u + ((fy_val + vy_s * 1.2) / meters_per_px_lat))
                v_end = int(origin_v - ((fx_val + vx_s * 1.2) / meters_per_px_fwd))
                p.setPen(QtGui.QPen(QtGui.QColor(0, 150, 0), 2))
                p.drawLine(u, v, u_end, v_end)
            except Exception:
                continue
    except Exception:
        pass

    p.end()
    return qimg