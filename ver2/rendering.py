#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rendering.py

[수정 사항]
1. Azimuth(방위각) 보정: 측면 속도 저하 해결
2. Kalman Filter: 속도 값 널뛰기(Jittering) 해결
"""

import math
import time
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import cv2
from PySide6 import QtGui, QtCore

import lane_utils
import perception_utils
import sensor_manager


# ==============================================================================
# 1. 칼만 필터 (속도 안정화용)
# ==============================================================================
class SimpleKalmanFilter:
    def __init__(self, initial_value=0.0, process_noise=0.1, measurement_noise=5.0):
        """
        1D Kalman Filter
        - process_noise (Q): 차량 속도가 실제로 얼마나 빨리 변할 수 있는가? (낮을수록 부드러움)
        - measurement_noise (R): 레이더 센서가 얼마나 부정확한가? (높을수록 기존 값 유지 성향 강함)
        """
        self.estimate = initial_value
        self.error_cov = 1.0
        self.Q = process_noise 
        self.R = measurement_noise

    def update(self, measurement):
        # 1. 예측 (Prediction)
        # 속도는 이전 상태를 유지한다고 가정 (모델 단순화)
        pred_estimate = self.estimate
        pred_error_cov = self.error_cov + self.Q

        # 2. 보정 (Correction)
        kalman_gain = pred_error_cov / (pred_error_cov + self.R)
        self.estimate = pred_estimate + kalman_gain * (measurement - pred_estimate)
        self.error_cov = (1 - kalman_gain) * pred_error_cov
        
        return self.estimate

# 차량 ID별로 필터를 유지하기 위한 전역 딕셔너리
# Key: Vehicle ID, Value: {'filter': KalmanFilter, 'last_seen': time}
VEHICLE_TRACKERS = {}


# ==============================================================================
# 2. 카메라 오버레이
# ==============================================================================
def overlay_camera(
    img: np.ndarray,
    manager: Any,
    lane_state: Dict[str, bool],
    bev_view_params: Dict[str, Any],
    overlay_radar_on: bool,
) -> np.ndarray:
    
    if img is None: return img
    if not manager or not getattr(manager, "sensor_manager", None): return img

    sm = manager.sensor_manager
    cam = sm.cam
    rad_world_history = sm.RAD_WORLD_HISTORY

    w = sensor_manager.IMG_WIDTH
    h = sensor_manager.IMG_HEIGHT

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    # (A) 레이더 포인트 오버레이
    if overlay_radar_on:
        radar_snap = list(rad_world_history)
        size = max(1, int(bev_view_params.get('point_size', 2)))
        for i, frame in enumerate(radar_snap):
            if not frame: continue
            fade = (i + 1) / max(1, len(radar_snap))
            color = (255, int(255 * fade), 0)
            for item in frame:
                if item is None or len(item) < 3: continue
                loc = item[0]
                uv = perception_utils.project_world_to_image(cam, loc, w, h)
                if uv: cv2.circle(img, uv, size, color, -1)

    # (B) YOLO 추론
    if not hasattr(manager, "yolo") or manager.yolo is None:
        sm.LAST_MEASUREMENTS = []
        return img

    img_for_yolo = img.copy()
    try:
        dets = manager.yolo.infer(img_for_yolo, conf=float(bev_view_params.get("yolo_conf", 0.35)))
    except Exception as e:
        print(f"[ERROR] YOLO inference failed: {e}")
        sm.LAST_MEASUREMENTS = []
        return img

    if not dets:
        sm.LAST_MEASUREMENTS = []
        return img

    # --- 헬퍼 함수: GT 정보 ---
    def _get_gt_info(bbox):
        if not manager.world: return (float("nan"), float("nan"), -1)
        bx1, by1, bx2, by2 = bbox
        vehicles = manager.world.get_actors().filter('vehicle.*')
        cam_loc = cam.get_transform().location
        best_dist = float("inf"); target_speed = float("nan"); target_dist = float("nan"); target_id = -1
        
        for veh in vehicles:
            if not veh.is_alive: continue
            if veh.get_location().distance(cam_loc) > 150.0: continue

            veh_loc = veh.get_location()
            uv = perception_utils.project_world_to_image(cam, veh_loc, w, h)
            if uv is None: continue
            u, v = uv
            if (bx1 <= u <= bx2) and (by1 <= v <= by2):
                dist_m = veh_loc.distance(cam_loc)
                if dist_m < best_dist:
                    best_dist = dist_m; target_dist = dist_m
                    vel = veh.get_velocity()
                    # 3D 벡터 크기 -> 속력 (항상 양수)
                    target_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    target_id = veh.id
        return (target_speed, target_dist, target_id)

    # --- 헬퍼 함수: 레이더 속도 (절대값 + 필터 + 보정) ---
    def _get_radar_speed_corrected(bbox, approx_dist_m: float):
        bx1, by1, bx2, by2 = bbox
        valid_vels = []
        
        use_depth_gate = (not math.isnan(approx_dist_m)) and (approx_dist_m > 0.5)
        depth_tol = max(3.0, 0.15 * float(approx_dist_m)) if use_depth_gate else None
        
        world_to_cam = np.array(cam.get_transform().get_inverse_matrix())

        for frame in list(rad_world_history)[-3:]:
            for item in frame:
                if item is None or len(item) < 3: continue
                loc, vel, r_depth = item[0], float(item[1]), float(item[2])
                
                # [수정 1] 정지 물체 필터링 (절대값 기준)
                if abs(vel) < 0.5: continue

                uv = perception_utils.project_world_to_image(cam, loc, w, h)
                if uv is None: continue
                u, v = uv

                if (bx1 <= u <= bx2) and (by1 <= v <= by2):
                    if use_depth_gate and abs(r_depth - float(approx_dist_m)) > float(depth_tol):
                        continue
                    
                    # [방위각 보정]
                    loc_np = np.array([loc.x, loc.y, loc.z, 1.0])
                    local_np = world_to_cam @ loc_np
                    lx, ly = local_np[0], local_np[1]
                    azimuth = math.atan2(ly, lx)
                    cos_v = max(0.2, abs(math.cos(azimuth)))
                    
                    # [수정 2] 속도 절대값 취하기 (항상 양수)
                    corrected_vel = abs(vel) / cos_v
                    valid_vels.append(corrected_vel)

        if len(valid_vels) < 1: return float("nan")
        return float(np.median(valid_vels)) * 3.6 # Median 사용

    # --- 렌더링 루프 ---
    current_time = time.time()
    frame_measurements: List[dict] = []
    
    # 오래된 트래커 정리
    to_delete = []
    for vid, data in VEHICLE_TRACKERS.items():
        if current_time - data['last_seen'] > 2.0:
            to_delete.append(vid)
    for vid in to_delete:
        del VEHICLE_TRACKERS[vid]

    for det in dets:
        if "bbox" not in det: continue
        x1, y1, x2, y2 = det["bbox"]
        
        x1 = max(0, min(w - 1, int(x1))); x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1))); y2 = max(0, min(h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1: continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if not lane_utils.is_in_selected_lanes(cx, cy, lane_state): continue

        # 1. 정보 획득
        v_real, dist_real, veh_id = _get_gt_info((x1, y1, x2, y2))
        v_meas_raw = _get_radar_speed_corrected((x1, y1, x2, y2), dist_real)
        
        # 2. 칼만 필터 적용
        v_meas_final = float("nan")
        if not math.isnan(v_meas_raw) and veh_id != -1:
            if veh_id not in VEHICLE_TRACKERS:
                VEHICLE_TRACKERS[veh_id] = {
                    'filter': SimpleKalmanFilter(initial_value=v_meas_raw, process_noise=0.05, measurement_noise=3.0),
                    'last_seen': current_time
                }
            tracker = VEHICLE_TRACKERS[veh_id]
            # [수정 3] 필터에도 절대값 전달 (이미 위에서 했지만 안전장치)
            v_meas_final = tracker['filter'].update(abs(v_meas_raw))
            tracker['last_seen'] = current_time
        elif not math.isnan(v_meas_raw):
             v_meas_final = abs(v_meas_raw)

        # 3. 데이터 저장 (유효한 값만)
        is_valid_real = (not math.isnan(v_real)) and (v_real > 0.1)
        is_valid_meas = (not math.isnan(v_meas_final)) and (v_meas_final > 0.1)
        
        if is_valid_real and is_valid_meas:
            frame_measurements.append({
                "time": current_time, "id": veh_id, "dist": dist_real, 
                "v_real": v_real, "v_meas": v_meas_final
            })

        # 4. 시각화 (무조건 양수로 표시)
        str_dist = f"Dist: {dist_real:.1f}m" if not math.isnan(dist_real) else "Dist: --"
        str_meas = f"Meas: {v_meas_final:.1f}km/h" if not math.isnan(v_meas_final) else "Meas: --"
        str_real = f"Real: {v_real:.1f}km/h" if not math.isnan(v_real) else "Real: --"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str_real, (x1, y1 - 5), font, 0.5, (0, 255, 255), 2)
        cv2.putText(img, str_meas, (x1, y1 - 25), font, 0.5, (50, 255, 50), 2)
        cv2.putText(img, str_dist, (x1, y1 - 45), font, 0.5, (255, 255, 255), 2)

    sm.LAST_MEASUREMENTS = frame_measurements
    return img


# ==============================================================================
# 3. 레이더 BEV 렌더링 (기존 유지)
# ==============================================================================

def render_radar_bev(
    manager: Any,
    sensor_params: Dict[str, float],
    bev_view_params: Dict[str, Any],
    canvas_size: Tuple[int, int] = (1000, 520),
) -> Optional[QtGui.QImage]:
    
    if not (manager and manager.world and manager.sensor_manager and manager.sensor_manager.rad):
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

    # 파라미터
    current_max_range = float(sensor_params.get('range', 70.0))
    h_fov_deg = float(sensor_params.get('h_fov', 120.0))
    v_fov_deg = float(sensor_params.get('v_fov', 30.0))
    vel_range = float(sensor_params.get('vel_range', 40.0))

    h_fov_rad = math.radians(h_fov_deg / 2.0)
    v_fov_rad = math.radians(v_fov_deg / 2.0)
    offset_x = int(bev_view_params.get('offset_x', 0))
    offset_y = int(bev_view_params.get('offset_y', 0))
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
    except: pass

    # 레이더 포인트
    size = max(1, int(bev_view_params.get('point_size', 2)))
    half = size // 2
    
    def _vel_to_color(v_ms: float, alpha: int) -> QtGui.QColor:
        norm = float(np.clip(v_ms / max(1e-3, vel_range), -1.0, 1.0))
        intensity = max(0.2, abs(norm))
        if norm > 0.05: r, g, b = 255, int(100*(1-intensity)), int(100*(1-intensity))
        elif norm < -0.05: r, g, b = int(100*(1-intensity)), int(100*(1-intensity)), 255
        else: r, g, b = 150, 150, 150
        col = QtGui.QColor(r,g,b); col.setAlpha(alpha)
        return col

    valid_radar_points = []
    hist = list(rad_history)
    for i, pts in enumerate(hist):
        alpha = int(255 * (i + 1) / max(1, len(hist)))
        for item in pts:
            if len(item) < 4: continue
            fx, fy, depth, vel = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            if len(item) >= 5:
                fz, depth, vel = float(item[2]), float(item[3]), float(item[4])
                if abs(math.atan2(fz, max(depth, 1e-3))) > v_fov_rad: continue
            if depth > current_max_range or depth <= 0.05: continue
            if abs(math.atan2(fy, max(fx, 1e-3))) > h_fov_rad: continue
            
            col = _vel_to_color(vel, alpha)
            u = int(origin_u + (fy / meters_per_px_lat))
            v = int(origin_v - (fx / meters_per_px_fwd))
            
            if margin <= u < W - margin and margin <= v < H - margin:
                if size <= 1: p.setPen(QtGui.QPen(col)); p.drawPoint(u, v)
                else: p.fillRect(u - half, v - half, size, size, col)
                valid_radar_points.append((fx, fy))

    if len(valid_radar_points) > 0: radar_pts_np = np.array(valid_radar_points, dtype=np.float32)
    else: radar_pts_np = None

    # GT 박스 (레이더 근처만)
    sensor_tf = rad.get_transform()
    sensor_yaw_deg = sensor_tf.rotation.yaw
    world_to_sensor = np.array(sensor_tf.get_inverse_matrix(), dtype=np.float64)
    vehs = manager.world.get_actors().filter('vehicle.*')
    
    for vact in vehs:
        if not vact.is_alive: continue
        try:
            t = vact.get_transform().location
            p_world = np.array([t.x, t.y, t.z, 1.0], dtype=np.float64)
            ps = world_to_sensor @ p_world
            fx_val, fy_val = float(ps[0]), float(ps[1])

            if fx_val < 0.0 or fx_val > current_max_range: continue
            
            if radar_pts_np is None: continue
            dists_sq = (radar_pts_np[:, 0] - fx_val)**2 + (radar_pts_np[:, 1] - fy_val)**2
            if np.min(dists_sq) > 9.0: continue

            u = int(origin_u + (fy_val / meters_per_px_lat))
            v = int(origin_v - (fx_val / meters_per_px_fwd))
            ext = vact.bounding_box.extent
            length_px = max(1, int((ext.x * 2.0) / meters_per_px_fwd))
            width_px  = max(1, int((ext.y * 2.0) / meters_per_px_lat))
            
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
        except: continue

    p.end()
    return qimg