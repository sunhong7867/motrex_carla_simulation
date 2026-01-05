#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rendering.py (Calibration-based version)

핵심 변경:
1) 레이더 오버레이: RAD_RCS_HISTORY + project_radar_to_image() 사용 (캘리브레이션 extrinsic 적용)
2) 거리 계산 추가: BBox 내부로 투영되는 레이더 점들로 dist_meas(평균 range) 계산 -> Dist 표시
3) 속도 계산 안정화:
   - 도플러 필터 기본 완화(0.5 -> 0.05)
   - estimate_speed 결과가 m/s일 가능성 대비 km/h 변환 보장
   - BBox 내부 점이 부족하면 NaN
4) 시뮬레이션 GT(Real speed)는 디버깅용으로 유지(월드 있을 때만)
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


# ==============================================================================
# 0. 캘리브레이션 로드 유틸
# ==============================================================================
EXTRINSIC_JSON_PATH = os.environ.get("EXTRINSIC_JSON", "extrinsic.json")
_EXTR_CACHE = {"R": None, "t": None, "loaded": False, "path": None}


def _load_extrinsic_cached() -> perception_utils.ExtrinsicRT:
    """
    extrinsic.json에서 Radar->Camera (R,t)를 로드.
    파일이 없거나 로드 실패 시 identity로 fallback (시뮬레이션 디버깅용).
    """
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
                print(f"[rendering] Loaded extrinsic: {EXTRINSIC_JSON_PATH}")
            else:
                print(f"[rendering] extrinsic not found: {EXTRINSIC_JSON_PATH} (fallback identity)")
        except Exception as e:
            print(f"[rendering] extrinsic load failed ({EXTRINSIC_JSON_PATH}): {e} (fallback identity)")
    else:
        print("[rendering] calibration_manager import failed (fallback identity)")

    _EXTR_CACHE = {"R": R, "t": t, "loaded": True, "path": EXTRINSIC_JSON_PATH}
    return perception_utils.ExtrinsicRT(R=R, t=t)

def _get_cam_model(cam_actor, w: int, h: int) -> perception_utils.CameraModel:
    try:
        fov = 70.0
        if cam_actor is not None:
            # CARLA camera blueprint attribute에서 FOV 읽기
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
# 1. 칼만 필터 (속도 안정화용)
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


VEHICLE_TRACKERS = {}  # {veh_id: {'filter': SimpleKalmanFilter, 'last_seen': t}}


# ==============================================================================
# 2. 카메라 오버레이 (Calibration-based)
# ==============================================================================
def get_gt_speed_in_bbox(manager, cam, bbox, w, h):
    """
    [수정됨] 좌표축 변환을 포함하여 BBox 내부의 차량 속도를 정확히 가져옵니다.
    """
    if not manager or not getattr(manager, "world", None) or cam is None:
        return float("nan"), -1

    # 1. CARLA 카메라의 월드 변환 행렬
    cam_tf = cam.get_transform()
    world_to_cam = np.array(cam_tf.get_inverse_matrix()) # 4x4 행렬

    # 2. 내부 파라미터 K (FOV 70도 가정)
    fov = 70.0
    if hasattr(cam, "attributes") and "fov" in cam.attributes:
        fov = float(cam.attributes["fov"])
    K = perception_utils.get_camera_intrinsic_from_fov(fov, w, h)

    vehicles = manager.world.get_actors().filter("vehicle.*")
    x1, y1, x2, y2 = bbox
    
    # 박스 중심점 (거리 계산 보조용)
    box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2

    best_dist = float("inf")
    best_speed = float("nan")
    best_id = -1

    for veh in vehicles:
        if not veh.is_alive:
            continue

        # 3. 차량 위치 (World) -> (Camera Actor Frame)
        veh_loc = veh.get_location()
        p_world = np.array([veh_loc.x, veh_loc.y, veh_loc.z, 1.0])
        
        # CARLA 좌표계 결과: [X(앞), Y(우), Z(상), 1]
        p_cam_raw = world_to_cam @ p_world

        # 4. [핵심 수정] CARLA 좌표계 -> 컴퓨터 비전(CV) 좌표계 변환
        # CARLA: X=Front, Y=Right, Z=Up
        # CV   : X=Right, Y=Down,  Z=Front
        
        cv_x = p_cam_raw[1]      # Y -> X
        cv_y = -p_cam_raw[2]     # Z -> -Y (위쪽이 -, 아래쪽이 +)
        cv_z = p_cam_raw[0]      # X -> Z (깊이)

        # 카메라 뒤쪽이면 패스
        if cv_z <= 0.5: 
            continue

        # 5. 투영 (Projection)
        # K @ [x, y, z]
        uv_homo = K @ np.array([cv_x, cv_y, cv_z])
        u = uv_homo[0] / uv_homo[2]
        v = uv_homo[1] / uv_homo[2]

        # 6. 박스 안에 중심점이 들어오는지 확인
        if (x1 <= u <= x2) and (y1 <= v <= y2):
            # 박스 중심과의 거리로 가장 적합한 차량 선정
            dist_sq = (u - box_cx)**2 + (v - box_cy)**2
            
            # (깊이가 가장 가까운 차를 우선하되, 박스 중심과 가까운지도 고려 가능)
            # 여기서는 단순히 깊이(Z)가 가장 가까운 차를 선택
            if cv_z < best_dist:
                best_dist = cv_z
                vel = veh.get_velocity()
                best_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                best_id = veh.id

    return best_speed, best_id

def overlay_camera(
    img: np.ndarray,
    manager: Any,
    lane_state: Dict[str, bool],
    bev_view_params: Dict[str, Any],
    overlay_radar_on: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    카메라 이미지 위에 레이더 포인트를 투영하고, YOLO 박스와 속도 정보를 그립니다.
    + [Visual Debug] 빨간 점(Target), 파란 점(Source), 노란 선(Error) 시각화 포함
    """
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
    # (A) Radar overlay (전체 점 표시) & Stats
    # --------------------------------------------------------------------------
    if hasattr(sm, "RAD_RCS_HISTORY"):
        radar_snap = list(sm.RAD_RCS_HISTORY)
        
        # 1. 통계 (최신 프레임)
        if len(radar_snap) > 0:
            xyz_latest = np.array(
                [[p[0], p[1], p[2]] for p in radar_snap[-1] if p is not None and len(p)>=3],
                dtype=np.float64
            )
            # 간단한 통계 계산 (내부 함수 없이 직접 처리)
            if xyz_latest.size > 0:
                stats["total"] = xyz_latest.shape[0]
                Xc = (extr.R @ xyz_latest.T) + extr.t.reshape(3, 1)
                stats["zpos"] = int(np.count_nonzero(Xc[2, :] > 0))
                # 투영된 점 개수는 생략하거나 필요시 추가

        # 2. 전체 점 그리기
        if overlay_radar_on:
            size = max(1, int(bev_view_params.get("point_size", 2)))
            for i, frame in enumerate(radar_snap):
                if not frame: continue
                fade = (i + 1) / max(1, len(radar_snap))
                color = (255, int(255 * fade), 0) # Cyan-ish

                xyz = np.array([[p[0],p[1],p[2]] for p in frame if p and len(p)>=3], dtype=np.float64)
                if xyz.size == 0: continue

                uv, _ = perception_utils.project_radar_to_image(
                    xyz, cam_model, extr, w, h, use_distortion=True
                )
                if uv is not None:
                    for (u, v) in uv:
                        cv2.circle(img, (int(u), int(v)), size, color, -1)

    # --------------------------------------------------------------------------
    # (B) YOLO Inference
    # --------------------------------------------------------------------------
    if not hasattr(manager, "yolo") or manager.yolo is None:
        sm.LAST_MEASUREMENTS = []
        return img, stats

    img_for_yolo = img.copy()
    try:
        dets = manager.yolo.infer(img_for_yolo, conf=float(bev_view_params.get("yolo_conf", 0.35)))
        manager.last_detections = dets
    except Exception:
        dets = []

    if not dets:
        sm.LAST_MEASUREMENTS = []
        return img, stats

    # --------------------------------------------------------------------------
    # Helper Functions (Distance & Speed) - 기존 로직 유지
    # --------------------------------------------------------------------------
    def _get_radar_distance_calibrated(bbox):
        # ... (이전 코드와 동일, 생략 없이 사용하려면 위 파일 내용 참조) ...
        # (간략화를 위해 로직이 필요하면 이전 답변의 코드를 그대로 사용하세요. 
        #  여기서는 BBox 내부 로직에 통합하여 처리할 수도 있지만, 
        #  속도/거리 계산 함수가 길어서 분리된 상태를 유지하는 것이 좋습니다.)
        if not hasattr(sm, "RAD_RCS_HISTORY"): return float("nan")
        frames = list(sm.RAD_RCS_HISTORY)[-5:] # 최근 5프레임
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        
        pts = []
        for fr in frames:
            if not fr: continue
            for p in fr:
                if p and len(p)>=3: pts.append((float(p[0]), float(p[1]), float(p[2])))
        if not pts: return float("nan")
        
        xyz = np.array(pts, dtype=np.float64)
        Xc = (extr.R @ xyz.T) + extr.t.reshape(3,1)
        z_mask = Xc[2,:] > 0.1
        if np.count_nonzero(z_mask)==0: return float("nan")
        
        Xc = Xc[:, z_mask]; xyz = xyz[z_mask]
        uvw = cam_model.K @ Xc
        uv = (uvw[:2,:] / uvw[2:3,:]).T
        
        in_bbox = (uv[:,0]>=x1) & (uv[:,0]<=x2) & (uv[:,1]>=y1) & (uv[:,1]<=y2)
        if np.count_nonzero(in_bbox) < 3: return float("nan")
        
        xyz_bb = xyz[in_bbox]
        uv_bb = uv[in_bbox]
        # 박스 중심과 가까운 점들 위주로
        dist2 = (uv_bb[:,0]-cx)**2 + (uv_bb[:,1]-cy)**2
        k = min(5, len(dist2))
        idx = np.argpartition(dist2, k-1)[:k]
        
        ranges = np.sqrt(np.sum(xyz_bb[idx]**2, axis=1))
        return float(np.median(ranges))

    def _get_radar_speed_calibrated(bbox):
        if not hasattr(sm, "RAD_RCS_HISTORY"): return float("nan")
        frames = list(sm.RAD_RCS_HISTORY)[-6:]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        
        pts = []
        for fr in frames:
            if not fr: continue
            for p in fr:
                if p and len(p)>=4:
                    v = float(p[3])
                    if abs(v) > 0.05: # min doppler
                        pts.append((float(p[0]), float(p[1]), float(p[2]), v))
        if len(pts) < 3: return float("nan")
        
        xyz = np.array([[p[0],p[1],p[2]] for p in pts], dtype=np.float64)
        dop = np.array([p[3] for p in pts], dtype=np.float64)
        
        Xc = (extr.R @ xyz.T) + extr.t.reshape(3,1)
        z_mask = Xc[2,:] > 0.1
        if np.count_nonzero(z_mask) < 3: return float("nan")
        
        Xc = Xc[:, z_mask]; dop = dop[z_mask]
        uvw = cam_model.K @ Xc
        uv = (uvw[:2,:] / uvw[2:3,:]).T
        
        in_bbox = (uv[:,0]>=x1) & (uv[:,0]<=x2) & (uv[:,1]>=y1) & (uv[:,1]<=y2)
        if np.count_nonzero(in_bbox) < 3: return float("nan")
        
        uv_bb = uv[in_bbox]; dop_bb = dop[in_bbox]
        dist2 = (uv_bb[:,0]-cx)**2 + (uv_bb[:,1]-cy)**2
        k = min(7, len(dist2))
        idx = np.argpartition(dist2, k-1)[:k]
        
        return float(np.median(abs(dop_bb[idx]))) * 3.6

    # --------------------------------------------------------------------------
    # (C) Rendering Loop (Visualization + Measurements)
    # --------------------------------------------------------------------------
    current_time = time.time()
    frame_measurements = []
    
    # 트래커 정리
    for vid in list(VEHICLE_TRACKERS.keys()):
        if current_time - VEHICLE_TRACKERS[vid]["last_seen"] > 2.0:
            del VEHICLE_TRACKERS[vid]

    for det in dets:
        if "bbox" not in det: continue
        x1, y1, x2, y2 = det["bbox"]
        
        # 화면 범위 클리핑
        x1, x2 = max(0, int(x1)), min(w-1, int(x2))
        y1, y2 = max(0, int(y1)), min(h-1, int(y2))
        if x2<=x1 or y2<=y1: continue

        cx, cy = (x1+x2)//2, (y1+y2)//2
        if not lane_utils.is_in_selected_lanes(cx, cy, lane_state):
            continue
        
        bbox = (x1, y1, x2, y2)

        # 1. 🔴 빨간 점 (Camera Target Center)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        # 2. 🔵 파란 점 (Radar Source Centroid) & 🟡 오차 선
        # 현재 박스 안에 들어오는 레이더 포인트들의 무게중심 계산 (시각화용)
        pts_for_vis = []
        vis_frames = list(sm.RAD_RCS_HISTORY)[-5:] if hasattr(sm, "RAD_RCS_HISTORY") else []
        for fr in vis_frames:
            if not fr: continue
            for p in fr:
                if p and len(p)>=3:
                    pts_for_vis.append((float(p[0]), float(p[1]), float(p[2])))
        
        if pts_for_vis:
            xyz_v = np.array(pts_for_vis, dtype=np.float64)
            # 투영
            Xc_v = (extr.R @ xyz_v.T) + extr.t.reshape(3, 1)
            z_mask = Xc_v[2, :] > 0.1
            Xc_v = Xc_v[:, z_mask]
            xyz_v = xyz_v[z_mask] # 원본 좌표도 필터링
            
            if Xc_v.shape[1] > 0:
                uvw_v = cam_model.K @ Xc_v
                uv_v = (uvw_v[:2, :] / uvw_v[2:3, :]).T
                
                # 박스 내부 필터링
                in_box = (uv_v[:,0]>=x1) & (uv_v[:,0]<=x2) & (uv_v[:,1]>=y1) & (uv_v[:,1]<=y2)
                
                if np.count_nonzero(in_box) > 0:
                    # 박스 내 포인트들의 3D 무게중심 계산
                    pts_in_box_3d = xyz_v[in_box]
                    centroid_3d = np.mean(pts_in_box_3d, axis=0).reshape(1, 3)
                    
                    # 무게중심 재투영
                    Xc_c = (extr.R @ centroid_3d.T) + extr.t.reshape(3, 1)
                    if Xc_c[2] > 0:
                        uvw_c = cam_model.K @ Xc_c
                        uc = int(uvw_c[0] / uvw_c[2])
                        vc = int(uvw_c[1] / uvw_c[2])
                        
                        # 시각화 그리기
                        cv2.circle(img, (uc, vc), 4, (255, 0, 0), -1)      # Blue Dot
                        cv2.line(img, (cx, cy), (uc, vc), (0, 255, 255), 1) # Yellow Line

        # 3. 측정 및 텍스트 표시
        v_real, veh_id = get_gt_speed_in_bbox(manager, cam, bbox, w, h)
        v_meas_raw = _get_radar_speed_calibrated(bbox)
        dist_meas = _get_radar_distance_calibrated(bbox)

        # Kalman Filter
        v_meas_final = float("nan")
        if not math.isnan(v_meas_raw) and veh_id != -1:
            if veh_id not in VEHICLE_TRACKERS:
                VEHICLE_TRACKERS[veh_id] = {
                    "filter": SimpleKalmanFilter(
                        initial_value=float(v_meas_raw),
                        process_noise=float(bev_view_params.get("kf_q", 0.05)),
                        measurement_noise=float(bev_view_params.get("kf_r", 6.0)),
                    ),
                    "last_seen": current_time,
                }
            v_meas_final = VEHICLE_TRACKERS[veh_id]["filter"].update(abs(float(v_meas_raw)))
            VEHICLE_TRACKERS[veh_id]["last_seen"] = current_time
        elif not math.isnan(v_meas_raw):
            v_meas_final = abs(float(v_meas_raw))

        # 데이터 저장용 리스트 추가
        if (not math.isnan(v_meas_final)) and (not math.isnan(dist_meas)) and (v_meas_final > 0.1):
            frame_measurements.append({
                "time": current_time,
                "id": veh_id,
                "dist": dist_meas,
                "v_real": v_real if (not math.isnan(v_real) and v_real > 0.1) else float("nan"),
                "v_meas": v_meas_final,
            })

        # 텍스트 그리기
        str_dist = f"Dist: {dist_meas:.1f}m" if not math.isnan(dist_meas) else "Dist: --"
        str_meas = f"Meas: {v_meas_final:.1f}km/h" if not math.isnan(v_meas_final) else "Meas: --"
        str_real = f"Real: {v_real:.1f}km/h" if not math.isnan(v_real) else "Real: --"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        ty = max(15, y1 - 5)
        cv2.putText(img, str_real, (x1, ty), font, 0.5, (0, 255, 255), 2)
        cv2.putText(img, str_meas, (x1, max(15, ty - 20)), font, 0.5, (50, 255, 50), 2)
        cv2.putText(img, str_dist, (x1, max(15, ty - 40)), font, 0.5, (255, 255, 255), 2)

    sm.LAST_MEASUREMENTS = frame_measurements
    return img, stats


# ==============================================================================
# 3. 레이더 BEV 렌더링 (기존 유지 + RAD_HISTORY 포맷 보정)
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
    rad_history = manager.sensor_manager.RAD_HISTORY  # (x,y,z,v) 로 들어오는 것을 가정

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