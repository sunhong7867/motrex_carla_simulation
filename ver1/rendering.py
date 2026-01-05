#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rendering.py

GUI(MainWindow)에서는 최대한 계산 로직을 빼고,
카메라 오버레이 / 레이더 BEV 이미지를 만들어 주는 순수 렌더링 모듈.
"""

import math
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from PySide6 import QtGui, QtCore

import lane_utils
import perception_utils
import sensor_manager


# ==============================================================================
# 카메라 오버레이 (레이더 포인트 + 차선 폴리곤 + BBox + 속도)
# ==============================================================================

def overlay_camera(
    img: np.ndarray,
    manager: Any,
    lane_state: Dict[str, bool],
    bev_view_params: Dict[str, Any],
    overlay_radar_on: bool,
) -> np.ndarray:
    """
    카메라 이미지에 차선, 레이더, BBox를 오버레이한다.

    Parameters
    ----------
    img : np.ndarray
        BGR 카메라 이미지 (복사본을 넘기는 걸 추천).
    manager : CarlaManager
        main.py 에서 사용하는 CarlaManager 인스턴스.
    lane_state : dict
        { 'lane_on', 'in1', 'in2', 'out1', 'out2' } 형태의 레인 활성화 상태.
    bev_view_params : dict
        { 'offset_x', 'offset_y', 'point_size' } 등 BEV 관련 설정.
        여기서는 point_size만 사용 (레이더 포인트 크기).
    overlay_radar_on : bool
        카메라에 레이더 포인트 오버레이 on/off.
    """
    if not manager or not manager.sensor_manager:
        return img

    sm = manager.sensor_manager
    cam = sm.cam
    depth_img = sm.DEPTH_IMG
    rad_world_history = sm.RAD_WORLD_HISTORY

    w = sensor_manager.IMG_WIDTH
    h = sensor_manager.IMG_HEIGHT

    # (A) 레이더 포인트 오버레이
    if overlay_radar_on:
        radar_snap = list(rad_world_history)
        size = max(1, bev_view_params.get('point_size', 2))
        for i, frame in enumerate(radar_snap):
            if not frame:
                continue
            fade = (i + 1) / max(1, len(radar_snap))
            # BGR: 파란색 → 초록/노랑 쪽으로 페이드
            color = (255, int(255 * fade), 0)
            for (loc, vel, depth) in frame:
                uv = perception_utils.project_world_to_image(cam, loc, w, h)
                if uv:
                    cv2.circle(img, uv, size, color, -1)

    # (B) 차선 영역 오버레이
    if lane_state.get('lane_on', False):
        overlay = img.copy()
        color_green = (60, 255, 60)
        color_purple = (200, 60, 200)  # BGR
        polys = lane_utils.get_lane_polys()

        if lane_state.get('in1', False) and polys['IN1'] is not None:
            cv2.fillPoly(overlay, [polys['IN1']], color_green)
        if lane_state.get('in2', False) and polys['IN2'] is not None:
            cv2.fillPoly(overlay, [polys['IN2']], color_purple)
        if lane_state.get('out1', False) and polys['OUT1'] is not None:
            cv2.fillPoly(overlay, [polys['OUT1']], color_green)
        if lane_state.get('out2', False) and polys['OUT2'] is not None:
            cv2.fillPoly(overlay, [polys['OUT2']], color_purple)

        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0.0)

    # (C) BBox 오버레이 (Occlusion, Lane Filter 포함)
    if not manager.world:
        return img

    vehicles = manager.world.get_actors().filter('vehicle.*')

    for v in vehicles:
        if not v.is_alive:
            continue

        bbox = perception_utils.get_2d_bbox(cam, v, w, h)
        if bbox is None:
            continue

        x1, y1, x2, y2, z_center = bbox

        # Occlusion 필터
        if perception_utils.is_occluded(depth_img, (x1, y1, x2, y2), z_center, w, h):
            continue

        # Lane 필터
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if not lane_utils.is_in_selected_lanes(cx, cy, lane_state):
            continue

        # 2륜차/4륜차 색상 구분
        type_id_str = str(v.type_id).lower()
        is_2_wheel = (
            'bicycle' in type_id_str
            or 'motorcycle' in type_id_str
            or 'harley' in type_id_str
        )
        color = (255, 0, 0) if is_2_wheel else (0, 0, 255)  # BGR (Blue / Red)

        # 속도 계산
        v_carla_kph = v.get_velocity().length() * 3.6
        v_calc_kph = perception_utils.estimate_speed_from_radar(
            cam, rad_world_history, (x1, y1, x2, y2), z_center, w, h
        )

        # 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_id = f"ID: {v.id}"
        label_v_calc = f"v_calc: {(0.0 if np.isnan(v_calc_kph) else v_calc_kph):.1f}"
        label_v_carla = f"v_carla: {v_carla_kph:.1f}"

        y_pos = y1 - 10
        line_h = 20
        cv2.putText(
            img,
            label_v_carla,
            (x1 + 5, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.putText(
            img,
            label_v_calc,
            (x1 + 5, y_pos - line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.putText(
            img,
            label_id,
            (x1 + 5, y_pos - line_h * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return img


# ==============================================================================
# 레이더 BEV 렌더링 (QImage)
# ==============================================================================

def render_radar_bev(
    manager: Any,
    sensor_params: Dict[str, float],
    bev_view_params: Dict[str, Any],
    canvas_size: Tuple[int, int] = (1000, 520),
) -> Optional[QtGui.QImage]:
    """
    레이더 데이터를 BEV (Bird's Eye View) QImage로 렌더링.

    Parameters
    ----------
    manager : CarlaManager
        main.py 의 CarlaManager 인스턴스.
    sensor_params : dict
        { 'range', 'h_fov', 'v_fov', 'pps', ... } 형태의 레이더 설정.
    bev_view_params : dict
        { 'offset_x', 'offset_y', 'point_size' } 형태의 BEV 뷰 설정.
    canvas_size : (W, H)
        출력 QImage 크기.

    Returns
    -------
    QtGui.QImage or None
        성공 시 BEV QImage, 실패 시 None.
    """
    if not (
        manager
        and manager.world
        and manager.sensor_manager
        and manager.sensor_manager.rad
    ):
        return None

    rad = manager.sensor_manager.rad
    rad_history = manager.sensor_manager.RAD_HISTORY

    W, H = canvas_size
    qimg = QtGui.QImage(W, H, QtGui.QImage.Format_RGB888)
    qimg.fill(QtGui.QColor(255, 255, 255))

    p = QtGui.QPainter(qimg)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 3))
    p.drawRect(10, 10, W - 20, H - 20)

    innerW, innerH = W - 20, H - 20

    offset_x = bev_view_params.get('offset_x', 100)
    offset_y = bev_view_params.get('offset_y', 0)

    # 원점 위치(레이더 센서 위치) – 기존 GUI와 동일한 좌표계 사용
    origin_u = 390 + offset_x
    origin_v = 250 + innerH // 2 + offset_y

    current_max_range = float(sensor_params.get('range', 70.0))

    # m/px 스케일
    meters_per_px_x = current_max_range / (innerW - 60)
    meters_per_px_y = (current_max_range / 2.0) / (innerH / 2.0 - 20)

    # 1. FoV 그리기
    try:
        p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220), 2, QtCore.Qt.DashLine))
        h_fov_deg = float(sensor_params.get('h_fov', 40.0))
        theta1 = math.radians(-h_fov_deg / 2.0)
        theta2 = math.radians(h_fov_deg / 2.0)
        r_line = current_max_range * 1.5

        p1_u = int(origin_u + (r_line * math.sin(theta1)) / meters_per_px_x)
        p1_v = int(origin_v - (r_line * math.cos(theta1)) / meters_per_px_y)
        p2_u = int(origin_u + (r_line * math.sin(theta2)) / meters_per_px_x)
        p2_v = int(origin_v - (r_line * math.cos(theta2)) / meters_per_px_y)

        p.drawLine(origin_u, origin_v, p1_u, p1_v)
        p.drawLine(origin_u, origin_v, p2_u, p2_v)
    except Exception:
        # FoV 그리다가 실패해도 전체 렌더를 망치지 않도록 함
        pass

    # 2. 레이더 포인트
    size = int(bev_view_params.get('point_size', 2))
    size = max(1, size)
    half = size // 2
    MAX_VEL_FOR_COLOR = 15.0

    for i, pts in enumerate(list(rad_history)):
        alpha = int(255 * (i + 1) / max(1, len(rad_history)))
        for (fx, fy, depth, vel) in pts:
            if depth > current_max_range:
                continue

            norm_vel = max(-1.0, min(1.0, vel / MAX_VEL_FOR_COLOR))
            intensity = max(0.2, abs(norm_vel))

            r, g, b = 128, 128, 128
            if norm_vel > 0.05:  # approaching?
                r = 255
                g = int(128 * (1 - intensity))
                b = int(128 * (1 - intensity))
            elif norm_vel < -0.05:  # receding?
                r = int(128 * (1 - intensity))
                g = int(128 * (1 - intensity))
                b = 255

            col = QtGui.QColor(r, g, b)
            col.setAlpha(alpha)

            u = int(origin_u + fy / meters_per_px_x)
            v = int(origin_v - fx / meters_per_px_y)
            if 10 <= u < W - 10 and 10 <= v < H - 10:
                if size <= 1:
                    p.setPen(QtGui.QPen(col))
                    p.drawPoint(u, v)
                else:
                    p.fillRect(u - half, v - half, size, size, col)

    # 3. 차량 박스 (Ground Truth)
    sensor_tf = rad.get_transform()
    sensor_yaw_deg = sensor_tf.rotation.yaw
    world_to_sensor = np.array(sensor_tf.get_inverse_matrix(), dtype=np.float64)

    vehs = manager.world.get_actors().filter('vehicle.*')
    for vact in vehs:
        try:
            if not vact.is_alive:
                continue

            t = vact.get_transform().location
            p_world = np.array([t.x, t.y, t.z, 1.0], dtype=np.float64)
            ps = world_to_sensor @ p_world

            fx_val = float(ps[0])
            fy_val = float(ps[1])

            if fx_val < -5.0 or fx_val > current_max_range:
                continue

            u = int(origin_u + fy_val / meters_per_px_x)
            v = int(origin_v - fx_val / meters_per_px_y)

            ext = vact.bounding_box.extent
            length_px = max(1, int((ext.x * 2.0) / meters_per_px_y))
            width_px = max(1, int((ext.y * 2.0) / meters_per_px_x))
            relative_yaw_deg = vact.get_transform().rotation.yaw - sensor_yaw_deg

            # 차량 박스 (보라색)
            p.save()
            p.translate(u, v)
            p.rotate(-relative_yaw_deg)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 0, 255), 2))
            p.drawRect(-width_px // 2, -length_px // 2, width_px, length_px)
            p.restore()

            # 속도 벡터
            vel = vact.get_velocity()
            v_world = np.array([vel.x, vel.y, vel.z, 0.0], dtype=np.float64)
            v_sensor = world_to_sensor @ v_world

            vx_s = float(v_sensor[0])
            vy_s = float(v_sensor[1])

            u_end = int(origin_u + (fy_val + vy_s * 1.2) / meters_per_px_x)
            v_end = int(origin_v - (fx_val + vx_s * 1.2) / meters_per_px_y)

            p.setPen(QtGui.QPen(QtGui.QColor(0, 150, 0), 2))
            p.drawLine(u, v, u_end, v_end)
        except Exception:
            # 개별 차량 처리에서 에러가 나도 전체 렌더링은 유지
            continue

    p.end()
    return qimg
