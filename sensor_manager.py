#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import weakref
import cv2
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

# ==============================================================================
# [설정] 전역 파라미터
# ==============================================================================
IMG_WIDTH, IMG_HEIGHT = 1280, 720

# 육교 기준 포즈(고정) - 시뮬레이션용
FIXED_X, FIXED_Y, FIXED_Z = 22.97, 136.05, 9.64
FIXED_PITCH, FIXED_YAW, FIXED_ROLL = -10.0, 0.0, 0.0

# [NEW] 레이더 센서 오프셋 설정 (카메라 위치 기준 상대 좌표)
# 단위: 미터(m)
# X: 전방(+), 후방(-)
# Y: 우측(+), 좌측(-)  <-- CARLA 좌표계 기준
# Z: 상방(+), 하방(-)
RADAR_OFFSET_X = 0   # 카메라보다 앞
RADAR_OFFSET_Y = 1  # 카메라보다 왼/오른쪽
RADAR_OFFSET_Z = -1   # 높이

CAM_TICK = 0.05 
RAD_TICK = 0.05
# ==============================================================================


class SensorManager:
    """
    실센서 적용을 고려한 최소 구성:
      - RGB Camera
      - Radar (RCS 좌표계 4D: x,y,z,doppler + time/frame)
    """

    def __init__(self, world: carla.World):
        self.world = world
        self.cam: Optional[carla.Actor] = None
        self.rad: Optional[carla.Actor] = None
        self.CAMERA_IMG: Optional[np.ndarray] = None
        self.RAD_HISTORY = deque(maxlen=15)
        self.RAD_RCS_HISTORY = deque(maxlen=100)
        self.gt_track = {"veh_id": -1, "last_seen": 0.0}

    def _tf_from(
        self,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        delta_yaw: float = 0.0
    ) -> carla.Transform:
        """육교 기준 위치에서 오프셋을 적용한 Transform"""
        return carla.Transform(
            carla.Location(FIXED_X + offset_x, FIXED_Y + offset_y, FIXED_Z + offset_z),
            carla.Rotation(FIXED_PITCH, FIXED_YAW + delta_yaw, FIXED_ROLL)
        )

    def spawn_sensors(self, sensor_params: Dict[str, Any], pos_params: Dict[str, Any]):
        """
        센서를 스폰합니다. 레이더 위치는 상단 전역 변수(RADAR_OFFSET)를 따릅니다.
        """
        self.destroy_sensors()

        new_range = sensor_params.get('range', 100.0)
        new_h_fov = sensor_params.get('h_fov', 100.0)
        new_v_fov = sensor_params.get('v_fov', 22.0)
        new_pps = sensor_params.get('pps', 100000) 

        pos_x = pos_params.get('x', 0.0)
        pos_y = pos_params.get('y', 0.0)
        pos_z = pos_params.get('z', 0.0)
        rot_yaw = pos_params.get('yaw', 0.0)

        # 1. 카메라 위치 설정
        pose_cam = self._tf_from(pos_x, pos_y, pos_z, rot_yaw)

        # 2. 레이더 위치 설정 (전역 오프셋 적용)
        # [수정] 하드코딩된 값(0.5, -0.5) 대신 변수 사용
        pose_rad = self._tf_from(
            pos_x + RADAR_OFFSET_X, 
            pos_y + RADAR_OFFSET_Y, 
            pos_z + RADAR_OFFSET_Z, 
            rot_yaw
        )

        bp = self.world.get_blueprint_library()
        weak_self = weakref.ref(self)

        try:
            # --- Camera Spawn ---
            cam_bp = bp.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            cam_bp.set_attribute('fov', '70')
            self.cam = self.world.spawn_actor(cam_bp, pose_cam)
            self.cam.listen(lambda image: SensorManager._camera_callback(weak_self, image))

            # --- Radar Spawn ---
            rad_bp = bp.find('sensor.other.radar')
            rad_bp.set_attribute('range', f'{new_range}')
            rad_bp.set_attribute('horizontal_fov', f'{new_h_fov}')
            rad_bp.set_attribute('vertical_fov', f'{new_v_fov}')
            rad_bp.set_attribute('points_per_second', f'{new_pps}')
            rad_bp.set_attribute('sensor_tick', f'{RAD_TICK}')
            
            self.rad = self.world.spawn_actor(rad_bp, pose_rad)
            self.rad.listen(lambda data: SensorManager._radar_callback(weak_self, data))

            # [로그] 실제 적용된 오프셋 출력
            print(
                f"[SensorManager] Sensors spawned (RETINA-4FN Spec).\n"
                f"  > Cam: (X:{pos_x:.2f}, Y:{pos_y:.2f}, Z:{pos_z:.2f})\n"
                f"  > Rad: (X:{pos_x + RADAR_OFFSET_X:.2f}, Y:{pos_y + RADAR_OFFSET_Y:.2f}, Z:{pos_z + RADAR_OFFSET_Z:.2f})\n"
                f"    - Offset: [X:{RADAR_OFFSET_X}, Y:{RADAR_OFFSET_Y}, Z:{RADAR_OFFSET_Z}]\n"
                f"    - PPS: {new_pps}, H-FOV: {new_h_fov}, V-FOV: {new_v_fov}"
            )

        except Exception as e:
            print(f"[SensorManager] ERROR: spawn_sensors failed: {e}")
            self.destroy_sensors()

    def destroy_sensors(self):
        actors_to_destroy = []
        for a in (self.cam, self.rad):
            if a and a.is_alive:
                actors_to_destroy.append(a)

        if actors_to_destroy:
            print(f"[SensorManager] Destroying {len(actors_to_destroy)} sensors...")
            for a in actors_to_destroy:
                try:
                    if a.is_listening: a.stop()
                    a.destroy()
                except Exception as e:
                    print(f"[SensorManager] Error destroying sensor {a.id}: {e}")

        self.cam = None
        self.rad = None
        self.RAD_HISTORY.clear()
        self.RAD_RCS_HISTORY.clear()
        self.CAMERA_IMG = None

    def get_sensor_actors(self) -> List[carla.Actor]:
        return [a for a in (self.cam, self.rad) if a and a.is_alive]

    @staticmethod
    def _camera_callback(weak_self, image: carla.Image):
        self = weak_self()
        if not self: return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.CAMERA_IMG = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR).copy()

    @staticmethod
    def _radar_callback(weak_self, radar_data: carla.RadarMeasurement):
        self = weak_self()
        if not self: return

        ts = float(getattr(radar_data, "timestamp", 0.0))
        fr = int(getattr(radar_data, "frame", -1))
        
        frame_pts_xyzv = []
        frame_pts_full = []

        for d in radar_data:
            # 1. 3D 좌표 계산 (기존과 동일)
            x = d.depth * math.cos(d.altitude) * math.cos(d.azimuth)
            y = d.depth * math.cos(d.altitude) * math.sin(d.azimuth)
            z = d.depth * math.sin(d.altitude)
            
            # 2. [수정] 3D 코사인 오차 보정 (Radial Velocity -> Real Velocity)
            # 레이더는 "다가오는 속도"만 측정하므로, 각도(Azimuth, Altitude)만큼
            # 줄어든 값을 다시 복원해줘야 실제 차량 속도가 나옵니다.
            
            # 보정 계수 = cos(좌우각) * cos(상하각)
            correction_factor = math.cos(d.azimuth) * math.cos(d.altitude)
            
            # 각도가 90도에 가까우면(차가 옆으로 지나감) factor가 0이 되어 무한대로 튀는 것 방지
            if abs(correction_factor) < 0.1:
                v_comp = 0.0 
            else:
                v_comp = float(d.velocity) / correction_factor

            # 원래 d.velocity 대신 보정된 v_comp를 저장
            frame_pts_xyzv.append((x, y, z, v_comp))
            frame_pts_full.append((x, y, z, v_comp, ts, fr, None))

        self.RAD_HISTORY.append(frame_pts_xyzv)
        if len(frame_pts_full) > 0:
            self.RAD_RCS_HISTORY.append(frame_pts_full)