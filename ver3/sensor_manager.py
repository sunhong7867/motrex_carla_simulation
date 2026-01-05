#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import weakref
import cv2
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

# --- 전역 설정값 ---
IMG_WIDTH, IMG_HEIGHT = 1280, 720

# 육교 기준 포즈(고정) - 시뮬레이션용
FIXED_X, FIXED_Y, FIXED_Z = 22.97, 136.05, 9.64
FIXED_PITCH, FIXED_YAW, FIXED_ROLL = -10.0, 0.0, 0.0

RAD_TICK = 0.05  # 20Hz


class SensorManager:
    """
    실센서 적용을 고려한 최소 구성:
      - RGB Camera
      - Radar (RCS 좌표계 4D: x,y,z,doppler + time/frame)

    버퍼:
      - CAMERA_IMG: (H,W,3) BGR
      - RAD_HISTORY: (x,y,z,doppler) (최근 15프레임)  -> BEV/디버깅용
      - RAD_RCS_HISTORY: (x,y,z,doppler,timestamp,frame_id,power) (최근 30프레임) -> 캘리브레이션/속도 메인
    """

    def __init__(self, world: carla.World):
        self.world = world

        # CARLA 센서 액터 핸들
        self.cam: Optional[carla.Actor] = None
        self.rad: Optional[carla.Actor] = None

        # 센서 데이터 버퍼
        self.CAMERA_IMG: Optional[np.ndarray] = None

        # Radar buffers
        self.RAD_HISTORY = deque(maxlen=15)
        self.RAD_RCS_HISTORY = deque(maxlen=30)

        self.gt_track = {
            "veh_id": -1,
            "last_seen": 0.0
        }

    def _tf_from(
        self,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        delta_yaw: float = 0.0
    ) -> carla.Transform:
        """육교 기준 위치에서 오프셋을 적용한 Transform (시뮬레이션용)"""
        return carla.Transform(
            carla.Location(FIXED_X + offset_x, FIXED_Y + offset_y, FIXED_Z + offset_z),
            carla.Rotation(FIXED_PITCH, FIXED_YAW + delta_yaw, FIXED_ROLL)
        )

    def spawn_sensors(self, sensor_params: Dict[str, Any], pos_params: Dict[str, Any]):
        """
        sensor_params: {'range', 'h_fov', 'v_fov', 'pps'}
        pos_params: {'x', 'y', 'z', 'yaw'}
        """
        self.destroy_sensors()

        new_range = sensor_params.get('range', 70.0)
        new_h_fov = sensor_params.get('h_fov', 40.0)
        new_v_fov = sensor_params.get('v_fov', 50.0)
        new_pps = sensor_params.get('pps', 12000)

        pos_x = pos_params.get('x', 0.0)
        pos_y = pos_params.get('y', 0.0)
        pos_z = pos_params.get('z', 0.0)
        rot_yaw = pos_params.get('yaw', 0.0)

        # -----------------------------------------------------------
        # [수정됨] 센서 위치 분리
        # -----------------------------------------------------------
        # 1. 카메라는 GUI에서 입력받은 위치 그대로 사용
        pose_cam = self._tf_from(pos_x, pos_y, pos_z, rot_yaw)

        # 2. 레이더는 Y축(오른쪽) 기준 -0.5m (즉, 왼쪽으로 0.5m 이동)
        pose_rad = self._tf_from(pos_x, pos_y - 0.5, pos_z, rot_yaw)
        # -----------------------------------------------------------

        bp = self.world.get_blueprint_library()
        weak_self = weakref.ref(self)

        try:
            # 1) RGB Camera (pose_cam 사용)
            cam_bp = bp.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            cam_bp.set_attribute('fov', '70')
            self.cam = self.world.spawn_actor(cam_bp, pose_cam)
            self.cam.listen(lambda image: SensorManager._camera_callback(weak_self, image))

            # 2) Radar (pose_rad 사용)
            rad_bp = bp.find('sensor.other.radar')
            rad_bp.set_attribute('range', f'{new_range}')
            rad_bp.set_attribute('horizontal_fov', f'{new_h_fov}')
            rad_bp.set_attribute('vertical_fov', f'{new_v_fov}')
            rad_bp.set_attribute('points_per_second', f'{new_pps}')
            rad_bp.set_attribute('sensor_tick', f'{RAD_TICK}')
            self.rad = self.world.spawn_actor(rad_bp, pose_rad)
            self.rad.listen(lambda data: SensorManager._radar_callback(weak_self, data))

            print(
                f"[SensorManager] Sensors spawned.\n"
                f"  > Cam: (X:{pos_x:.1f}, Y:{pos_y:.1f}, Z:{pos_z:.1f})\n"
                f"  > Rad: (X:{pos_x:.1f}, Y:{pos_y-0.5:.1f}, Z:{pos_z:.1f}) [Left 0.5m]"
            )

        except Exception as e:
            print(f"[SensorManager] ERROR: spawn_sensors failed: {e}")
            self.destroy_sensors()

    def destroy_sensors(self):
        """현재 관리 중인 센서 액터 파괴"""
        actors_to_destroy = []
        for a in (self.cam, self.rad):
            if a and a.is_alive:
                actors_to_destroy.append(a)

        if actors_to_destroy:
            print(f"[SensorManager] Destroying {len(actors_to_destroy)} sensors...")
            for a in actors_to_destroy:
                try:
                    if a.is_listening:
                        a.stop()
                    a.destroy()
                except Exception as e:
                    print(f"[SensorManager] Error destroying sensor {a.id}: {e}")

        self.cam = None
        self.rad = None

        self.RAD_HISTORY.clear()
        self.RAD_RCS_HISTORY.clear()
        self.CAMERA_IMG = None

    def get_sensor_actors(self) -> List[carla.Actor]:
        """현재 활성화된 센서 액터 리스트"""
        return [a for a in (self.cam, self.rad) if a and a.is_alive]

    # ---------------------------
    # Sensor Callbacks (Static)
    # ---------------------------

    @staticmethod
    def _camera_callback(weak_self, image: carla.Image):
        self = weak_self()
        if not self:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.CAMERA_IMG = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR).copy()

    @staticmethod
    def _radar_callback(weak_self, radar_data: carla.RadarMeasurement):
        self = weak_self()
        if not self:
            return

        ts = float(getattr(radar_data, "timestamp", 0.0))
        fr = int(getattr(radar_data, "frame", -1))
        power = None  # CARLA radar에는 power 없음

        frame_pts_xyzv = []
        frame_pts_full = []

        for d in radar_data:
            x = d.depth * math.cos(d.altitude) * math.cos(d.azimuth)
            y = d.depth * math.cos(d.altitude) * math.sin(d.azimuth)
            z = d.depth * math.sin(d.altitude)
            v = float(d.velocity)  # Doppler (m/s)

            frame_pts_xyzv.append((x, y, z, v))
            frame_pts_full.append((x, y, z, v, ts, fr, power))

        # BEV / 디버그용
        self.RAD_HISTORY.append(frame_pts_xyzv)

        # === 핵심: 프레임 단위로 저장 ===
        if len(frame_pts_full) > 0:
            self.RAD_RCS_HISTORY.append(frame_pts_full)