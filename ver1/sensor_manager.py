#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
import weakref
import cv2
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

# --- 인식 모듈 임포트 ---
import perception_utils

# --- 전역 설정값 ---
IMG_WIDTH, IMG_HEIGHT = 1280, 720

# 육교 기준 포즈(고정)
FIXED_X, FIXED_Y, FIXED_Z = 22.97, 136.05, 9.64
FIXED_PITCH, FIXED_YAW, FIXED_ROLL = -10.0, 0.0, 0.0

RAD_TICK = 0.05 # 20Hz

class SensorManager:
    def __init__(self, world: carla.World):
        self.world = world
        
        # 센서 액터 핸들
        self.cam: Optional[carla.Actor] = None
        self.dep: Optional[carla.Actor] = None
        self.rad: Optional[carla.Actor] = None
        
        # 센서 데이터 버퍼
        self.CAMERA_IMG: Optional[np.ndarray] = None
        self.DEPTH_IMG: Optional[np.ndarray] = None
        self.RAD_HISTORY = deque(maxlen=15)         # BEV용 (센서 좌표)
        self.RAD_WORLD_HISTORY = deque(maxlen=5)    # 오버레이용 (월드 좌표)

    def _tf_from(self, offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0,
                 delta_yaw: float = 0.0) -> carla.Transform:
        """육교 기준 위치에서 오프셋을 적용한 Transform을 반환"""
        return carla.Transform(
            carla.Location(FIXED_X + offset_x, FIXED_Y + offset_y, FIXED_Z + offset_z),
            carla.Rotation(FIXED_PITCH, FIXED_YAW + delta_yaw, FIXED_ROLL)
        )

    def spawn_sensors(self, sensor_params: Dict[str, Any], pos_params: Dict[str, Any]):
        """
        기존 센서를 모두 파괴하고, 새 파라미터로 센서를 스폰합니다.
        
        sensor_params: {'range', 'h_fov', 'v_fov', 'pps'}
        pos_params: {'x', 'y', 'z', 'yaw'}
        """
        self.destroy_sensors() # 기존 센서 파괴

        # 파라미터 추출
        new_range = sensor_params.get('range', 70.0)
        new_h_fov = sensor_params.get('h_fov', 40.0)
        new_v_fov = sensor_params.get('v_fov', 50.0)
        new_pps   = sensor_params.get('pps', 12000)
        
        pos_x = pos_params.get('x', 0.0)
        pos_y = pos_params.get('y', 0.0)
        pos_z = pos_params.get('z', 0.0)
        rot_yaw = pos_params.get('yaw', 0.0)
        
        # 모든 센서는 동일한 오프셋 포즈를 사용
        pose_offset = self._tf_from(pos_x, pos_y, pos_z, rot_yaw)
        
        bp = self.world.get_blueprint_library()
        weak_self = weakref.ref(self) # 콜백용 weakref

        try:
            # 1. RGB Camera
            cam_bp = bp.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            cam_bp.set_attribute('fov', '70')
            self.cam = self.world.spawn_actor(cam_bp, pose_offset)
            self.cam.listen(lambda image: SensorManager._camera_callback(weak_self, image))

            # 2. Depth Camera
            dep_bp = bp.find('sensor.camera.depth')
            dep_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            dep_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            dep_bp.set_attribute('fov', '70')
            self.dep = self.world.spawn_actor(dep_bp, pose_offset)
            self.dep.listen(lambda image: SensorManager._depth_callback(weak_self, image))

            # 3. Radar
            rad_bp = bp.find('sensor.other.radar')
            rad_bp.set_attribute('range', f'{new_range}')
            rad_bp.set_attribute('horizontal_fov', f'{new_h_fov}')
            rad_bp.set_attribute('vertical_fov', f'{new_v_fov}')
            rad_bp.set_attribute('points_per_second', f'{new_pps}')
            rad_bp.set_attribute('sensor_tick', f'{RAD_TICK}')
            self.rad = self.world.spawn_actor(rad_bp, pose_offset)
            self.rad.listen(lambda data: SensorManager._radar_callback(weak_self, data))

            print(f"[SensorManager] Sensors spawned @ offset (X:{pos_x:.1f}, Y:{pos_y:.1f}, Z:{pos_z:.1f}, Yaw:{rot_yaw:.1f}°)")
            
        except Exception as e:
            print(f"[SensorManager] ERROR: spawn_sensors failed: {e}")
            self.destroy_sensors() # 실패 시 모두 롤백

    def destroy_sensors(self):
        """
        현재 관리 중인 모든 센서 액터를 파괴합니다.
        """
        actors_to_destroy = []
        for a in (self.cam, self.dep, self.rad):
            if a and a.is_alive:
                actors_to_destroy.append(a)
                
        if not actors_to_destroy:
            return

        print(f"[SensorManager] Destroying {len(actors_to_destroy)} sensors...")
        for a in actors_to_destroy:
            try:
                if a.is_listening:
                    a.stop()
                a.destroy()
            except Exception as e:
                print(f"[SensorManager] Error destroying sensor {a.id}: {e}")
                
        self.cam = None
        self.dep = None
        self.rad = None
        self.RAD_HISTORY.clear()
        self.RAD_WORLD_HISTORY.clear()

    def get_sensor_actors(self) -> List[carla.Actor]:
        """
        현재 활성화된 센서 액터 리스트를 반환합니다.
        """
        return [a for a in (self.cam, self.dep, self.rad) if a and a.is_alive]

    # --- Sensor Callbacks (Static methods) ---

    @staticmethod
    def _camera_callback(weak_self, image: carla.Image):
        self = weak_self()
        if not self: return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.CAMERA_IMG = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR).copy()

    @staticmethod
    def _depth_callback(weak_self, image: carla.Image):
        self = weak_self()
        if not self: return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.DEPTH_IMG = perception_utils.depth_to_meters(arr)

    @staticmethod
    def _radar_callback(weak_self, radar_data: carla.RadarMeasurement):
        self = weak_self()
        if not self: return
        
        # 1. BEV용 (센서 좌표)
        frame_pts = []
        for d in radar_data:
            fx = d.depth * math.cos(d.altitude) * math.cos(d.azimuth)
            fy = d.depth * math.cos(d.altitude) * math.sin(d.azimuth)
            frame_pts.append((fx, fy, d.depth, float(d.velocity)))
        self.RAD_HISTORY.append(frame_pts)

        # 2. 오버레이용 (월드 좌표)
        if self.rad is None: return
        rtf = self.rad.get_transform()
        frame_world = []
        for d in radar_data:
            # 센서 로컬 좌표
            fx = d.depth * math.cos(d.altitude) * math.cos(d.azimuth)
            fy = d.depth * math.cos(d.altitude) * math.sin(d.azimuth)
            fz = d.depth * math.sin(d.altitude)
            rel = carla.Location(fx, fy, fz)
            
            # 월드 좌표로 변환
            wp = rtf.transform(rel)
            frame_world.append((wp, float(d.velocity), float(d.depth)))
        self.RAD_WORLD_HISTORY.append(frame_world)