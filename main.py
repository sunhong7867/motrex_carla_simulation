#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import socket
import os, sys, subprocess, time, datetime, csv
import numpy as np  # np.isnan 사용을 위해 추가
from typing import Optional, List, Dict, Any, Tuple

# YOLO 모듈 (없을 경우 예외처리)
try:
    from yolo_detector import YOLODetector
except ImportError:
    print("[Main] Warning: 'yolo_detector' module not found. YOLO features may fail.")
    YOLODetector = None

# --- 분리된 모듈 임포트 ---
import sensor_manager
import vehicle_manager
# import weather_manager  <-- [삭제됨] 날씨 매니저 제거
import perception_utils
import lane_utils

# 스펙테이터 뷰 오프셋
Z_OFFSET_VIEW = 5.0

class CarlaManager:
    def __init__(self):
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        
        self.sensor_manager: Optional[sensor_manager.SensorManager] = None
        self.vehicle_manager: Optional[vehicle_manager.VehicleManager] = None
        # self.weather_manager: Optional[weather_manager.WeatherManager] = None <-- [삭제됨]
        
        self.is_connected: bool = False
        self.is_paused: bool = False
        
        self.carla_server_process: Optional[subprocess.Popen] = None
        self.manual_control_process: Optional[subprocess.Popen] = None

        if YOLODetector is not None:
            self.yolo = YOLODetector(
                weight_name="best.pt",
                device="cuda"  # RK3588이면 "cpu"로 변경 필요
            )
        else:
            self.yolo = None

    @staticmethod
    def _is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((host, port)) == 0

    def _start_carla_server_process(self, host='127.0.0.1', port=2000) -> bool:
        """
        Popen 실행 시 CWD(Current Working Directory)를 
        CarlaUE4.sh가 있는 루트 폴더로 강제 지정합니다.
        """
        carla_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        carla_path = os.path.join(carla_root_dir, 'CarlaUE4.sh')

        if self._is_port_in_use(host, port):
            # print(f"[Manager] Port {port} is already in use")
            return True

        if not os.path.exists(carla_path):
            print(f"[Manager] CarlaUE4.sh not found at {carla_path}. Assuming server is already running.")
            return True

        try:
            cmd = [carla_path, "-qualityLevel=Low", f"-carla-rpc-port={port}", "-carla-streaming-port=0"]
            
            self.carla_server_process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                cwd=carla_root_dir  # 작업 디렉토리를 carla 루트로 설정
            )
            print("[Manager] Starting CARLA server... waiting 20s")
            time.sleep(20)
            return True
        except Exception as e:
            print(f"[Manager] ERROR: CARLA server run failed: {e}")
            return False

    def connect(self, start_server: bool = True, host='127.0.0.1', port=2000, tm_port=8000):
        if self.is_connected:
            print("[Manager] Already connected."); return
        if start_server and not self._start_carla_server_process(host, port):
            raise RuntimeError("Failed to start CARLA server process.")

        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()

            # 동기 모드 설정
            s = self.world.get_settings()
            s.synchronous_mode = True
            s.fixed_delta_seconds = 0.05
            self.world.apply_settings(s)

            # 매니저들 초기화
            self.sensor_manager = sensor_manager.SensorManager(self.world)
            self.vehicle_manager = vehicle_manager.VehicleManager(self.client, self.world, tm_port=tm_port)
            # self.weather_manager = weather_manager.WeatherManager(self.world) <-- [삭제됨]
            
            # 차선 데이터 로드
            try:
                lane_utils.load_lane_polys()
            except Exception as e:
                print(f"[Manager] Warning: Failed to load lane polys: {e}")

            # 초기 스펙테이터 뷰 설정
            self.move_spectator_to_sensor_view(pos_z_offset=Z_OFFSET_VIEW)
            
            self.is_connected = True
            print("[Manager] Connection successful. Synchronous mode enabled.")
            
        except Exception as e:
            self.is_connected = False
            self.world = None
            self.client = None
            raise RuntimeError(f"CARLA connection failed: {e}")

    def tick(self) -> bool:
        """
        월드의 틱을 1회 진행합니다.
        GUI의 QTimer에 의해 주기적으로 호출됩니다.
        """
        if not (self.is_connected and self.world and not self.is_paused):
            return False # 틱 진행 안 함

        try:
            # [삭제됨] 날씨 매니저 틱 로직 제거
            
            # 메인 월드 틱
            self.world.tick()
            return True
            
        except Exception as e:
            print(f"[Manager] ERROR: tick failed: {e}")
            self.is_connected = False # 연결 끊김으로 처리
            return False

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        print(f"[Manager] Pause {'Enabled' if self.is_paused else 'Disabled'}")

    def step_once(self):
        """
        일시정지 상태일 때 한 프레임만 강제로 틱을 실행합니다.
        """
        if self.is_connected and self.world and self.is_paused:
            print("[Manager] Step one frame")
            try:
                self.world.tick()
                return True
            except Exception as e:
                print(f"[Manager] ERROR: step_once failed: {e}")
        return False

    def move_spectator_to_sensor_view(self, pos_z_offset: float = 0.0, 
                                      pos_params: Optional[Dict[str, Any]] = None):
        """
        스펙테이터 뷰를 현재 센서 위치(+오프셋)로 이동합니다.
        """
        if not self.world: return
        
        if pos_params is None:
             pos_params = {} # 기본값

        target_tf = sensor_manager.SensorManager._tf_from(
            self=None, # static method 호출
            offset_x = pos_params.get('x', 0.0), 
            offset_y = pos_params.get('y', 0.0), 
            offset_z = pos_params.get('z', 0.0) + pos_z_offset, # 뷰 오프셋 추가
            delta_yaw = pos_params.get('yaw', 0.0)
        )
        self.world.get_spectator().set_transform(target_tf)

    def draw_sensor_debug_shapes(self, pos_params: Dict[str, Any]):
        """
        시뮬레이션 월드에 센서 위치를 빨간 박스로 그립니다.
        """
        if not self.world: return
        
        box_extent = carla.Location(0.25, 0.25, 0.25)
        life_time = 0.1 # 2프레임

        offset_rot = carla.Rotation(
            sensor_manager.FIXED_PITCH,
            sensor_manager.FIXED_YAW + pos_params.get('yaw', 0.0),
            sensor_manager.FIXED_ROLL
        )
        offset_loc = carla.Location(
            sensor_manager.FIXED_X + pos_params.get('x', 0.0),
            sensor_manager.FIXED_Y + pos_params.get('y', 0.0),
            sensor_manager.FIXED_Z + pos_params.get('z', 0.0)
        )
        
        self.world.debug.draw_box(
            carla.BoundingBox(offset_loc, box_extent),
            offset_rot, 
            life_time,
            carla.Color(255, 0, 0, 255), # Red
            1.0 # thickness
        )

    def start_manual_control(self) -> Tuple[bool, str]:
        """
        manual_control.py 스크립트를 별도 프로세스로 실행합니다.
        """
        if self.manual_control_process and self.manual_control_process.poll() is None:
            return False, "Manual control is already running."

        # 스크립트 경로 확인 (현재 파일과 같은 폴더에 있다고 가정)
        script_path = os.path.join(os.path.dirname(__file__), "manual_control.py")
        if not os.path.exists(script_path):
            return False, f"'manual_control.py' not found in script directory."
            
        try:
            # 1. 메인 GUI의 틱을 멈춤
            self.is_paused = True
            
            # 2. 별도 프로세스로 실행
            cmd = [sys.executable, script_path, "--sync", "--res=1600x900"]
            self.manual_control_process = subprocess.Popen(cmd)
            return True, "Manual control process started."

        except Exception as e:
            self.is_paused = False # 실패 시 원복
            return False, f"Failed to start manual_control.py: {e}"

    def check_manual_control_finished(self) -> bool:
        """
        수동 주행 프로세스가 종료되었는지 확인합니다.
        """
        if self.manual_control_process and self.manual_control_process.poll() is not None:
            self.manual_control_process = None
            
            # 1. 메인 GUI 틱 재개
            self.is_paused = False
            
            # 2. 동기 모드 재설정 (중요)
            if self.world:
                try:
                    s = self.world.get_settings()
                    if not s.synchronous_mode:
                        s.synchronous_mode = True
                        s.fixed_delta_seconds = 0.05
                        self.world.apply_settings(s)
                        print("[Manager] Re-enabled synchronous mode for main world.")
                except Exception as e:
                     print(f"[Manager] ERROR: Failed to re-apply sync settings: {e}")
            return True # 종료됨
        return False # 아직 실행 중

    def save_visible_vehicle_data(self) -> Tuple[bool, str]:
        """
        현재 보이는 차량의 ID, 계산 속도, CARLA 속도를 CSV 파일로 저장합니다.
        """
        if not self.world or not self.sensor_manager or not self.sensor_manager.cam:
            return False, "센서가 스폰되지 않았습니다."
            
        cam = self.sensor_manager.cam
        depth_img = self.sensor_manager.DEPTH_IMG
        rad_history = self.sensor_manager.RAD_WORLD_HISTORY
        w, h = sensor_manager.IMG_WIDTH, sensor_manager.IMG_HEIGHT

        vehicles = self.world.get_actors().filter('vehicle.*')
        rows = []
        
        for v in vehicles:
            if not v.is_alive: continue
            
            # 1. BBox 계산
            bbox = perception_utils.get_2d_bbox(cam, v, w, h)
            if bbox is None: continue
            x1,y1,x2,y2,z_center = bbox
            
            # 2. 가려짐(Occlusion) 체크
            if perception_utils.is_occluded(depth_img, (x1,y1,x2,y2), z_center, w, h):
                continue
                
            # (차선 필터링은 GUI의 overlay_camera에서 하므로 여기서는 생략)
            
            # 3. 속도 계산
            v_carla_kph = v.get_velocity().length() * 3.6
            v_calc_kph = perception_utils.estimate_speed_from_radar(
                cam, rad_history, (x1,y1,x2,y2), z_center, w, h
            )
            
            rows.append((v.id, ("" if np.isnan(v_calc_kph) else f"{v_calc_kph: .3f}"), f"{v_carla_kph:.3f}"))

        if not rows:
            return True, "보이는 차량 없음 (저장 안 함)"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "data_log"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"visible_stats_{ts}.csv")
        
        try:
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "v_calc_kph", "v_carla_kph"])
                for r in rows: w.writerow(r)
            return True, f"저장 완료: {out_path}"
        except Exception as e:
            return False, f"파일 저장 실패: {e}"

    def cleanup(self):
        """
        모든 액터와 프로세스를 정리하고 연결을 종료합니다.
        """
        print("[Manager] Cleaning up...")
        
        # 1. 수동 주행 프로세스 강제 종료
        if self.manual_control_process and self.manual_control_process.poll() is None:
            self.manual_control_process.terminate()
            self.manual_control_process.wait()
            self.manual_control_process = None
            print("[Manager] Manual control process terminated.")
            
        # 2. 모든 액터 파괴
        if self.client and self.world:
            try:
                if self.sensor_manager:
                    self.sensor_manager.destroy_sensors()
                if self.vehicle_manager:
                    self.vehicle_manager.reset_all_vehicles()
                    
                s = self.world.get_settings()
                if s.synchronous_mode:
                    s.synchronous_mode = False
                    s.fixed_delta_seconds = None
                    self.world.apply_settings(s)
                    print("[Manager] Synchronous mode disabled.")
            except Exception as e:
                print(f"[Manager] ERROR during actor cleanup: {e}")

        # 4. CARLA 서버 프로세스 종료
        if self.carla_server_process and self.carla_server_process.poll() is None:
            try:
                print("[Manager] Sending SIGTERM (15) to server process...")
                self.carla_server_process.terminate() # SIGTERM 전송
                self.carla_server_process.wait(timeout=5) # 5초 대기
                print("[Manager] CARLA server process terminated (SIGTERM).")
            except subprocess.TimeoutExpired:
                print("[Manager] Server did not terminate, sending SIGKILL (9)...")
                self.carla_server_process.kill() 
                self.carla_server_process.wait() 
                print("[Manager] CARLA server process killed (SIGKILL).")
            except Exception as e:
                print(f"[Manager] Failed to terminate server process: {e}")
                self.carla_server_process.kill() 
                print("[Manager] CARLA server process killed.")
                
            self.carla_server_process = None
            
        self.client = None
        self.world = None
        self.is_connected = False