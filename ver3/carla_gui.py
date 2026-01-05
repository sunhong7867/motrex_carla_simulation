#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === GUI 라이브러리 ===
import sys, os, math, time, re, subprocess, csv, datetime
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import cv2
from PySide6 import QtWidgets, QtGui, QtCore

# === CARLA 및 백엔드 모듈 임포트 ===
try:
    import carla
except ImportError:
    print("[GUI] carla 모듈을 찾을 수 없습니다. (carla_manager가 처리하므로 무시 가능)")

# 분리된 백엔드 모듈 임포트
import main as carla_manager
import lane_utils
import perception_utils
import sensor_manager
from rendering import overlay_camera, render_radar_bev
from calibration_manager import calibrate_radar_to_camera_pnp, save_extrinsic_json


# ==============================================================================
# --- (1/2) 로그 다이얼로그 및 헬퍼 위젯 ---
# ==============================================================================

class LogDialog(QtWidgets.QDialog):
    """터미널 로그를 표시하는 별도 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terminal Log")
        self.setModal(False)
        self.resize(800, 500)

        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        
        self.txt_log.textChanged.connect(
            lambda: self.txt_log.verticalScrollBar().setValue(
                self.txt_log.verticalScrollBar().maximum()
            )
        )
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.txt_log)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def log_message(self, message: str):
        self.txt_log.append(message)

    def closeEvent(self, event):
        self.hide()
        event.ignore()

class SensorImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(True)
        self.setMinimumSize(480, 360)
        
    @QtCore.Slot(QtGui.QImage)
    def update_image(self, qimg: QtGui.QImage):
        if qimg.isNull():
            return
        self.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

# ==============================================================================
# --- (2/2) 메인 윈도우 ---
# ==============================================================================

class MainWindow(QtWidgets.QMainWindow):
    updateCamImage = QtCore.Signal(QtGui.QImage)
    updateRadImage = QtCore.Signal(QtGui.QImage)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.manager = carla_manager.CarlaManager()
        
        # 기본 센서 파라미터 (GUI 설정창은 삭제되었으나, 백엔드 로직 유지를 위해 변수는 남겨둠)
        self.param_sensor_params = {'range': 120.0, 'h_fov': 120.0, 'v_fov': 30.0, 'pps': 12000, 'vel_range': 40.0}
        self.param_sensor_pos = { 'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'step': 0.5 }
        self.param_bev_view = { 'offset_x': -40, 'offset_y': 0, 'point_size': 2 }
        
        self.param_lane_state = { 'lane_on': True, 'in1': True, 'in2': True, 'out1': True, 'out2': True }
        self.param_overlay_radar_on = True
        self.param_show_sensor_debug_box = False
        
        # 데이터 로깅용 변수
        self.is_recording = False
        self.record_buffer = []

        self.last_cam_frame_bgr: Optional[np.ndarray] = None
        self._last_cam_qimg: Optional[QtGui.QImage] = None
        self._last_rad_qimg: Optional[QtGui.QImage] = None
        
        self.setWindowTitle("CARLA Sensor GUI")
        self.resize(1600, 900)

        self.central = QtWidgets.QWidget(); self.setCentralWidget(self.central)
        self.hbox = QtWidgets.QHBoxLayout(self.central)

        self.log_dialog = LogDialog(self)
        
        self._build_left_controls()
        self._build_right_views()
        
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.on_tick)

    def log_message(self, message: str):
        self.log_dialog.log_message(message)

    def _build_left_controls(self):
            v_main = QtWidgets.QVBoxLayout()
            v_main.setSpacing(10)

            # --- 1. Simulation Control ---
            grp1 = QtWidgets.QGroupBox("1. Simulation Control")
            h1 = QtWidgets.QHBoxLayout(grp1)
            self.btn_run = QtWidgets.QPushButton("실행 (서버 시작)")
            self.btn_pause = QtWidgets.QPushButton("일시정지"); self.btn_pause.setCheckable(True)
            self.btn_step = QtWidgets.QPushButton("한 프레임")
            self.btn_exit = QtWidgets.QPushButton("종료")
            h1.addWidget(self.btn_run); h1.addWidget(self.btn_pause)
            h1.addWidget(self.btn_step); h1.addWidget(self.btn_exit)
            v_main.addWidget(grp1)
            
            self.btn_run.clicked.connect(self.on_run)
            self.btn_pause.toggled.connect(self.on_toggle_pause)
            self.btn_step.clicked.connect(self.on_step_once)
            self.btn_exit.clicked.connect(self.close)

            # --- 2. Spawners ---
            grp2 = QtWidgets.QGroupBox("2. Spawners")
            f2 = QtWidgets.QFormLayout(grp2)
            self.spin_veh = QtWidgets.QSpinBox(); self.spin_veh.setRange(0, 300); self.spin_veh.setValue(100)
            self.btn_spawn_veh = QtWidgets.QPushButton("차량 스폰")
            self.btn_reset_veh = QtWidgets.QPushButton("리셋")
            self.btn_spawn_sensors = QtWidgets.QPushButton("센서 스폰")
            
            h_spawn_reset = QtWidgets.QHBoxLayout()
            h_spawn_reset.addWidget(self.btn_spawn_veh)
            h_spawn_reset.addWidget(self.btn_reset_veh)

            f2.addRow("차량 수:", self.spin_veh)
            f2.addRow(h_spawn_reset)
            f2.addRow(self.btn_spawn_sensors)
            v_main.addWidget(grp2)
            
            self.btn_spawn_veh.clicked.connect(self.on_spawn_vehicles)
            self.btn_reset_veh.clicked.connect(self.on_reset_vehicles)
            self.btn_spawn_sensors.clicked.connect(self.on_spawn_sensors)

            # --- 3. Sensor Parameters ---
            grp3 = QtWidgets.QGroupBox("3. Sensor Parameters")
            v3 = QtWidgets.QVBoxLayout(grp3)
            self.chk_overlay = QtWidgets.QCheckBox("카메라에 레이더 오버레이")
            self.chk_debug_box = QtWidgets.QCheckBox("시뮬레이션에 센서 위치 표시")
            
            self.chk_overlay.setChecked(self.param_overlay_radar_on)
            self.chk_debug_box.setChecked(self.param_show_sensor_debug_box)
            
            v3.addWidget(self.chk_overlay)
            v3.addWidget(self.chk_debug_box)
            v_main.addWidget(grp3)
            
            self.chk_overlay.toggled.connect(lambda c: setattr(self, 'param_overlay_radar_on', c))
            self.chk_debug_box.toggled.connect(lambda c: setattr(self, 'param_show_sensor_debug_box', c))

            # --- 4. Calibration (Modified) ---
            grp_calib = QtWidgets.QGroupBox("4. Calibration")
            v_calib = QtWidgets.QVBoxLayout(grp_calib)

            self.btn_calibrate = QtWidgets.QPushButton("📸 데이터 수집 및 계산 (Click Multiple Times)")
            self.btn_calib_reset = QtWidgets.QPushButton("데이터 초기화 (Reset)") # [추가됨]
            
            self.btn_calibrate.setToolTip(
                "한 번 클릭에 계산되지 않습니다.\n"
                "여러 프레임(다른 차량 위치)에서 반복 클릭하여\n"
                "데이터(점)를 8개 이상 모으면 자동으로 계산됩니다."
            )

            v_calib.addWidget(self.btn_calibrate)
            v_calib.addWidget(self.btn_calib_reset) # [추가됨]
            v_main.addWidget(grp_calib)
            
            self.btn_calibrate.clicked.connect(self.on_click_calibrate)
            self.btn_calib_reset.clicked.connect(self.on_click_calib_reset) # [추가됨]

            # --- 5. Lane (기존 4번 -> 5번) ---
            grp4 = QtWidgets.QGroupBox("5. Lane")
            v4 = QtWidgets.QVBoxLayout(grp4)
            g4 = QtWidgets.QGridLayout()
            self.chk_in1 = QtWidgets.QCheckBox("IN1"); self.chk_in2 = QtWidgets.QCheckBox("IN2")
            self.chk_out1 = QtWidgets.QCheckBox("OUT1"); self.chk_out2 = QtWidgets.QCheckBox("OUT2")
            self.chk_in1.setChecked(self.param_lane_state['in1'])
            self.chk_in2.setChecked(self.param_lane_state['in2'])
            self.chk_out1.setChecked(self.param_lane_state['out1'])
            self.chk_out2.setChecked(self.param_lane_state['out2'])
            g4.addWidget(self.chk_in1, 0, 0); g4.addWidget(self.chk_in2, 0, 1)
            g4.addWidget(self.chk_out1, 1, 0); g4.addWidget(self.chk_out2, 1, 1)
            v4.addLayout(g4)
            v_main.addWidget(grp4)

            self.chk_in1.toggled.connect(lambda c: self.param_lane_state.update(in1=c))
            self.chk_in2.toggled.connect(lambda c: self.param_lane_state.update(in2=c))
            self.chk_out1.toggled.connect(lambda c: self.param_lane_state.update(out1=c))
            self.chk_out2.toggled.connect(lambda c: self.param_lane_state.update(out2=c))
            
            # === 6. Data Logging (기존 5번 -> 6번) ===
            grp5 = QtWidgets.QGroupBox("6. Data Logging (CSV)")
            h5 = QtWidgets.QHBoxLayout(grp5)
            self.btn_record = QtWidgets.QPushButton("🔴 기록 시작 (Start Logging)")
            self.lbl_record_info = QtWidgets.QLabel("Ready")
            h5.addWidget(self.btn_record)
            h5.addWidget(self.lbl_record_info, 1)
            v_main.addWidget(grp5)
            
            self.btn_record.clicked.connect(self.on_toggle_record)
            
            # === 7. Terminal Log (기존 6번 -> 7번) ===
            grp6 = QtWidgets.QGroupBox("7. Terminal Log")
            h6 = QtWidgets.QHBoxLayout(grp6)
            self.btn_open_log = QtWidgets.QPushButton("터미널 로그 보기...")
            h6.addWidget(self.btn_open_log)
            v_main.addWidget(grp6)
            self.btn_open_log.clicked.connect(self.on_open_log)

            # === 8. Radar Status (기존 7번 -> 8번) ===
            grp7 = QtWidgets.QGroupBox("8. Radar Status")
            v7 = QtWidgets.QVBoxLayout(grp7)
            v7.setSpacing(5)
            
            # 정보 표시용 라벨들
            self.lbl_rad_total = QtWidgets.QLabel("Radar Points: -")
            self.lbl_rad_z = QtWidgets.QLabel("Valid (Z>0): -")
            self.lbl_rad_img = QtWidgets.QLabel("In Image: -")
            self.lbl_rad_file = QtWidgets.QLabel("Extrinsic: -")
            self.lbl_rad_warn = QtWidgets.QLabel("")
            self.lbl_rad_warn.setStyleSheet("color: red; font-weight: bold;")

            v7.addWidget(self.lbl_rad_total)
            v7.addWidget(self.lbl_rad_z)
            v7.addWidget(self.lbl_rad_img)
            v7.addWidget(self.lbl_rad_file)
            v7.addWidget(self.lbl_rad_warn)
            
            v_main.addWidget(grp7)

            v_main.addStretch(1)
            self.hbox.addLayout(v_main, 1)

            
    def _build_right_views(self):
        v = QtWidgets.QVBoxLayout()

        grp_cam = QtWidgets.QGroupBox("Camera View — BBox + Radar Overlay")
        self.view_cam = SensorImageView()
        lay_cam = QtWidgets.QVBoxLayout(grp_cam); lay_cam.addWidget(self.view_cam)
        v.addWidget(grp_cam, 2)

        grp_rad = QtWidgets.QGroupBox("3D Radar Point Cloud (BEV)")
        self.view_rad = SensorImageView()
        lay_rad = QtWidgets.QVBoxLayout(grp_rad); lay_rad.addWidget(self.view_rad)
        v.addWidget(grp_rad, 1)
        
        self.updateCamImage.connect(self.view_cam.update_image)
        self.updateRadImage.connect(self.view_rad.update_image)

        self.hbox.addLayout(v, 3)

    # ---------- (SLOTS) ----------

    def on_run(self):
        try:
            self.manager.connect(start_server=True)
            self.log_message("[OK] Connected. '센서 스폰'을 눌러주세요.")
            self.timer.start()
        except Exception as e:
            self.log_message(f"[ERR] connect failed: {e}")

    def on_toggle_pause(self, checked: bool):
        self.manager.toggle_pause()
        self.btn_pause.setText("▶ 재개" if checked else "일시정지")
        self.log_message("[SIM] 일시정지" if checked else "[SIM] 재개")

    def on_step_once(self):
        if self.manager.step_once():
            self.log_message("[SIM] 한 프레임 진행")
            self._render_and_update_views()
        else:
            self.log_message("[WARN] '일시정지' 상태에서만 한 프레임 진행이 가능합니다.")

    def on_spawn_sensors(self):
        if not (self.manager and self.manager.sensor_manager):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다.")
            return
        try:
            self.log_message(f"[SIM] Spawning sensors with params...")
            self.manager.sensor_manager.spawn_sensors(
                self.param_sensor_params,
                self.param_sensor_pos
            )
            self.manager.move_spectator_to_sensor_view(
                pos_z_offset=carla_manager.Z_OFFSET_VIEW,
                pos_params=self.param_sensor_pos
            )
            self.log_message("[OK] Sensors spawned.")
        except Exception as e:
            self.log_message(f"[ERR] on_spawn_sensors: {e}")

    def on_spawn_vehicles(self):
        if not (self.manager and self.manager.vehicle_manager and self.manager.world):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다.")
            return

        n = int(self.spin_veh.value())
        tm_port = 8000
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), "generate_traffic.py")
            cmd = [sys.executable, script_path, "-n", str(n), "--safe", "--tm-port", str(tm_port)]
            subprocess.Popen(cmd)
            self.log_message(f"[OK] Spawning {n} vehicles (external script).")
        except Exception as e:
            self.log_message(f"[ERR] Failed to execute command: {e}")

    def on_reset_vehicles(self):
        if not (self.manager and self.manager.vehicle_manager and self.manager.world):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다.")
            return
        try:
            removed = self.manager.vehicle_manager.reset_all_vehicles()
            self.manager.world.tick() 
            self.log_message(f"[OK] Vehicle Reset: Removed {removed} vehicles.")
        except Exception as e:
            self.log_message(f"[ERR] Destruction failed: {e}")

    def on_click_calibrate(self):
        """
        GUI는 이제 단순히 데이터를 수집해서 Manager에게 넘기기만 합니다.
        (데이터 참조 위치 수정됨: sm.LAST_DETECTIONS -> self.manager.last_detections)
        """
        if not getattr(self, "manager", None) or not getattr(self.manager, "sensor_manager", None):
            self.log_message("[CALIB] Manager not ready.")
            return

        sm = self.manager.sensor_manager
        
        # ==========================================================
        # [수정] YOLO 결과를 올바른 위치(manager)에서 가져옵니다.
        # ==========================================================
        dets = getattr(self.manager, "last_detections", [])
        if not dets:
            # 혹시 모르니 sm 쪽도 확인 (fallback)
            dets = getattr(sm, "LAST_DETECTIONS", [])
            
        if not dets:
            self.log_message("[CALIB] Error: 감지된 차량이 없습니다. (화면에 빨간 박스가 보여야 합니다)")
            return

        if not hasattr(sm, "RAD_RCS_HISTORY") or len(sm.RAD_RCS_HISTORY) == 0:
            self.log_message("[CALIB] Error: 레이더 데이터가 없습니다.")
            return
        radar_frame = sm.RAD_RCS_HISTORY[-1]

        # 카메라 파라미터 준비
        w, h = sensor_manager.IMG_WIDTH, sensor_manager.IMG_HEIGHT
        fov = 70.0
        K = perception_utils.get_camera_intrinsic_from_fov(fov, w, h)

        self.log_message(f"[CALIB] 차량 {len(dets)}대 감지됨. 캘리브레이션 요청 중...")

        # 2. Manager의 파이프라인 호출
        from calibration_manager import run_calibration_pipeline
        
        success, msg = run_calibration_pipeline(
            detections=dets,
            radar_frame=radar_frame,
            K=K,
            width=w,
            height=h
        )

        # 3. 결과 표시
        if success:
            print(f"[CALIB] {msg}")
            self.log_message(f"[CALIB] 성공! \n{msg}")
            self.lbl_rad_file.setText("Extrinsic: extrinsic.json (New)")
            self.lbl_rad_warn.setText("")
        else:
            print(f"[CALIB] Error: {msg}")
            self.log_message(f"[CALIB] 실패: {msg}")


    def on_click_calib_reset(self):
        """캘리브레이션 버퍼 초기화"""
        try:
            from calibration_manager import reset_calibration_buffer
            reset_calibration_buffer()
            self.log_message("[CALIB] 누적 데이터가 초기화되었습니다. (Buffer Cleared)")
            self.lbl_rad_warn.setText("") # 경고 메시지도 지움
        except Exception as e:
            self.log_message(f"[ERR] 초기화 실패: {e}")

            
    def on_toggle_record(self):
        if not self.manager: return

        if not self.is_recording:
            # === 기록 시작 ===
            self.is_recording = True
            self.record_buffer = [] 
            self.btn_record.setText("⬛ 기록 중지 (Stop & Save)")
            self.btn_record.setStyleSheet("background-color: #ffcccc;")
            self.lbl_record_info.setText("Recording... [0 pts]")
            self.log_message("[Data] 로깅 시작 (속도 0.1 이하는 자동 제외됨)...")
        else:
            # === 기록 중지 및 저장 ===
            self.is_recording = False
            self.btn_record.setText("🔴 기록 시작 (Start Logging)")
            self.btn_record.setStyleSheet("")
            
            if not self.record_buffer:
                self.log_message("[Data] 저장할 데이터가 없습니다 (유효 데이터 0건).")
                self.lbl_record_info.setText("No data saved")
                return

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = "data_log"
            os.makedirs(base_dir, exist_ok=True)

            try:
                df = pd.DataFrame(self.record_buffer)
                
                # 1. 오차 계산 (측정 - 참값)
                df["err"] = df["v_meas"] - df["v_real"]
                
                # [수정] 절대 오차(Absolute Error) 추가
                df["abs_err"] = df["err"].abs()

                # --- (1) Raw Data 저장 ---
                raw_dir = os.path.join(base_dir, "raw")
                os.makedirs(raw_dir, exist_ok=True)
                raw_path = os.path.join(raw_dir, f"raw_{ts}.csv")
                
                df_raw = df[["time", "id", "dist", "v_real", "v_meas", "err", "abs_err"]]
                df_raw.columns = ["Time", "VehicleID", "Distance_m", "GT_Speed_kmh", "Radar_kmh", "Error_Signed", "Error_Abs"]
                df_raw.to_csv(raw_path, index=False)
                self.log_message(f"[Data] Raw 저장: {raw_path}")

                # --- (2) Vehicle-wise Analysis (차량별) ---
                veh_dir = os.path.join(base_dir, "per_vehicle")
                os.makedirs(veh_dir, exist_ok=True)
                veh_path = os.path.join(veh_dir, f"vehicle_error_{ts}.csv")

                # [수정] MAE (절대 오차 평균) 계산 추가
                veh_df = df.groupby("id").agg(
                    samples=("err", "count"),
                    mae=("abs_err", "mean"),       # 절대 오차 평균 (요청하신 값)
                    rmse=("err", lambda x: np.sqrt(np.mean(x**2))),
                    bias=("err", "mean")           # 단순 평균 (경향성 파악용)
                ).reset_index()
                
                # 컬럼명 보기 좋게 변경
                veh_df.rename(columns={"id": "VehicleID", "mae": "MAE_kmh", "rmse": "RMSE_kmh", "bias": "Bias_kmh"}, inplace=True)
                veh_df.to_csv(veh_path, index=False)
                self.log_message(f"[Data] Vehicle 분석(MAE 포함) 저장: {veh_path}")

                # --- (3) Distance-wise Analysis (거리별) ---
                dist_dir = os.path.join(base_dir, "distance")
                os.makedirs(dist_dir, exist_ok=True)
                dist_path = os.path.join(dist_dir, f"distance_error_{ts}.csv")

                df["dist_bin"] = (df["dist"] // 10) * 10
                dist_df = df.groupby("dist_bin").agg(
                    samples=("err", "count"),
                    mae=("abs_err", "mean"),       # 절대 오차 평균
                    rmse=("err", lambda x: np.sqrt(np.mean(x**2))),
                    bias=("err", "mean")
                ).reset_index()
                
                dist_df.rename(columns={"dist_bin": "DistanceBin_m", "mae": "MAE_kmh", "rmse": "RMSE_kmh", "bias": "Bias_kmh"}, inplace=True)
                dist_df.to_csv(dist_path, index=False)
                self.log_message(f"[Data] Distance 분석(MAE 포함) 저장: {dist_path}")

                self.lbl_record_info.setText(f"Saved {len(df)} pts")

            except Exception as e:
                self.log_message(f"[ERR] 파일 저장 실패: {e}")
                self.lbl_record_info.setText("Save failed")
            
            self.record_buffer = []

    def on_open_log(self):
        self.log_dialog.show()
        self.log_dialog.raise_()
        self.log_dialog.activateWindow()

    # ---------- 메인 루프 ----------
    
    def on_tick(self):
        if not self.manager: return

        self.manager.tick()

        if not (self.manager.is_connected and 
                self.manager.sensor_manager and 
                self.manager.sensor_manager.cam):
            return

        self._render_and_update_views()

        if self.param_show_sensor_debug_box and not self.manager.is_paused:
            self.manager.draw_sensor_debug_shapes(self.param_sensor_pos)

    def _render_and_update_views(self):
            if not self.manager.sensor_manager: return

            # 1. 카메라 뷰 (+데이터 수집)
            cam_data = self.manager.sensor_manager.CAMERA_IMG
            if cam_data is not None:
                self.last_cam_frame_bgr = cam_data.copy()
                
                # overlay_camera가 이제 (img, stats) 튜플을 반환
                try:
                    ret = overlay_camera(
                        cam_data.copy(),
                        manager=self.manager,
                        lane_state=self.param_lane_state,
                        bev_view_params=self.param_bev_view,
                        overlay_radar_on=self.param_overlay_radar_on,
                    )
                    
                    # 반환값이 튜플인지 확인 (rendering.py 수정 적용 여부 체크)
                    if isinstance(ret, tuple):
                        img_overlay, stats = ret
                        
                        # [7번 그룹 업데이트]
                        self.lbl_rad_total.setText(f"Radar Points: {stats.get('total', 0)}")
                        self.lbl_rad_z.setText(f"Valid (Z>0): {stats.get('zpos', 0)}")
                        self.lbl_rad_img.setText(f"In Image: {stats.get('in_img', 0)}")
                        
                        # 파일명만 깔끔하게 표시
                        ext_path = str(stats.get('ext_file', 'None'))
                        self.lbl_rad_file.setText(f"Extrinsic: {os.path.basename(ext_path)}")
                        
                        # 경고 메시지
                        self.lbl_rad_warn.setText(stats.get('warning', ''))
                    else:
                        # 구버전 호환
                        img_overlay = ret 
                
                except Exception as e:
                    print(f"[GUI] Overlay Error: {e}")
                    img_overlay = cam_data

                # 렌더링 후 sensor_manager에서 가져오기
                if self.is_recording:
                    sm = self.manager.sensor_manager
                    if hasattr(sm, "LAST_MEASUREMENTS"):
                        self.record_buffer.extend(sm.LAST_MEASUREMENTS)
                        self.lbl_record_info.setText(f"Recording... [{len(self.record_buffer)} pts]")


                h, w, ch = img_overlay.shape
                qimg = QtGui.QImage(img_overlay.data, w, h, ch*w, QtGui.QImage.Format_BGR888)
                self._last_cam_qimg = qimg
                
            # 2. 레이더 뷰
            rad_qimg = render_radar_bev(
                manager=self.manager,
                sensor_params=self.param_sensor_params,
                bev_view_params=self.param_bev_view,
            )
            if rad_qimg is not None:
                self._last_rad_qimg = rad_qimg
                
            if self._last_cam_qimg: self.updateCamImage.emit(self._last_cam_qimg)
            if self._last_rad_qimg: self.updateRadImage.emit(self._last_rad_qimg)

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.log_message("[GUI] Closing application...")
        self.timer.stop()
        try:
            if self.manager: self.manager.cleanup()
            self.log_dialog.close()
        except Exception as err:
            self.log_message(f"[ERR] Cleanup failed: {err}")
        finally:
            self.log_message("[GUI] Exit.")
            e.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()