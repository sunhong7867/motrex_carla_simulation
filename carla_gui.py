#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    print("[GUI] carla 모듈을 찾을 수 없습니다.")

import main as carla_manager
import lane_utils
import perception_utils
import sensor_manager
from rendering import overlay_camera, render_radar_bev
from calibration_manager import save_extrinsic_json, run_calibration_pipeline, reset_calibration_buffer

# ==============================================================================
# --- (1/3) 차선 폴리곤 에디터 클래스 ---
# ==============================================================================

class LaneCanvas(QtWidgets.QWidget):
    """이미지 위에 차선 폴리곤을 그리고 편집하는 캔버스 위젯"""
    def __init__(self, bg_img, lane_polys, parent=None):
        super().__init__(parent)
        self.bg_img = bg_img  # OpenCV BGR Image
        self.h, self.w = self.bg_img.shape[:2]
        self.setFixedSize(self.w, self.h)
        
        # 데이터 관리
        self._lane_polys = lane_polys  # {name: np.array([[x,y],...])}
        self._current_lane_name = lane_utils.LANE_NAMES[0]
        self._editing_pts = []  # 현재 찍고 있는 점들
        
        # 그리기 도구 설정
        self.pen_boundary = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
        self.brush_fill = QtGui.QBrush(QtGui.QColor(0, 255, 0, 60))
        self.pen_editing = QtGui.QPen(QtGui.QColor(50, 180, 255), 2)
        self.font_label = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)

    def set_current_lane(self, name):
        self._current_lane_name = name
        self._editing_pts = [] # 차선 변경 시 편집 중인 점 초기화
        self.update()

    def finish_current_polygon(self):
        """현재 편집 중인 점들을 해당 차선의 폴리곤으로 저장"""
        if len(self._editing_pts) < 3:
            print(f"[LaneCanvas] 점이 부족합니다 (최소 3개).")
            return
        
        arr = np.array(self._editing_pts, dtype=np.int32)
        self._lane_polys[self._current_lane_name] = arr
        self._editing_pts = []
        self.update()
        print(f"[LaneCanvas] '{self._current_lane_name}' 저장됨.")

    def clear_current_lane(self):
        """현재 선택된 차선 데이터 삭제"""
        if self._current_lane_name in self._lane_polys:
            self._lane_polys[self._current_lane_name] = None
        self._editing_pts = []
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x, y = event.position().x(), event.position().y()
            self._editing_pts.append([x, y])
            self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        
        # 1. 배경 이미지 그리기
        qimg = QtGui.QImage(self.bg_img.data, self.w, self.h, 3*self.w, QtGui.QImage.Format_BGR888)
        p.drawImage(0, 0, qimg)

        # 2. 저장된 폴리곤 그리기
        for name, poly in self._lane_polys.items():
            if poly is None or len(poly) == 0: continue
            
            # 현재 선택된 차선은 붉은색 테두리로 강조
            if name == self._current_lane_name:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 3))
            else:
                p.setPen(self.pen_boundary)
            p.setBrush(self.brush_fill)

            path = QtGui.QPainterPath()
            pts = [QtCore.QPoint(int(x), int(y)) for x, y in poly]
            if not pts: continue
            
            path.moveTo(pts[0])
            for pt in pts[1:]: path.lineTo(pt)
            path.closeSubpath()
            
            p.drawPath(path)
            
            # 라벨 표시
            cx = int(np.mean(poly[:,0]))
            cy = int(np.mean(poly[:,1]))
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setFont(self.font_label)
            p.drawText(cx, cy, name)

        # 3. 편집 중인 선 그리기
        if self._editing_pts:
            p.setPen(self.pen_editing)
            pts = [QtCore.QPoint(int(x), int(y)) for x, y in self._editing_pts]
            for i in range(len(pts)-1):
                p.drawLine(pts[i], pts[i+1])
            # 마우스 커서 따라다니는 선은 생략(단순화)


class LaneEditorDialog(QtWidgets.QDialog):
    """LaneCanvas를 포함하는 다이얼로그 창 - 이미지 크기에 맞춰 자동 확장"""
    def __init__(self, bg_img, current_polys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lane Polygon Editor")

        # 1. 이미지 해상도 추출
        img_h, img_w = bg_img.shape[:2]
        
        # 2. 창 크기 설정 (너비는 이미지+여백, 높이는 이미지+상하단 컨트롤바 여유분)
        # 1280x720 이미지라면 약 1320x840 정도로 창이 뜹니다.
        target_w = img_w + 50
        target_h = img_h + 150
        self.resize(target_w, target_h)
        self.setMinimumSize(800, 600) # 최소 크기 방어선

        self.polys = current_polys.copy() # 사본 작업
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- 컨트롤 영역 (상단) ---
        h_ctrl = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        # lane_utils에 정의된 이름들 (IN1, IN2, OUT1, OUT2, OUT3 등)
        self.combo.addItems(lane_utils.LANE_NAMES)
        self.combo.currentTextChanged.connect(self.on_combo_changed)
        
        btn_finish = QtWidgets.QPushButton("이 레인 닫기 (Polygon Close)")
        btn_finish.clicked.connect(self.on_finish_poly)
        
        btn_clear = QtWidgets.QPushButton("이 레인 지우기")
        btn_clear.clicked.connect(self.on_clear_poly)
        
        h_ctrl.addWidget(QtWidgets.QLabel("Select Lane:"))
        h_ctrl.addWidget(self.combo)
        h_ctrl.addWidget(btn_finish)
        h_ctrl.addWidget(btn_clear)
        layout.addLayout(h_ctrl)

        # --- 캔버스 영역 (중앙, 이미지 원본 크기 유지) ---
        scroll = QtWidgets.QScrollArea()
        self.canvas = LaneCanvas(bg_img, self.polys)
        
        # 중요: 캔버스가 스크롤 영역 내에서 이미지 크기를 그대로 유지하도록 함
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False) # True로 하면 이미지가 창에 맞춰 축소될 수 있음
        scroll.setAlignment(QtCore.Qt.AlignCenter) # 창이 이미지보다 크면 중앙 정렬
        layout.addWidget(scroll)

        # --- 하단 버튼 (저장/취소) ---
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
        # 초기값 설정
        self.canvas.set_current_lane(self.combo.currentText())

    def on_combo_changed(self, text):
        self.canvas.set_current_lane(text)

    def on_finish_poly(self):
        self.canvas.finish_current_polygon()

    def on_clear_poly(self):
        self.canvas.clear_current_lane()

    def get_polys(self):
        return self.polys

# ==============================================================================
# --- (2/3) 로그 및 이미지 뷰어 헬퍼 ---
# ==============================================================================

class LogDialog(QtWidgets.QDialog):
    """터미널 로그 표시 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terminal Log")
        self.resize(800, 500)
        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.textChanged.connect(lambda: self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum()))
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.txt_log)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def log_message(self, message: str):
        self.txt_log.append(message)

class SensorImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(True)
        self.setMinimumSize(480, 360)
        
    @QtCore.Slot(QtGui.QImage)
    def update_image(self, qimg: QtGui.QImage):
        if qimg.isNull(): return
        self.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

# ==============================================================================
# --- (3/3) 메인 윈도우 ---
# ==============================================================================

class MainWindow(QtWidgets.QMainWindow):
    updateCamImage = QtCore.Signal(QtGui.QImage)
    updateRadImage = QtCore.Signal(QtGui.QImage)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.manager = carla_manager.CarlaManager()
        
        # 파라미터 초기화
        self.param_sensor_params = {'range': 120.0, 'h_fov': 120.0, 'v_fov': 30.0, 'pps': 12000, 'vel_range': 40.0}
        self.param_sensor_pos = { 'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'step': 0.5 }
        self.param_bev_view = { 'offset_x': -40, 'offset_y': 0, 'point_size': 2 }
        
        # 차선 상태 (5번 레인까지 포함)
        self.param_lane_state = { 
            'lane_on': True, 
            'in1': True, 'in2': True, 
            'out1': True, 'out2': True, 
        }
        self.param_overlay_radar_on = False
        self.param_show_sensor_debug_box = False
        
        # 로깅 변수
        self.is_recording = False
        self.record_buffer = []

        self.last_cam_frame_bgr: Optional[np.ndarray] = None
        self._last_cam_qimg: Optional[QtGui.QImage] = None
        self._last_rad_qimg: Optional[QtGui.QImage] = None
        
        self.setWindowTitle("CARLA Sensor GUI (Integrated)")
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
        v1 = QtWidgets.QVBoxLayout(grp1) # 레이아웃 변수명 통일 v1
        h1 = QtWidgets.QHBoxLayout()
        
        self.btn_run = QtWidgets.QPushButton("실행 (서버 시작)")
        self.btn_pause = QtWidgets.QPushButton("일시정지"); self.btn_pause.setCheckable(True)
        self.btn_step = QtWidgets.QPushButton("한 프레임")
        self.btn_exit = QtWidgets.QPushButton("종료")
        
        h1.addWidget(self.btn_run); h1.addWidget(self.btn_pause)
        h1.addWidget(self.btn_step); h1.addWidget(self.btn_exit)
        v1.addLayout(h1)
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

        # --- 4. Calibration ---
        grp4 = QtWidgets.QGroupBox("4. Calibration")
        v4 = QtWidgets.QVBoxLayout(grp4)

        self.btn_calibrate = QtWidgets.QPushButton("📸 데이터 수집 및 계산")
        self.btn_calib_reset = QtWidgets.QPushButton("데이터 초기화 (Reset)")
        
        self.btn_calibrate.setToolTip("여러 프레임에서 차량 데이터를 모은 뒤 계산합니다.")
        v4.addWidget(self.btn_calibrate)
        v4.addWidget(self.btn_calib_reset)
        v_main.addWidget(grp4)
        
        self.btn_calibrate.clicked.connect(self.on_click_calibrate)
        self.btn_calib_reset.clicked.connect(self.on_click_calib_reset)

        # --- 5. Lane Configuration (차선 폴리곤 편집 기능 포함) ---
        grp5 = QtWidgets.QGroupBox("5. Lane Configuration")
        v5 = QtWidgets.QVBoxLayout(grp5)
        
        # (1) 차선 On/Off 체크박스
        g5 = QtWidgets.QGridLayout()
        self.chk_in1 = QtWidgets.QCheckBox("IN1"); self.chk_in2 = QtWidgets.QCheckBox("IN2")
        self.chk_out1 = QtWidgets.QCheckBox("OUT1"); self.chk_out2 = QtWidgets.QCheckBox("OUT2")
        
        # 상태 반영
        self.chk_in1.setChecked(self.param_lane_state.get('in1', True))
        self.chk_in2.setChecked(self.param_lane_state.get('in2', True))
        self.chk_out1.setChecked(self.param_lane_state.get('out1', True))
        self.chk_out2.setChecked(self.param_lane_state.get('out2', True))

        g5.addWidget(self.chk_in1, 0, 0); g5.addWidget(self.chk_in2, 0, 1)
        g5.addWidget(self.chk_out1, 1, 0); g5.addWidget(self.chk_out2, 1, 1)
        v5.addLayout(g5)
        
        # (2) 폴리곤 편집 버튼
        self.btn_edit_lanes = QtWidgets.QPushButton("🖌️ 차선 폴리곤 그리기/편집")
        v5.addWidget(self.btn_edit_lanes)
        v_main.addWidget(grp5)

        # 시그널 연결
        self.chk_in1.toggled.connect(lambda c: self.param_lane_state.update(in1=c))
        self.chk_in2.toggled.connect(lambda c: self.param_lane_state.update(in2=c))
        self.chk_out1.toggled.connect(lambda c: self.param_lane_state.update(out1=c))
        self.chk_out2.toggled.connect(lambda c: self.param_lane_state.update(out2=c))
        self.btn_edit_lanes.clicked.connect(self.on_edit_lanes)

        # --- 6. Data Logging ---
        grp6 = QtWidgets.QGroupBox("6. Data Logging (CSV)")
        h6 = QtWidgets.QHBoxLayout(grp6)
        self.btn_record = QtWidgets.QPushButton("🔴 기록 시작")
        self.lbl_record_info = QtWidgets.QLabel("Ready")
        h6.addWidget(self.btn_record)
        h6.addWidget(self.lbl_record_info, 1)
        v_main.addWidget(grp6)
        
        self.btn_record.clicked.connect(self.on_toggle_record)
        
        # --- 7. Terminal Log ---
        grp7 = QtWidgets.QGroupBox("7. Terminal Log")
        h7 = QtWidgets.QHBoxLayout(grp7)
        self.btn_open_log = QtWidgets.QPushButton("터미널 로그 보기...")
        h7.addWidget(self.btn_open_log)
        v_main.addWidget(grp7)
        self.btn_open_log.clicked.connect(self.on_open_log)

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
            self.log_message(f"[SIM] Spawning sensors...")
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

    def on_edit_lanes(self):
        """5번 그룹: 차선 편집 다이얼로그 실행"""
        if self.last_cam_frame_bgr is None:
            self.log_message("[ERR] 편집할 카메라 영상이 없습니다.")
            return
        
        # 현재 저장된 폴리곤 로드
        current_polys = lane_utils.get_lane_polys()
        
        # 다이얼로그 실행
        dlg = LaneEditorDialog(self.last_cam_frame_bgr.copy(), current_polys, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            updated = dlg.get_polys()
            lane_utils.set_lane_polys(updated)
            self.log_message("[OK] 차선 폴리곤이 저장되었습니다.")

    def on_click_calibrate(self):
        if not getattr(self, "manager", None) or not getattr(self.manager, "sensor_manager", None):
            self.log_message("[CALIB] Manager not ready.")
            return

        sm = self.manager.sensor_manager
        dets = getattr(self.manager, "last_detections", [])
        if not dets:
            dets = getattr(sm, "LAST_DETECTIONS", [])
            
        if not dets:
            self.log_message("[CALIB] Error: 감지된 차량이 없습니다.")
            return

        if not hasattr(sm, "RAD_RCS_HISTORY") or len(sm.RAD_RCS_HISTORY) == 0:
            self.log_message("[CALIB] Error: 레이더 데이터가 없습니다.")
            return
        
        radar_history = list(sm.RAD_RCS_HISTORY)
        w, h = sensor_manager.IMG_WIDTH, sensor_manager.IMG_HEIGHT
        fov = 70.0
        K = perception_utils.get_camera_intrinsic_from_fov(fov, w, h)

        self.log_message(f"[CALIB] 누적 데이터({len(radar_history)} frames)로 계산 시작...")
        
        try:
            success, msg = run_calibration_pipeline(
                detections=dets,
                radar_history=radar_history,
                K=K, width=w, height=h
            )
        except Exception as e:
            self.log_message(f"[ERR] 파이프라인 실행 중 오류: {e}")
            return

        if success:
            print(f"[CALIB] {msg}")
            self.log_message(f"[CALIB] 성공! {msg}")
            self.lbl_rad_file.setText("Extrinsic: extrinsic.json")
            self.lbl_rad_warn.setText("")
        else:
            self.log_message(f"[CALIB] 진행 중: {msg}")

    def on_click_calib_reset(self):
        try:
            reset_calibration_buffer()
            self.log_message("[CALIB] Buffer Cleared.")
            self.lbl_rad_warn.setText("")
        except Exception as e:
            self.log_message(f"[ERR] 초기화 실패: {e}")

    def on_toggle_record(self):
        if not self.manager: return

        if not self.is_recording:
            self.is_recording = True
            self.record_buffer = [] 
            self.btn_record.setText("⬛ 기록 중지 & 저장")
            self.btn_record.setStyleSheet("background-color: #ffcccc;")
            self.lbl_record_info.setText("Recording...")
            self.log_message("[Data] 로깅 시작...")
        else:
            self.is_recording = False
            self.btn_record.setText("🔴 기록 시작")
            self.btn_record.setStyleSheet("")
            
            if not self.record_buffer:
                self.log_message("[Data] 데이터 없음.")
                self.lbl_record_info.setText("No data")
                return

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = "data_log"
            os.makedirs(base_dir, exist_ok=True)

            try:
                df = pd.DataFrame(self.record_buffer)
                df["err"] = df["v_meas"] - df["v_real"]
                df["abs_err"] = df["err"].abs()

                # Raw Save
                raw_dir = os.path.join(base_dir, "raw")
                os.makedirs(raw_dir, exist_ok=True)
                raw_path = os.path.join(raw_dir, f"raw_{ts}.csv")
                
                df_raw = df[["time", "id", "dist", "v_real", "v_meas", "err", "abs_err"]]
                df_raw.columns = ["Time", "VehicleID", "Distance_m", "GT_Speed_kmh", "Radar_kmh", "Error_Signed", "Error_Abs"]
                df_raw.to_csv(raw_path, index=False)
                self.log_message(f"[Data] Raw Saved: {raw_path}")

                # Vehicle Analysis
                veh_dir = os.path.join(base_dir, "per_vehicle")
                os.makedirs(veh_dir, exist_ok=True)
                veh_path = os.path.join(veh_dir, f"vehicle_error_{ts}.csv")

                veh_df = df.groupby("id").agg(
                    samples=("err", "count"),
                    mae=("abs_err", "mean"),
                    rmse=("err", lambda x: np.sqrt(np.mean(x**2))),
                    bias=("err", "mean")
                ).reset_index()
                veh_df.rename(columns={"id": "VehicleID", "mae": "MAE_kmh", "rmse": "RMSE_kmh", "bias": "Bias_kmh"}, inplace=True)
                veh_df.to_csv(veh_path, index=False)

                # Distance Analysis
                dist_dir = os.path.join(base_dir, "distance")
                os.makedirs(dist_dir, exist_ok=True)
                dist_path = os.path.join(dist_dir, f"distance_error_{ts}.csv")

                df["dist_bin"] = (df["dist"] // 10) * 10
                dist_df = df.groupby("dist_bin").agg(
                    samples=("err", "count"),
                    mae=("abs_err", "mean"),
                    rmse=("err", lambda x: np.sqrt(np.mean(x**2))),
                    bias=("err", "mean")
                ).reset_index()
                dist_df.rename(columns={"dist_bin": "DistanceBin_m", "mae": "MAE_kmh", "rmse": "RMSE_kmh", "bias": "Bias_kmh"}, inplace=True)
                dist_df.to_csv(dist_path, index=False)

                self.lbl_record_info.setText(f"Saved {len(df)} pts")

            except Exception as e:
                self.log_message(f"[ERR] Save Failed: {e}")
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

            # Camera View
            cam_data = self.manager.sensor_manager.CAMERA_IMG
            if cam_data is not None:
                self.last_cam_frame_bgr = cam_data.copy()
                
                try:
                    ret = overlay_camera(
                        cam_data.copy(),
                        manager=self.manager,
                        lane_state=self.param_lane_state,
                        bev_view_params=self.param_bev_view,
                        overlay_radar_on=self.param_overlay_radar_on,
                    )
                    
                    if isinstance(ret, tuple):
                        img_overlay, stats = ret
                    else:
                        img_overlay = ret 
                
                except Exception as e:
                    print(f"[GUI] Overlay Error: {e}")
                    img_overlay = cam_data

                if self.is_recording:
                    sm = self.manager.sensor_manager
                    if hasattr(sm, "LAST_MEASUREMENTS"):
                        self.record_buffer.extend(sm.LAST_MEASUREMENTS)
                        self.lbl_record_info.setText(f"Recording... [{len(self.record_buffer)} pts]")

                h, w, ch = img_overlay.shape
                qimg = QtGui.QImage(img_overlay.data, w, h, ch*w, QtGui.QImage.Format_BGR888)
                self._last_cam_qimg = qimg
                
            # Radar View
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
        self.log_message("[GUI] Closing...")
        self.timer.stop()
        try:
            if self.manager: self.manager.cleanup()
            self.log_dialog.close()
        except Exception as err:
            self.log_message(f"[ERR] Cleanup failed: {err}")
        finally:
            e.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()