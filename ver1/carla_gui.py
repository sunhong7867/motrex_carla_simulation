#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === GUI 라이브러리 ===
import sys, os, math, time, re, subprocess
from typing import Optional, List, Tuple, Dict
import numpy as np
import cv2
from PySide6 import QtWidgets, QtGui, QtCore

# === CARLA 및 백엔드 모듈 임포트 ===
try:
    # CARLA 모듈은 carla_manager가 내부적으로 처리하므로 여기서는 임포트 X
    import carla
except ImportError:
    print("[GUI] carla 모듈을 찾을 수 없습니다. (carla_manager가 처리하므로 무시 가능)")

# 분리된 백엔드 모듈 임포트
import main as carla_manager
import lane_utils         # load/save/is_in
import perception_utils   # get_2d_bbox, is_occluded 등 (편집기/차선 등에 여전히 사용)
import sensor_manager     # IMG_WIDTH, IMG_HEIGHT 등 설정값
from rendering import overlay_camera, render_radar_bev

# ==============================================================================
# --- (1/4) 로그 다이얼로그 ---
# ==============================================================================

class LogDialog(QtWidgets.QDialog):
    """[9] 터미널 로그를 표시하는 별도 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("9. Terminal Log")
        self.setModal(False) # 모달리스(Modaless)로 설정
        self.resize(800, 500) # 창 크기 키움

        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        
        # 텍스트가 추가될 때 자동으로 스크롤 내리기
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
        # 닫기 버튼을 눌렀을 때 파괴되는 대신 숨겨짐
        self.hide()
        event.ignore()

# ==============================================================================
# --- (2/4) GUI 헬퍼 위젯 (뷰어, 차선 편집기) ---
# ==============================================================================

class SensorImageView(QtWidgets.QLabel):
    """센서 이미지를 표시하는 QLabel 위젯"""
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

class LaneCanvas(QtWidgets.QWidget):
    """[6] 차선 폴리곤 편집용 캔버스"""
    pointAdded = QtCore.Signal()
    def __init__(self, base_img_bgr, lane_polys_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.base_bgr = base_img_bgr
        self.base_rgb = cv2.cvtColor(self.base_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = self.base_rgb.shape
        self.setMinimumSize(w, h)
        self._lane_polys = {k:(v.copy() if isinstance(v,np.ndarray) else None) for k,v in lane_polys_dict.items()}
        self._current_lane = lane_utils.LANE_NAMES[0]
        self._editing_pts = []
        self._hover = None

    def setCurrentLane(self, name: str):
        self._current_lane = name; self.update()
    def clearCurrentLane(self):
        self._lane_polys[self._current_lane] = None; self._editing_pts = []; self.update()
    def getPolys(self):
        return {k:(v.copy() if isinstance(v,np.ndarray) else None) for k,v in self._lane_polys.items()}
    def finishPolygon(self):
        if len(self._editing_pts) >= 3:
            self._lane_polys[self._current_lane] = np.array(self._editing_pts, dtype=np.int32)
            self._editing_pts = []; self.update()
    def undoPoint(self):
        if self._editing_pts: self._editing_pts.pop(); self.update()
    def mouseMoveEvent(self, e): 
        self._hover = (e.pos().x(), e.pos().y()); self.update()
    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._editing_pts.append((e.pos().x(), e.pos().y())); self.pointAdded.emit(); self.update()
    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        qimg = QtGui.QImage(self.base_rgb.data, self.base_rgb.shape[1], self.base_rgb.shape[0],
                            self.base_rgb.strides[0], QtGui.QImage.Format_RGB888)
        p.drawImage(0, 0, qimg)
        pen_saved = QtGui.QPen(QtGui.QColor(0, 200, 0), 2); brush_saved = QtGui.QBrush(QtGui.QColor(0, 255, 0, 60))
        for name, poly in self._lane_polys.items():
            if poly is None: continue
            path = QtGui.QPainterPath(); pts = [QtCore.QPoint(int(x), int(y)) for x,y in poly.reshape(-1,2)]
            if not pts: continue
            path.moveTo(pts[0]); [path.lineTo(pt) for pt in pts[1:]]; path.closeSubpath()
            p.setPen(pen_saved); p.fillPath(path, brush_saved); p.drawPath(path)
        if self._editing_pts:
            pen_edit = QtGui.QPen(QtGui.QColor(50, 180, 255), 2); p.setPen(pen_edit)
            for i in range(len(self._editing_pts)-1):
                p.drawLine(self._editing_pts[i][0], self._editing_pts[i][1], self._editing_pts[i+1][0], self._editing_pts[i+1][1])
            for (x,y) in self._editing_pts: p.drawEllipse(QtCore.QPoint(x,y), 3, 3)
        if self._hover is not None:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 1, QtCore.Qt.DashLine))
            p.drawEllipse(QtCore.QPoint(self._hover[0], self._hover[1]), 3, 3)

class LaneEditorDialog(QtWidgets.QDialog):
    def __init__(self, base_img_bgr, lane_polys_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("6. 폴리곤 편집")
        self.setModal(True)
        self.resize(1300, 900) # 창 크기 키움
        
        self.canvas = LaneCanvas(base_img_bgr, lane_polys_dict)
        self.cbo_lane = QtWidgets.QComboBox(); self.cbo_lane.addItems(lane_utils.LANE_NAMES)
        self.cbo_lane.currentTextChanged.connect(self.canvas.setCurrentLane)
        btn_undo  = QtWidgets.QPushButton("마지막 점 취소"); btn_clear = QtWidgets.QPushButton("이 레인 지우기")
        btn_done  = QtWidgets.QPushButton("이 레인 닫기(완성)"); btn_undo.clicked.connect(self.canvas.undoPoint)
        btn_clear.clicked.connect(self.canvas.clearCurrentLane); btn_done.clicked.connect(self.canvas.finishPolygon)
        top = QtWidgets.QHBoxLayout(); top.addWidget(QtWidgets.QLabel("편집 레인:")); top.addWidget(self.cbo_lane)
        top.addStretch(1); top.addWidget(btn_undo); top.addWidget(btn_clear); top.addWidget(btn_done)
        
        lbl_help = QtWidgets.QLabel("이미지 위를 클릭해서 꼭짓점을 추가하세요. 최소 3점 후 ‘이 레인 닫기(완성)’. 각 레인을 선택해 반복합니다.")
        btn_save = QtWidgets.QPushButton("저장"); btn_cancel = QtWidgets.QPushButton("취소")
        btns = QtWidgets.QHBoxLayout(); btns.addStretch(1); btns.addWidget(btn_save); btns.addWidget(btn_cancel)

        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(top)
        scroll_area = QtWidgets.QScrollArea(); scroll_area.setWidget(self.canvas); scroll_area.setWidgetResizable(True)
        lay.addWidget(scroll_area, 1); lay.addWidget(lbl_help); lay.addLayout(btns)
        btn_save.clicked.connect(self.accept); btn_cancel.clicked.connect(self.reject)
        
    def get_polys(self): return self.canvas.getPolys()

# ==============================================================================
# --- (3/4) 설정 다이얼로그 (날씨, 레이더, 위치, BEV) ---
# ==============================================================================

class WeatherControlDialog(QtWidgets.QDialog):
    """[3] 날씨 조절 다이얼로그"""
    def __init__(self, manager: carla_manager.CarlaManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3. 날씨 조절")
        self.setModal(True)
        self.manager = manager
        self.resize(400, 200) # 창 크기 키움

        layout = QtWidgets.QVBoxLayout(self)
        self.cbo_weather = QtWidgets.QComboBox()
        if self.manager.weather_manager:
            self.cbo_weather.addItems(self.manager.weather_manager.get_preset_names())
        
        self.chk_dynamic = QtWidgets.QCheckBox("동적 날씨 활성화 (Dynamic Weather)")
        if self.manager.weather_manager:
            self.chk_dynamic.setChecked(self.manager.weather_manager.is_dynamic)

        btn_apply = QtWidgets.QPushButton("적용")
        btn_apply.clicked.connect(self.on_apply)

        layout.addWidget(QtWidgets.QLabel("날씨 프리셋 선택 (동적 날씨 체크 해제 시 적용):"))
        layout.addWidget(self.cbo_weather)
        layout.addWidget(self.chk_dynamic)
        layout.addWidget(btn_apply)

    def on_apply(self):
        if not self.manager.weather_manager:
            self.parent().log_message("[ERR] WeatherManager가 초기화되지 않았습니다.")
            return

        try:
            is_dynamic = self.chk_dynamic.isChecked()
            if is_dynamic:
                if not self.manager.weather_manager.is_dynamic:
                    self.manager.weather_manager.start_dynamic_weather()
                    self.parent().log_message("[SIM] 동적 날씨 시작됨")
            else:
                if self.manager.weather_manager.is_dynamic:
                    self.manager.weather_manager.stop_dynamic_weather()
                    self.parent().log_message("[SIM] 동적 날씨 중지됨")
                
                preset_name = self.cbo_weather.currentText()
                self.manager.weather_manager.set_weather_by_preset_name(preset_name)
                self.parent().log_message(f"[SIM] 날씨 변경: {preset_name}")
            
            self.accept()
        except Exception as e:
            self.parent().log_message(f"[ERR] 날씨 변경 실패: {e}")

class RadarSettingsDialog(QtWidgets.QDialog):
    """[4] 레이더 설정 다이얼로그"""
    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("4. 레이더 설정")
        self.setModal(True)
        self.resize(450, 250) # 창 크기 키움
        
        f = QtWidgets.QFormLayout(self)
        self.spin_range = QtWidgets.QDoubleSpinBox()
        self.spin_h_fov = QtWidgets.QDoubleSpinBox()
        self.spin_v_fov = QtWidgets.QDoubleSpinBox()
        self.spin_pps = QtWidgets.QDoubleSpinBox()

        self.spin_range.setRange(10.0, 200.0); self.spin_range.setSuffix(" m"); self.spin_range.setSingleStep(5.0)
        self.spin_h_fov.setRange(1.0, 180.0); self.spin_h_fov.setSuffix(" °"); self.spin_h_fov.setSingleStep(5.0)
        self.spin_v_fov.setRange(1.0, 90.0); self.spin_v_fov.setSuffix(" °"); self.spin_v_fov.setSingleStep(2.0)
        self.spin_pps.setRange(1000.0, 200000.0); self.spin_pps.setSuffix(" pps"); self.spin_pps.setSingleStep(1000.0); self.spin_pps.setDecimals(0)
        
        if params:
            self.spin_range.setValue(params.get('range', 70.0))
            self.spin_h_fov.setValue(params.get('h_fov', 40.0))
            self.spin_v_fov.setValue(params.get('v_fov', 50.0))
            self.spin_pps.setValue(params.get('pps', 12000))

        f.addRow("최대 거리 (Range)", self.spin_range)
        f.addRow("좌우 범위 (H-FoV)", self.spin_h_fov)
        f.addRow("상하 범위 (V-FoV)", self.spin_v_fov)
        f.addRow("초당 포인트 (PPS)", self.spin_pps)
        f.addRow(QtWidgets.QLabel("<b>참고:</b> 'OK'를 누르면 센서가 다시 스폰됩니다."))
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        f.addRow(btn_box)
        
    def get_params(self):
        return {
            'range': self.spin_range.value(), 'h_fov': self.spin_h_fov.value(),
            'v_fov': self.spin_v_fov.value(), 'pps': self.spin_pps.value()
        }

class SensorPositionDialog(QtWidgets.QDialog):
    """[5] 센서 위치 설정 다이얼로그"""
    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("5. 센서 위치 설정")
        self.setModal(True)
        self.resize(450, 400) # 창 크기 키움
        
        self.params = params.copy() if params else {}
        
        g = QtWidgets.QGridLayout(self)
        self.spin_pos_step = QtWidgets.QDoubleSpinBox()
        self.spin_pos_step.setRange(0.1, 10.0); self.spin_pos_step.setValue(self.params.get('step', 0.5))
        self.spin_pos_step.setSuffix(" m (이동 스텝)"); self.spin_pos_step.setSingleStep(0.1)
        self.spin_pos_step.valueChanged.connect(lambda v: self.params.update(step=v))

        btn_px = QtWidgets.QPushButton("+X (Forward)"); btn_nx = QtWidgets.QPushButton("-X (Back)")
        btn_py = QtWidgets.QPushButton("+Y (Right)");   btn_ny = QtWidgets.QPushButton("-Y (Left)")
        btn_pz = QtWidgets.QPushButton("+Z (Up)");      btn_nz = QtWidgets.QPushButton("-Z (Down)")
        btn_rot_l = QtWidgets.QPushButton("Rotate -90° (Left)"); btn_rot_r = QtWidgets.QPushButton("Rotate +90° (Right)")
        btn_pos_reset = QtWidgets.QPushButton("Reset Position (0,0,0)"); btn_rot_reset = QtWidgets.QPushButton("Reset Rotation (0°)")
        
        btn_px.clicked.connect(lambda: self._shift_pos(dx=self.params['step']))
        btn_nx.clicked.connect(lambda: self._shift_pos(dx=-self.params['step']))
        btn_py.clicked.connect(lambda: self._shift_pos(dy=self.params['step']))
        btn_ny.clicked.connect(lambda: self._shift_pos(dy=-self.params['step']))
        btn_pz.clicked.connect(lambda: self._shift_pos(dz=self.params['step']))
        btn_nz.clicked.connect(lambda: self._shift_pos(dz=-self.params['step']))
        btn_pos_reset.clicked.connect(self._reset_pos)
        btn_rot_l.clicked.connect(lambda: self._shift_rot(-90.0))
        btn_rot_r.clicked.connect(lambda: self._shift_rot(90.0))
        btn_rot_reset.clicked.connect(self._reset_rot)

        self.lbl_cam_pos_xyz = QtWidgets.QLabel()
        self.lbl_cam_pos_rot = QtWidgets.QLabel()
        self._update_labels()

        g.addWidget(self.spin_pos_step, 0, 0, 1, 3)
        g.addWidget(btn_nx, 1, 0); g.addWidget(btn_px, 1, 2)
        g.addWidget(btn_ny, 2, 0); g.addWidget(btn_py, 2, 2)
        g.addWidget(btn_nz, 3, 0); g.addWidget(btn_pz, 3, 2)
        g.addWidget(btn_rot_l, 4, 0); g.addWidget(btn_rot_r, 4, 2) 
        g.addWidget(self.lbl_cam_pos_xyz, 5, 0, 1, 3)
        g.addWidget(self.lbl_cam_pos_rot, 6, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("<b>참고:</b> 'OK'를 누르면 센서가 다시 스폰됩니다."), 8, 0, 1, 3)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept); btn_box.rejected.connect(self.reject)
        g.addWidget(btn_box, 9, 0, 1, 3)

    def _shift_pos(self, dx=0, dy=0, dz=0):
        self.params['x'] = self.params.get('x', 0.0) + dx
        self.params['y'] = self.params.get('y', 0.0) + dy
        self.params['z'] = self.params.get('z', 0.0) + dz
        self._update_labels()
    def _reset_pos(self):
        self.params['x'] = 0.0; self.params['y'] = 0.0; self.params['z'] = 0.0; self._update_labels()
    def _shift_rot(self, dyaw):
        self.params['yaw'] = (self.params.get('yaw', 0.0) + dyaw) % 360.0; self._update_labels()
    def _reset_rot(self):
        self.params['yaw'] = 0.0; self._update_labels()
    def _update_labels(self):
        self.lbl_cam_pos_xyz.setText(f"Offset: X={self.params.get('x', 0.0):.1f}, Y={self.params.get('y', 0.0):.1f}, Z={self.params.get('z', 0.0):.1f}")
        self.lbl_cam_pos_rot.setText(f"Rotation: Yaw={self.params.get('yaw', 0.0):.1f}°")
    def get_params(self): return self.params

class BevSettingsDialog(QtWidgets.QDialog):
    """[4] 포인트 클라우드 뷰 설정 다이얼로그 (구 RadarViewAdjust)"""
    paramsChanged = QtCore.Signal(dict)
    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("4. 포인트 클라우드 뷰 설정")
        self.setModal(False) # 모달리스(Modaless)로 설정
        self.resize(400, 250) # 창 크기 키움
        
        self.params = params.copy() if params else {}

        g = QtWidgets.QGridLayout(self)
        btn_up = QtWidgets.QPushButton("⬆"); btn_down = QtWidgets.QPushButton("⬇")
        btn_left = QtWidgets.QPushButton("⬅"); btn_right = QtWidgets.QPushButton("➡")
        btn_reset_offset = QtWidgets.QPushButton("오프셋 리셋")
        
        btn_up.clicked.connect(lambda: self._shift_bev(0, -10))
        btn_down.clicked.connect(lambda: self._shift_bev(0, +10))
        btn_left.clicked.connect(lambda: self._shift_bev(-10, 0))
        btn_right.clicked.connect(lambda: self._shift_bev(+10, 0))
        btn_reset_offset.clicked.connect(lambda: self._set_bev_offset(0,0))
        
        g.addWidget(btn_up, 0, 1); g.addWidget(btn_left, 1, 0)
        g.addWidget(btn_right, 1, 2); g.addWidget(btn_down, 2, 1)
        g.addWidget(btn_reset_offset, 3, 0, 1, 3)

        sld_radar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sld_radar.setRange(1, 6); sld_radar.setValue(self.params.get('point_size', 2))
        self.lbl_radar_size = QtWidgets.QLabel(f"점 크기: {self.params.get('point_size', 2)}")
        sld_radar.valueChanged.connect(self._on_radar_size_changed)

        g.addWidget(sld_radar, 4, 0, 1, 2); g.addWidget(self.lbl_radar_size, 4, 2)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        g.addWidget(btn_box, 5, 0, 1, 3)
        
    def _emit_changes(self):
        self.paramsChanged.emit(self.params)
    def _shift_bev(self, dx: int, dy: int):
        self.params['offset_x'] = self.params.get('offset_x', 0) + dx
        self.params['offset_y'] = self.params.get('offset_y', 0) + dy
        self._emit_changes()
    def _set_bev_offset(self, x: int, y: int):
        self.params['offset_x'], self.params['offset_y'] = x, y; self._emit_changes()
    def _on_radar_size_changed(self, val: int):
        self.params['point_size'] = val
        self.lbl_radar_size.setText(f"점 크기: {val}"); self._emit_changes()

# ==============================================================================
# --- (4/4) 메인 윈도우 ---
# ==============================================================================

class MainWindow(QtWidgets.QMainWindow):
    # GUI 렌더링을 위한 시그널
    updateCamImage = QtCore.Signal(QtGui.QImage)
    updateRadImage = QtCore.Signal(QtGui.QImage)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- 백엔드 매니저 초기화 ---
        self.manager = carla_manager.CarlaManager()
        self.manual_control_process = None

        # --- GUI 상태 파라미터 ---
        self.param_sensor_params = { # 4. Radar Settings
            'range': 70.0, 'h_fov': 40.0, 'v_fov': 50.0, 'pps': 12000
        }
        self.param_sensor_pos = { # 5. Sensor Position
            'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'step': 0.5
        }
        self.param_bev_view = { # 4. BEV View
            'offset_x': 100, 'offset_y': 0, 'point_size': 2
        }
        self.param_lane_state = { # 6. Lane Settings
            'lane_on': True, 'in1': True, 'in2': True, 'out1': True, 'out2': True
        }
        self.param_overlay_radar_on = True # 4. Radar Overlay
        self.param_show_sensor_debug_box = False # 4. Sensor Debug Box
        
        self.last_cam_frame_bgr: Optional[np.ndarray] = None # 차선 편집기용
        self._last_cam_qimg: Optional[QtGui.QImage] = None
        self._last_rad_qimg: Optional[QtGui.QImage] = None
        
        # --- UI 빌드 ---
        self.setWindowTitle("CARLA Sensor GUI")
        self.resize(1600, 900)

        self.central = QtWidgets.QWidget(); self.setCentralWidget(self.central)
        self.hbox = QtWidgets.QHBoxLayout(self.central) # [ (Left Panel) | (Right Views) ]

        # --- [9] 로그 다이얼로그 생성 ---
        self.log_dialog = LogDialog(self)
        
        self._build_left_controls() # 왼쪽 1열 컨트롤 패널
        self._build_right_views()   # 오른쪽 뷰어 (CAM, RAD)
        
        # --- 타이머 설정 ---
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)  # 20 FPS (50ms)
        self.timer.timeout.connect(self.on_tick)
        
        # [7] 수동 주행 종료 감지용 타이머
        self.manual_control_timer = QtCore.QTimer(self)
        self.manual_control_timer.setInterval(1000) 
        self.manual_control_timer.timeout.connect(self.on_check_manual_control)

    def log_message(self, message: str):
        """중앙 로그 함수. 로그 다이얼로그에 메시지 전송"""
        self.log_dialog.log_message(message)

    # ---------- (LEFT) 1열 컨트롤 패널 빌드 ----------
    def _build_left_controls(self):
        v_main = QtWidgets.QVBoxLayout() # 메인 수직 1열
        v_main.setSpacing(10)

        # === 1. Simulation Control ===
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
        self.btn_exit.clicked.connect(self.close) # self.close()가 closeEvent 트리거

        # === 2. Spawners ===
        grp2 = QtWidgets.QGroupBox("2. Spawners")
        f2 = QtWidgets.QFormLayout(grp2)
        self.spin_veh = QtWidgets.QSpinBox(); self.spin_veh.setRange(0, 300); self.spin_veh.setValue(100)
        self.btn_spawn_veh = QtWidgets.QPushButton("차량 스폰")
        self.btn_reset_veh = QtWidgets.QPushButton("리셋")
        self.btn_spawn_sensors = QtWidgets.QPushButton("센서 스폰")
        
        # 스폰/리셋 버튼을 가로로 묶는 레이아웃
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

        # === 3. Weather ===
        grp3 = QtWidgets.QGroupBox("3. Weather")
        h3 = QtWidgets.QHBoxLayout(grp3)
        self.btn_weather = QtWidgets.QPushButton("날씨 조절...")
        h3.addWidget(self.btn_weather)
        v_main.addWidget(grp3)
        
        self.btn_weather.clicked.connect(self.on_open_weather)
        
        # === 4. Sensor Parameters ===
        grp4 = QtWidgets.QGroupBox("4. Sensor Parameters")
        v4 = QtWidgets.QVBoxLayout(grp4)
        self.chk_overlay = QtWidgets.QCheckBox("카메라에 레이더 오버레이")
        self.chk_debug_box = QtWidgets.QCheckBox("시뮬레이션에 센서 위치 표시")
        self.btn_open_radar_settings = QtWidgets.QPushButton("레이더 설정 (Range, FoV, PPS)...")
        self.btn_open_bev_settings = QtWidgets.QPushButton("포인트 클라우드 뷰 (BEV)...")
        
        self.chk_overlay.setChecked(self.param_overlay_radar_on)
        self.chk_debug_box.setChecked(self.param_show_sensor_debug_box)
        
        v4.addWidget(self.chk_overlay); v4.addWidget(self.chk_debug_box)
        v4.addWidget(self.btn_open_radar_settings); v4.addWidget(self.btn_open_bev_settings)
        v_main.addWidget(grp4)
        
        self.chk_overlay.toggled.connect(lambda c: setattr(self, 'param_overlay_radar_on', c))
        self.chk_debug_box.toggled.connect(lambda c: setattr(self, 'param_show_sensor_debug_box', c))
        self.btn_open_radar_settings.clicked.connect(self.on_open_radar_settings)
        self.btn_open_bev_settings.clicked.connect(self.on_open_bev_settings)

        # === 5. Sensor Offset ===
        grp5 = QtWidgets.QGroupBox("5. Sensor Offset")
        h5 = QtWidgets.QHBoxLayout(grp5)
        self.btn_open_sensor_pos = QtWidgets.QPushButton("센서 위치 설정 (X, Y, Z, Yaw)...")
        h5.addWidget(self.btn_open_sensor_pos)
        v_main.addWidget(grp5)
        
        self.btn_open_sensor_pos.clicked.connect(self.on_open_sensor_pos)

        # === 6. Lane ===
        grp6 = QtWidgets.QGroupBox("6. Lane")
        v6 = QtWidgets.QVBoxLayout(grp6)
        g6 = QtWidgets.QGridLayout()
        self.chk_in1 = QtWidgets.QCheckBox("IN1"); self.chk_in2 = QtWidgets.QCheckBox("IN2")
        self.chk_out1 = QtWidgets.QCheckBox("OUT1"); self.chk_out2 = QtWidgets.QCheckBox("OUT2")
        self.chk_in1.setChecked(self.param_lane_state['in1'])
        self.chk_in2.setChecked(self.param_lane_state['in2'])
        self.chk_out1.setChecked(self.param_lane_state['out1'])
        self.chk_out2.setChecked(self.param_lane_state['out2'])
        g6.addWidget(self.chk_in1, 0, 0); g6.addWidget(self.chk_in2, 0, 1)
        g6.addWidget(self.chk_out1, 1, 0); g6.addWidget(self.chk_out2, 1, 1)
        v6.addLayout(g6)
        
        self.btn_edit_lanes = QtWidgets.QPushButton("폴리곤 편집...")
        v6.addWidget(self.btn_edit_lanes)
        v_main.addWidget(grp6)

        self.chk_in1.toggled.connect(lambda c: self.param_lane_state.update(in1=c))
        self.chk_in2.toggled.connect(lambda c: self.param_lane_state.update(in2=c))
        self.chk_out1.toggled.connect(lambda c: self.param_lane_state.update(out1=c))
        self.chk_out2.toggled.connect(lambda c: self.param_lane_state.update(out2=c))
        self.btn_edit_lanes.clicked.connect(self.on_edit_lanes)
        
        # === 7. Manual Control ===
        grp7 = QtWidgets.QGroupBox("7. Manual Control")
        h7 = QtWidgets.QHBoxLayout(grp7)
        self.btn_manual_control = QtWidgets.QPushButton("수동 주행 시작")
        h7.addWidget(self.btn_manual_control)
        v_main.addWidget(grp7)
        
        self.btn_manual_control.clicked.connect(self.on_manual_control)

        # === 8. Data ===
        grp8 = QtWidgets.QGroupBox("8. Data (ID | v_calc | v_carla)")
        h8 = QtWidgets.QHBoxLayout(grp8)
        self.btn_save = QtWidgets.QPushButton("보이는 차량 정보 저장")
        self.lbl_saved = QtWidgets.QLabel("")
        h8.addWidget(self.btn_save); h8.addWidget(self.lbl_saved, 1)
        v_main.addWidget(grp8)
        
        self.btn_save.clicked.connect(self.on_save_visible)
        
        # === 9. Terminal Log ===
        grp9 = QtWidgets.QGroupBox("9. Terminal Log")
        h9 = QtWidgets.QHBoxLayout(grp9)
        self.btn_open_log = QtWidgets.QPushButton("터미널 로그 보기...")
        h9.addWidget(self.btn_open_log)
        v_main.addWidget(grp9)

        self.btn_open_log.clicked.connect(self.on_open_log)

        v_main.addStretch(1) # 하단 여백
        self.hbox.addLayout(v_main, 1) # 왼쪽 패널 (비율 1)

    # ---------- (RIGHT) 뷰어 패널 빌드 ----------
    def _build_right_views(self):
        v = QtWidgets.QVBoxLayout()

        grp_cam = QtWidgets.QGroupBox("Camera View — BBox + Radar Overlay")
        self.view_cam = SensorImageView()
        lay_cam = QtWidgets.QVBoxLayout(grp_cam); lay_cam.addWidget(self.view_cam)
        v.addWidget(grp_cam, 2) # 카메라 뷰 (비율 2)

        grp_rad = QtWidgets.QGroupBox("4D Radar Point Cloud (BEV)")
        self.view_rad = SensorImageView()
        lay_rad = QtWidgets.QVBoxLayout(grp_rad); lay_rad.addWidget(self.view_rad)
        v.addWidget(grp_rad, 1) # 레이더 뷰 (비율 1)
        
        # 시그널 슬롯 연결
        self.updateCamImage.connect(self.view_cam.update_image)
        self.updateRadImage.connect(self.view_rad.update_image)

        self.hbox.addLayout(v, 3) # 오른쪽 뷰어 (비율 3)

    # ---------- (SLOTS) 버튼/액션 콜백 함수 ----------

    def on_run(self):
        try:
            self.manager.connect(start_server=True)
            self.log_message("[OK] Connected. '센서 스폰'을 눌러주세요.")
            self.timer.start() # 메인 틱 타이머 시작
            self.manual_control_timer.start() # 수동 주행 감지 타이머 시작
        except Exception as e:
            self.log_message(f"[ERR] connect failed: {e}")

    def on_toggle_pause(self, checked: bool):
        self.manager.toggle_pause()
        self.btn_pause.setText("▶ 재개" if checked else "일시정지")
        self.log_message("[SIM] 일시정지" if checked else "[SIM] 재개")

    def on_step_once(self):
        if self.manager.step_once():
            self.log_message("[SIM] 한 프레임 진행")
            # 틱이 발생했으므로 뷰를 강제로 1회 갱신
            self._render_and_update_views()
        else:
            self.log_message("[WARN] '일시정지' 상태에서만 한 프레임 진행이 가능합니다.")

    def on_spawn_sensors(self):
        """[중요] 모든 파라미터를 취합하여 센서 스폰을 요청하는 중앙 함수"""
        if not (self.manager and self.manager.sensor_manager):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다. '실행'을 먼저 눌러주세요.")
            return
        try:
            self.log_message(f"[SIM] Spawning sensors with params...")
            self.log_message(f"  > Radar: {self.param_sensor_params}")
            self.log_message(f"  > Pos: {self.param_sensor_pos}")
            
            # 1. 센서 스폰
            self.manager.sensor_manager.spawn_sensors(
                self.param_sensor_params,
                self.param_sensor_pos
            )
            # 2. 스펙테이터 뷰 이동
            self.manager.move_spectator_to_sensor_view(
                pos_z_offset=carla_manager.Z_OFFSET_VIEW,
                pos_params=self.param_sensor_pos
            )
            self.log_message("[OK] Sensors spawned and spectator view updated.")
        except Exception as e:
            self.log_message(f"[ERR] on_spawn_sensors: {e}")

    def on_spawn_vehicles(self):
        """
        '차량 스폰' 버튼 액션: 외부 명령어(generate_traffic.py)를 실행하여 생성만 합니다.
        """
        if not (self.manager and self.manager.vehicle_manager and self.manager.world):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다. '실행'을 먼저 눌러주세요.")
            return

        n = int(self.spin_veh.value())
        tm_port = 8000
        
        try:
            script_path = os.path.join(os.path.dirname(__file__), "generate_traffic.py")
            cmd = [
                sys.executable, script_path, 
                "-n", str(n),
                "--safe",
                "--tm-port", str(tm_port)
            ]
            subprocess.Popen(cmd)
            
            self.log_message(f"[OK] External command executed: Spawning {n} vehicles with --safe.")
            self.log_message(f"[INFO] Command: {' '.join(cmd)}")
            
        except Exception as e:
            self.log_message(f"[ERR] on_spawn_vehicles: Failed to execute command: {e}")

    def on_reset_vehicles(self):
        """
        '리셋' 버튼 액션: 내부 로직으로 현재 월드의 모든 차량을 제거합니다.
        """
        if not (self.manager and self.manager.vehicle_manager and self.manager.world):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다. '실행'을 먼저 눌러주세요.")
            return
        
        try:
            removed = self.manager.vehicle_manager.reset_all_vehicles()
            self.manager.world.tick() 
            self.log_message(f"[OK] Vehicle Reset: Removed {removed} vehicles successfully.")
        except Exception as e:
            self.log_message(f"[ERR] on_reset_vehicles: Destruction failed: {e}")

    def on_open_weather(self):
        if not (self.manager and self.manager.weather_manager):
            self.log_message("[ERR] Manager가 초기화되지 않았습니다."); 
            return
        dlg = WeatherControlDialog(self.manager, self)
        dlg.exec()
    
    def on_open_radar_settings(self):
        dlg = RadarSettingsDialog(self, self.param_sensor_params)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.param_sensor_params = dlg.get_params()
            self.log_message(f"[GUI] 레이더 설정 저장됨. (자동 센서 스폰 실행)")
            self.on_spawn_sensors()

    def on_open_sensor_pos(self):
        dlg = SensorPositionDialog(self, self.param_sensor_pos)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.param_sensor_pos = dlg.get_params()
            self.log_message(f"[GUI] 센서 위치 저장됨. (자동 센서 스폰 실행)")
            self.on_spawn_sensors()

    @QtCore.Slot(dict)
    def on_bev_settings_changed(self, params):
        self.param_bev_view = params
        self.log_message(f"[BEV] offset:({params['offset_x']},{params['offset_y']}), size:{params['point_size']}")

    def on_open_bev_settings(self):
        dlg = BevSettingsDialog(self, self.param_bev_view)
        dlg.paramsChanged.connect(self.on_bev_settings_changed)
        dlg.show() # 모달리스로 열기

    def on_edit_lanes(self):
        if self.last_cam_frame_bgr is None:
            self.log_message("[ERR] 카메라 프레임이 없습니다. 센서 스폰 후 시도하세요.")
            return
        
        current_polys = lane_utils.get_lane_polys()
        dlg = LaneEditorDialog(self.last_cam_frame_bgr.copy(), current_polys, self)
        
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            updated = dlg.get_polys()
            lane_utils.set_lane_polys(updated) # set_lane_polys가 내부적으로 save 호출
            self.log_message("[OK] 차선 폴리곤 저장됨.")

    def on_manual_control(self):
        if not self.manager:
            self.log_message("[ERR] Manager가 초기화되지 않았습니다.")
            return
                
        success, message = self.manager.start_manual_control()
        self.log_message(f"[Manual] {message}")
        if success:
            # QTimer 는 그대로 둔 채,
            # manager.start_manual_control() 안에서 is_paused = True 로 설정하여
            # 우리 쪽에서 world.tick() 만 잠시 멈추게 한다.
            self.btn_pause.setChecked(True)
            self.log_message(
                "[SIM] Manual Control 모드 진입: "
                "월드 틱은 manual_control.py 가 담당, "
                "GUI 는 계속 센서 버퍼를 렌더링합니다."
            )
            
    @QtCore.Slot()
    def on_check_manual_control(self):
        if not self.manager:
            return
        
        if self.manager.check_manual_control_finished():
            self.log_message("[Manual] 수동 주행 프로세스 종료 감지.")
            self.log_message(
                "[SIM] Manual Control 종료: "
                "다시 GUI 쪽에서 world.tick() 을 담당합니다."
            )
            self.btn_pause.setChecked(False)

    def on_save_visible(self):
        if not self.manager:
            self.log_message("[ERR] Manager가 초기화되지 않았습니다.")
            return
            
        success, message = self.manager.save_visible_vehicle_data()
        self.log_message(f"[Data] {message}")
        if success and "저장 완료" in message:
            self.lbl_saved.setText(f"저장됨: {os.path.basename(message.split(': ')[-1])}")
        elif success:
            self.lbl_saved.setText(message)
        else:
            self.lbl_saved.setText("저장 실패")
            
    def on_open_log(self):
        """[9] 로그 다이얼로그 보이기"""
        self.log_dialog.show()
        self.log_dialog.raise_()
        self.log_dialog.activateWindow()

    # ---------- (TICK) 메인 틱 및 렌더링 ----------
    
    def on_tick(self):
        """
        QTimer에 의해 주기적으로 호출되는 메인 루프.
        - manager.tick() 은 월드 틱을 시도하지만, is_paused=True 이면 내부에서 아무 것도 안 함
        - manual_control 이 월드를 돌리고 있을 때도, 센서 버퍼를 읽어서 계속 렌더링만 한다.
        """
        if not self.manager:
            return

        # 1) 월드 틱 (일시정지 상태라면 manager.tick() 안에서 그냥 False 리턴만 하고 끝)
        self.manager.tick()

        # 2) 연결/센서 상태 확인
        if not (self.manager.is_connected and 
                self.manager.sensor_manager and 
                self.manager.sensor_manager.cam):
            return

        # 3) 카메라/레이더 뷰 렌더링 (월드 틱이 안 돌아도, 버퍼에 있는 최신 프레임으로 계속 그림)
        self._render_and_update_views()

        # 4) 디버그 박스 (월드 틱 돌 때만)
        if self.param_show_sensor_debug_box and not self.manager.is_paused:
            self.manager.draw_sensor_debug_shapes(self.param_sensor_pos)

    def _render_and_update_views(self):
        """
        현재 버퍼에 있는 데이터를 가져와 QImage로 변환하고 Signal을 emit.
        (일시정지 시에도 마지막 이미지를 계속 그림)
        """
        if not self.manager.sensor_manager:
            return

        # (A) Camera View 렌더링
        cam_data = self.manager.sensor_manager.CAMERA_IMG
        if cam_data is not None:
            # 1. 원본 복사 (차선 편집기용)
            self.last_cam_frame_bgr = cam_data.copy()
            
            # 2. 오버레이 적용 (rendering.py)
            img_overlay = overlay_camera(
                cam_data.copy(),
                manager=self.manager,
                lane_state=self.param_lane_state,
                bev_view_params=self.param_bev_view,
                overlay_radar_on=self.param_overlay_radar_on,
            )
            
            # 3. QImage 변환 및 캐시
            h, w, ch = img_overlay.shape
            qimg = QtGui.QImage(img_overlay.data, w, h, ch*w, QtGui.QImage.Format_BGR888)
            self._last_cam_qimg = qimg
            
        # (B) Radar BEV 렌더링 (rendering.py)
        rad_qimg = render_radar_bev(
            manager=self.manager,
            sensor_params=self.param_sensor_params,
            bev_view_params=self.param_bev_view,
        )
        if rad_qimg is not None:
            self._last_rad_qimg = rad_qimg
            
        # (C) QImage Emit (캐시된 이미지 사용)
        if self._last_cam_qimg:
            self.updateCamImage.emit(self._last_cam_qimg)
        if self._last_rad_qimg:
            self.updateRadImage.emit(self._last_rad_qimg)

    # ---------- (CLOSE) 종료 이벤트 ----------
    
    def closeEvent(self, e: QtGui.QCloseEvent):
        """
        메인 윈도우가 닫힐 때 모든 것을 정리합니다.
        """
        self.log_message("[GUI] Closing application...")
        self.timer.stop()
        self.manual_control_timer.stop()
        try:
            if self.manager:
                self.manager.cleanup()
            self.log_dialog.close()
        except Exception as err:
            self.log_message(f"[ERR] Cleanup failed: {err}")
        finally:
            self.log_message("[GUI] Exit.")
            e.accept()

# ===================== main =====================
def main():
    app = QtWidgets.QApplication(sys.argv)

    # CWD 강제 변경 제거 (필요하면 여기서만 별도 처리)
    try:
        # script_dir = os.path.dirname(os.path.realpath(__file__))
        # os.chdir(script_dir)
        pass
    except Exception as e:
        print(f"Failed to change CWD: {e}")

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
