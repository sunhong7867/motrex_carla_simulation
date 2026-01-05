#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    import perception_utils
except ImportError:
    perception_utils = None

try:
    import cv2
except Exception as e:
    raise ImportError("calibration_manager requires opencv-python (cv2).") from e

# ==============================================================================
# Global Buffer (데이터 누적용)
# ==============================================================================
_BUFFER_3D = []  # Radar Points (Cumulative)
_BUFFER_2D = []  # Image Points (Cumulative)

def reset_calibration_buffer():
    """누적된 데이터를 초기화합니다."""
    global _BUFFER_3D, _BUFFER_2D
    _BUFFER_3D = []
    _BUFFER_2D = []
    print("[CALIB] Buffer cleared.")

@dataclass
class CalibrationResult:
    R: np.ndarray        # (3,3)
    t: np.ndarray        # (3,1)
    reproj_error_px: float
    inliers: np.ndarray  # (M,1) indices

def _reprojection_error(
    obj_pts: np.ndarray, img_pts: np.ndarray,
    K: np.ndarray, D: Optional[np.ndarray],
    rvec: np.ndarray, tvec: np.ndarray
) -> float:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1)
    return float(np.mean(err))

# ==============================================================================
# [NEW] CSV 저장 함수
# ==============================================================================
def save_calibration_log(result: CalibrationResult, total_pts: int, pts_3d: np.ndarray, pts_2d: np.ndarray):
    """
    캘리브레이션 성공 시 결과(History)와 상세 포인트 데이터(Points)를 CSV로 저장합니다.
    """
    base_dir = "calibration_logs"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. 히스토리 요약 저장 (누적)
    history_path = os.path.join(base_dir, "calibration_history.csv")
    file_exists = os.path.isfile(history_path)
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tx, ty, tz = result.t.flatten()
    
    with open(history_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 헤더가 없으면 생성
        if not file_exists:
            writer.writerow([
                "Timestamp", "Total_Samples", "Inliers_Count", 
                "Reproj_Error_px", "Tx_m", "Ty_m", "Tz_m"
            ])
        
        # 데이터 추가
        writer.writerow([
            now_str, total_pts, len(result.inliers), 
            f"{result.reproj_error_px:.4f}", 
            f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}"
        ])
    
    # 2. (옵션) 이번 계산에 쓰인 상세 포인트 저장 (디버깅/분석용)
    # 파일명: points_YYYYMMDD_HHMMSS.csv
    ts_safe = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    points_path = os.path.join(base_dir, f"points_{ts_safe}.csv")
    
    # Inlier 여부 표시를 위해
    is_inlier = np.zeros(total_pts, dtype=bool)
    if result.inliers is not None:
        is_inlier[result.inliers.flatten()] = True
        
    try:
        df_data = []
        for i in range(total_pts):
            row = {
                "id": i,
                "rad_x": pts_3d[i, 0], "rad_y": pts_3d[i, 1], "rad_z": pts_3d[i, 2],
                "img_u": pts_2d[i, 0], "img_v": pts_2d[i, 1],
                "is_inlier": int(is_inlier[i])
            }
            df_data.append(row)
            
        import pandas as pd
        pd.DataFrame(df_data).to_csv(points_path, index=False)
        # print(f"[CALIB] Detailed points saved to {points_path}")
    except ImportError:
        pass # pandas 없으면 생략

    print(f"[CALIB] Log saved: {history_path}")


def calibrate_radar_to_camera_pnp(
    radar_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
    K: np.ndarray,
    D: Optional[np.ndarray] = None,
    ransac_reproj_threshold: float = 10.0,
    ransac_confidence: float = 0.99,
    ransac_iters: int = 10000,
    refine: bool = True
) -> CalibrationResult:
    """Core PnP Logic - Iterative Solver"""
    
    if radar_points_3d.shape[0] < 6:
        raise ValueError(f"Need more points. (Current: {radar_points_3d.shape[0]}, Need >= 6)")

    obj = radar_points_3d.astype(np.float64).reshape(-1, 1, 3)
    img = image_points_2d.astype(np.float64).reshape(-1, 1, 2)

    # 초기값 강제 (CARLA Radar -> Camera)
    R_guess = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ], dtype=np.float64)
    rvec_guess, _ = cv2.Rodrigues(R_guess)
    tvec_guess = np.zeros((3, 1), dtype=np.float64)

    # 1. SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=D,
        rvec=rvec_guess,
        tvec=tvec_guess,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        raise RuntimeError("PnP Iterative Solver failed.")

    # 2. 검증 (3.0m 제한)
    dist_norm = np.linalg.norm(tvec)
    if dist_norm > 3.0: 
        raise RuntimeError(f"Unrealistic result: dist {dist_norm:.2f}m is too large. (Check Axis Alignment)")

    # 3. 에러 계산
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err_vec = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    
    inliers_idx = np.where(err_vec < ransac_reproj_threshold)[0]
    avg_err = float(np.mean(err_vec))

    R, _ = cv2.Rodrigues(rvec)
    
    return CalibrationResult(
        R=R, 
        t=tvec.reshape(3, 1), 
        reproj_error_px=avg_err, 
        inliers=inliers_idx.reshape(-1, 1)
    )


def save_extrinsic_json(path: str, R: np.ndarray, t: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
    import json
    payload = {
        "R": R.tolist(),
        "t": t.reshape(3).tolist()
    }
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_extrinsic_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    R = np.array(payload["R"], dtype=np.float64)
    t = np.array(payload["t"], dtype=np.float64).reshape(3, 1)
    return R, t


def run_calibration_pipeline(
    detections: List[Dict],
    radar_frame: List[Tuple],
    K: np.ndarray,
    width: int,
    height: int,
    output_path: str = "extrinsic.json"
) -> Tuple[bool, str]:
    """
    데이터 누적 후 PnP 수행 및 CSV 로깅 추가
    """
    global _BUFFER_3D, _BUFFER_2D
    
    if perception_utils is None:
        return False, "perception_utils missing."

    # 1. 초기 투영용 (필터링)
    R_init = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    t_init = np.zeros((3, 1), dtype=np.float64)
    extr_init = perception_utils.ExtrinsicRT(R=R_init, t=t_init)
    cam_model = perception_utils.CameraModel(K=K, D=None)

    new_pts_count = 0
    margin = 30  # 박스 마진

    # 2. 현재 프레임 데이터 추출
    for det in detections:
        if "bbox" not in det: continue
        x1, y1, x2, y2 = det["bbox"]
        box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        points_in_box = []
        for p in radar_frame:
            if p is None or len(p) < 3: continue
            rx, ry, rz = float(p[0]), float(p[1]), float(p[2])
            
            if rx < 2.0 or rx > 80.0: continue
            
            xyz = np.array([[rx, ry, rz]], dtype=np.float64)
            uv, _ = perception_utils.project_radar_to_image(
                xyz, cam_model, extr_init, width, height, use_distortion=False
            )
            if len(uv) > 0:
                u, v = uv[0]
                if (x1 - margin <= u <= x2 + margin) and (y1 - margin <= v <= y2 + margin):
                    points_in_box.append([rx, ry, rz])
        
        if len(points_in_box) > 0:
            pts_arr = np.array(points_in_box)
            centroid = np.mean(pts_arr, axis=0) 
            
            _BUFFER_3D.append(centroid)
            _BUFFER_2D.append([box_cx, box_cy])
            new_pts_count += 1

    total_pts = len(_BUFFER_3D)
    
    # 3. 데이터 부족 시
    if total_pts < 8:
        return False, f"Collecting data... (+{new_pts_count})\nTotal: {total_pts}/8 needed.\nClick 'Calibrate' again with different vehicles."

    # 4. PnP 수행
    try:
        pts_3d = np.array(_BUFFER_3D, dtype=np.float64)
        pts_2d = np.array(_BUFFER_2D, dtype=np.float64)
        
        result = calibrate_radar_to_camera_pnp(
            radar_points_3d=pts_3d,
            image_points_2d=pts_2d,
            K=K,
            D=None
        )
        
        # 5. 결과 저장 (JSON)
        save_extrinsic_json(
            path=output_path,
            R=result.R,
            t=result.t,
            meta={
                "method": "PnP_Accumulated_Iterative",
                "total_samples": total_pts,
                "inliers": int(len(result.inliers)),
                "reproj_error": float(result.reproj_error_px)
            }
        )

        # 6. [NEW] 결과 로깅 (CSV)
        save_calibration_log(result, total_pts, pts_3d, pts_2d)
        
        # 버퍼 초기화
        reset_calibration_buffer()
        
        msg = (f"CALIBRATION SUCCESS!\n"
               f"Samples: {len(result.inliers)}/{total_pts}\n"
               f"Error: {result.reproj_error_px:.2f}px\n"
               f"Trans: {result.t.flatten().round(2)}")
        return True, msg

    except RuntimeError as re:
        return False, f"Optimization Failed ({re}).\nCurrent pts: {total_pts}. Keep clicking to add more diverse data."
    except Exception as e:
        return False, f"Calc Failed: {e}"