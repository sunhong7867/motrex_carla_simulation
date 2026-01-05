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
# [핵심] sensor_manager의 오프셋 설정을 가져옵니다. (자동 동기화)
# ==============================================================================
from sensor_manager import RADAR_OFFSET_X, RADAR_OFFSET_Y, RADAR_OFFSET_Z
print(f"[CALIB] Loaded Sensor Offsets from sensor_manager: X={RADAR_OFFSET_X}, Y={RADAR_OFFSET_Y}, Z={RADAR_OFFSET_Z}")
# ==============================================================================


# ==============================================================================
# [설정] 캘리브레이션 필터 파라미터 (여기서 조절하세요!)
# ==============================================================================
MIN_CALIB_DIST = 23.0    # [수정] 너무 가까우면 오차가 크므로 15m 이상 권장
REQUIRED_SAMPLES = 10    # 최소 샘플 수
DUPLICATE_THR_PX = 50    # 픽셀 중복 체크 (조금 더 여유 있게 10px)
# ==============================================================================


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
    method: str = "Unknown" # 사용된 최적화 방식 기록

def _reprojection_error(
    obj_pts: np.ndarray, img_pts: np.ndarray,
    K: np.ndarray, D: Optional[np.ndarray],
    rvec: np.ndarray, tvec: np.ndarray
) -> float:
    """재투영 오차(RMS)를 계산하는 헬퍼 함수"""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1)
    return float(np.mean(err))


# ==============================================================================
# Z-Score 필터링 (논문 기반 이상치 제거)
# ==============================================================================
def _filter_outliers_zscore(points: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    Z-Score를 사용하여 이상치를 제거합니다.
    points: (N, 3) array
    threshold: 표준편차의 몇 배까지 허용할지 (보통 2.0)
    """
    if len(points) < 5:  # 점이 너무 적으면 통계적 의미가 없으므로 스킵
        return points
        
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    
    # 표준편차가 0인 경우(모든 점이 같은 위치) 방어 코드
    std[std == 0] = 1e-6
    
    # 각 축(x, y, z)별로 Z-score 계산: $Z = \frac{|x - \mu|}{\sigma}$
    z_scores = np.abs((points - mean) / std)
    
    # 모든 축(axis=1)에서 threshold 이내인 점만 선택 (AND 조건)
    valid_mask = (z_scores < threshold).all(axis=1)
    
    # 만약 필터링 후 점이 너무 적어지면(예: 3개 미만), 필터링을 완화하거나 원본 반환
    if np.sum(valid_mask) < 3:
        return points
        
    return points[valid_mask]


# ==============================================================================
# CSV 저장 함수
# ==============================================================================
def save_calibration_log(result: CalibrationResult, total_pts: int, pts_3d: np.ndarray, pts_2d: np.ndarray):
    """
    캘리브레이션 결과와 상세 포인트 데이터를 CSV로 저장합니다.
    """
    base_dir = "calibration_logs"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. 히스토리 요약 저장
    history_path = os.path.join(base_dir, "calibration_history.csv")
    file_exists = os.path.isfile(history_path)
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tx, ty, tz = result.t.flatten()

    if result.inliers is None:
        count_inliers = 0
    else:
        count_inliers = len(result.inliers)
    
    with open(history_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Method", "Total_Samples", "Inliers_Count", 
                "Reproj_Error_px", "Tx_m", "Ty_m", "Tz_m"
            ])
        
        writer.writerow([
            now_str, result.method, total_pts, count_inliers, 
            f"{result.reproj_error_px:.4f}", 
            f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}"
        ])
    
    # 2. 상세 포인트 저장
    ts_safe = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    points_path = os.path.join(base_dir, f"points_{ts_safe}.csv")
    
    is_inlier = np.zeros(total_pts, dtype=bool)
    if result.inliers is not None and len(result.inliers) > 0:
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
    except ImportError:
        pass 

    print(f"[CALIB] Log saved: {history_path} (Method: {result.method})")


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


# ==============================================================================
# 메인 파이프라인
# ==============================================================================
def run_calibration_pipeline(
    detections: List[Dict],
    radar_history: List[List[Tuple]], 
    K: np.ndarray,
    width: int,
    height: int,
    output_path: str = "extrinsic.json"
) -> Tuple[bool, str]:

    global _BUFFER_3D, _BUFFER_2D
    
    if perception_utils is None:
        return False, "perception_utils missing."

    # [1] 기본 축 정렬 (CARLA -> OpenCV)
    # 텍스트 흐름에 맞게 일반 텍스트 표현 '벡터 u', '실수 R' 등을 고려함
    R_align = np.array([
        [0.0, 1.0, 0.0],   # Rad Y -> Cam X
        [0.0, 0.0, -1.0],  # Rad Z -> Cam -Y
        [1.0, 0.0, 0.0]    # Rad X -> Cam Z
    ], dtype=np.float64)
    
    # --------------------------------------------------------------------------
    # [수정] 초기값 로드: 정답(오프셋)을 절대 참조하지 않고 파일 기록만 사용함
    # --------------------------------------------------------------------------
    # 파일이 없으면 [0, 0, 0], 있으면 마지막 성공 지점에서 시작함
    prev_R, prev_t = load_extrinsic_json(output_path)
    tvec_init = prev_t.astype(np.float64)
    rvec_init = np.zeros((3, 1), dtype=np.float64) 
    
    # 시각화 및 데이터 수집용 임시 모델 (이전 상태를 투영 지표로 사용)
    extr_init = perception_utils.ExtrinsicRT(R=R_align, t=tvec_init)
    cam_model = perception_utils.CameraModel(K=K, D=None)

    new_pts_count = 0
    margin = 30 

    if not detections:
        return False, "No detections found."
        
    # [2] 데이터 수집 (기존 필터링 로직 유지)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        
        # 타겟점을 범퍼 위치(하단 65% 지점)로 매칭하여 투영 정확도 향상
        box_h = y2 - y1
        box_cy = y1 + (box_h * 0.65) 
        box_cx = (x1 + x2) / 2
        
        # 중복 방지 (이미 버퍼에 저장된 2D 좌표와 근접하면 제외)
        is_duplicate = False
        for stored_uv in _BUFFER_2D:
            dist_sq = (stored_uv[0] - box_cx)**2 + (stored_uv[1] - box_cy)**2
            if dist_sq < DUPLICATE_THR_PX**2:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
        
        points_for_this_car = []

        for frame in radar_history:
            if not frame: continue
            for p in frame:
                if p and len(p) >= 4:
                    pt3d = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)
                    
                    # 현재 추정치(extr_init)를 기준으로 해당 객체 내 포인트 필터링
                    uv, _ = perception_utils.project_radar_to_image(
                        pt3d.reshape(1,3), cam_model, extr_init, width, height, use_distortion=False
                    )
                    
                    if len(uv) > 0:
                        u, v = uv[0]
                        if (x1 - margin <= u <= x2 + margin) and (y1 - margin <= v <= y2 + margin):
                            if pt3d[0] > 10.0:
                                points_for_this_car.append(pt3d)

        # 유효 포인트 클러스터링 및 대표점(Centroid) 추출
        if len(points_for_this_car) >= 5:
            pts_arr = np.array(points_for_this_car, dtype=np.float64)
            filtered_pts = _filter_outliers_zscore(pts_arr, threshold=2.0)
            
            if len(filtered_pts) >= 3:
                filtered_pts = filtered_pts[filtered_pts[:, 0].argsort()]
                cutoff_idx = max(3, int(len(filtered_pts) * 0.3))
                closest_cluster = filtered_pts[:cutoff_idx]
                
                centroid = np.median(closest_cluster, axis=0)
                
                if centroid[0] < MIN_CALIB_DIST:
                    continue
                
                _BUFFER_3D.append(centroid)
                _BUFFER_2D.append([box_cx, box_cy])
                new_pts_count += 1

    total_pts = len(_BUFFER_3D)
    
    if new_pts_count > 0:
        print(f"[CALIB] Added {new_pts_count} pts. Total: {total_pts}")
    
    if total_pts < REQUIRED_SAMPLES:
        return False, f"Collecting... {total_pts}/{REQUIRED_SAMPLES}"


    # --------------------------------------------------------------------------
    # [Step 3] 2-Stage 최적화: SQPnP(전역 탐색) + Iterative(미세 조정)
    # --------------------------------------------------------------------------
    # 원인: Iterative 방식은 초기값이 나쁘면 거울상(Mirror) 오답으로 빠짐
    # 해결: 초기값 없이도 정답을 찾는 SQPnP로 '큰 방향'을 먼저 잡고 다듬음
    try:
        pts_3d = np.array(_BUFFER_3D, dtype=np.float64)
        pts_2d = np.array(_BUFFER_2D, dtype=np.float64)
        
        pts_3d_aligned = (R_align @ pts_3d.T).T
        
        obj = pts_3d_aligned.reshape(-1, 1, 3)
        img = pts_2d.reshape(-1, 1, 2)
        
        best_rvec, best_tvec, best_err, best_method = None, None, float('inf'), "None"

        # 1단계: SQPnP (Global Solver) - 초기값에 의존하지 않고 수학적 해를 도출
        if hasattr(cv2, 'SOLVEPNP_SQPNP'):
            ret_sq, r_sq, t_sq = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if ret_sq:
                err_sq = _reprojection_error(obj, img, K, None, r_sq, t_sq)
                best_rvec, best_tvec, best_err, best_method = r_sq, t_sq, err_sq, "SQPnP"
        
        # 2단계: Iterative (Refinement) - 1단계 결과 혹은 JSON 초기값을 기반으로 정밀 보정
        init_r = best_rvec if best_rvec is not None else rvec_init
        init_t = best_tvec if best_tvec is not None else tvec_init
        
        success, r_final, t_final = cv2.solvePnP(
            obj, img, K, None,
            rvec=init_r, tvec=init_t,
            useExtrinsicGuess=True,             
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            final_err = _reprojection_error(obj, img, K, None, r_final, t_final)
            # 최종 에러가 개선되었을 때만 업데이트
            if final_err < best_err:
                best_rvec, best_tvec, best_err, best_method = r_final, t_final, final_err, best_method + "+Refine"

        if best_rvec is None:
             return False, "All Solver Failed."

        # 결과 변환 및 저장
        R_delta, _ = cv2.Rodrigues(best_rvec)
        R_final = R_delta @ R_align
        
        safe_inliers = np.arange(len(obj)).astype(int)
        
        result_obj = CalibrationResult(
            R=R_final, t=best_tvec, reproj_error_px=best_err, 
            inliers=safe_inliers, 
            method=best_method
        )
        
        save_extrinsic_json(output_path, R_final, best_tvec)
        save_calibration_log(result_obj, total_pts, pts_3d, pts_2d)
        
        reset_calibration_buffer()
        
        tx, ty, tz = best_tvec.flatten()
        return True, f"Success! T=[{tx:.2f}, {ty:.2f}, {tz:.2f}] (Err: {best_err:.2f}px)"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Optimization Failed: {e}"




