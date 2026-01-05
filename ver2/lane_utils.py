#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import cv2
import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

LANE_CFG_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / "lane_polys.json"
LANE_NAMES = ["IN1", "IN2", "OUT1", "OUT2"]

# 전역으로 폴리곤 데이터를 관리
LANE_POLYS: Dict[str, Optional[np.ndarray]] = {name: None for name in LANE_NAMES}

def load_lane_polys():
    """
    json 파일에서 차선 폴리곤 데이터를 로드하여 LANE_POLYS 전역 변수를 업데이트합니다.
    """
    global LANE_POLYS
    try:
        if LANE_CFG_PATH.exists():
            with open(LANE_CFG_PATH, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if v is not None and k in LANE_POLYS:
                    LANE_POLYS[k] = np.array(v, dtype=np.int32)
            print(f"[lane] loaded {LANE_CFG_PATH}")
        else:
            print("[lane] lane_polys.json not found (skipping)")
    except Exception as e:
        print(f"[lane] load error: {e}")

def save_lane_polys():
    """
    현재 LANE_POLYS의 데이터를 json 파일로 저장합니다.
    """
    try:
        out = {}
        for k, v in LANE_POLYS.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else None
        with open(LANE_CFG_PATH, "w") as f:
            json.dump(out, f)
        print(f"[lane] saved to {LANE_CFG_PATH}")
    except Exception as e:
        print(f"[lane] save error: {e}")

def get_lane_polys() -> Dict[str, Optional[np.ndarray]]:
    """
    현재 로드된 폴리곤 데이터를 반환합니다.
    """
    return LANE_POLYS

def set_lane_polys(polys_dict: Dict[str, Optional[np.ndarray]]):
    """
    외부(GUI)에서 폴리곤 데이터를 받아 LANE_POLYS를 업데이트합니다.
    """
    global LANE_POLYS
    LANE_POLYS = polys_dict
    save_lane_polys() # 업데이트 시 즉시 저장

def is_in_selected_lanes(u: int, v: int, lane_state: Dict[str, bool]) -> bool:
    """
    (u, v) 좌표가 활성화된 차선 내부에 있는지 확인합니다.
    lane_state: {'lane_on': True, 'in1': True, 'in2': False, ...}
    """
    if not lane_state.get('lane_on', True):
        return True # 필터가 꺼져있으면 무조건 True

    selected_names = [name for name in LANE_NAMES if lane_state.get(name.lower(), False)]
    if not selected_names:
        return True # 선택된 차선이 없으면 무조건 True (필터링 안 함)

    pt = (u, v)
    for name in selected_names:
        poly = LANE_POLYS.get(name)
        if poly is None:
            continue
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return True # 하나라도 포함되면 True
            
    return False # 모든 선택된 차선에 포함되지 않으면 False
