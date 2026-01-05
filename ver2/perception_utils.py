#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import numpy as np
import math
from typing import Optional, Tuple, Deque, List

def get_camera_intrinsic(cam_actor: carla.Actor, w: int, h: int) -> np.ndarray:
    """
    CARLA 카메라 액터로부터 3x3 K 매트릭스를 계산합니다.
    """
    fov = float(cam_actor.attributes['fov'])
    f = w / (2.0 * math.tan((fov * math.pi / 180.0) / 2.0))
    return np.array([[f, 0, w/2.0],
                     [0, f, h/2.0],
                     [0, 0, 1.0]], dtype=np.float64)

def project_world_to_image(cam_actor: carla.Actor,
                           loc: carla.Location,
                           w: int, h: int) -> Optional[Tuple[int, int]]:
    """
    월드 좌표(loc)를 카메라 이미지 좌표(u, v)로 투영합니다.
    """
    K = get_camera_intrinsic(cam_actor, w, h)
    world_to_sensor = np.array(cam_actor.get_transform().get_inverse_matrix(), dtype=np.float64)
    p_world = np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float64)
    
    # World -> Sensor
    p_sensor = world_to_sensor @ p_world
    
    # Sensor -> Camera (UE4 standard)
    x = p_sensor[1]   # y
    y = -p_sensor[2]  # -z
    z = p_sensor[0]   # x (forward)
    
    if z <= 0.0:
        return None
        
    # Camera -> Pixel
    pix = K @ np.array([x, y, z], dtype=np.float64)
    u = int(pix[0] / pix[2])
    v = int(pix[1] / pix[2])
    
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None

def get_2d_bbox(cam_actor: carla.Actor, target_actor: carla.Actor,
                w: int, h: int) -> Optional[Tuple[int,int,int,int,float]]:
    """
    타겟 액터의 3D Bounding Box를 2D 이미지 좌표로 투영합니다.
    (2륜 차량의 BBox 크기를 보정하는 로직 포함)
    """
    K = get_camera_intrinsic(cam_actor, w, h)
    world_to_sensor = np.array(cam_actor.get_transform().get_inverse_matrix(), dtype=np.float64)
    
    bbox = target_actor.bounding_box
    tf = target_actor.get_transform()
    ext = bbox.extent 
    
    # 2륜차 BBox 크기 보정
    type_id_str = str(target_actor.type_id).lower()
    is_2_wheel = 'bicycle' in type_id_str or 'motorcycle' in type_id_str or 'harley' in type_id_str
    
    if is_2_wheel:
        ext.x = max(ext.x, 0.8)  # 1.6m
        ext.y = max(ext.y, 0.25) # 0.5m
        ext.z = max(ext.z, 0.7)  # 1.4m
        
    center = bbox.location 
    verts = [
        carla.Location(center.x - ext.x, center.y - ext.y, center.z - ext.z),
        carla.Location(center.x + ext.x, center.y - ext.y, center.z - ext.z),
        carla.Location(center.x - ext.x, center.y + ext.y, center.z - ext.z),
        carla.Location(center.x + ext.x, center.y + ext.y, center.z - ext.z),
        carla.Location(center.x - ext.x, center.y - ext.y, center.z + ext.z),
        carla.Location(center.x + ext.x, center.y - ext.y, center.z + ext.z),
        carla.Location(center.x - ext.x, center.y + ext.y, center.z + ext.z),
        carla.Location(center.x + ext.x, center.y + ext.y, center.z + ext.z),
    ]

    xs, ys, zs = [], [], []
    
    for local_p in verts:
        P_world = tf.transform(local_p) # Local -> World
        p_world = np.array([P_world.x, P_world.y, P_world.z, 1.0], dtype=np.float64)
        
        # World -> Sensor
        p_sensor = world_to_sensor @ p_world
        
        # Sensor -> Camera (UE4 standard)
        x = p_sensor[1]; y = -p_sensor[2]; z = p_sensor[0]
        
        if z <= 0.0:
            continue
            
        # Camera -> Pixel
        pix = K @ np.array([x, y, z], dtype=np.float64)
        u = int(pix[0] / pix[2]); v = int(pix[1] / pix[2])
        
        if 0 <= u < w and 0 <= v < h:
            xs.append(u); ys.append(v); zs.append(z)
            
    if len(xs) < 2: # 최소 2개 이상의 점이 투영되어야 BBox 생성
        return None
        
    z_center = float(np.median(zs)) if zs else 1e9
    return (min(xs), min(ys), max(xs), max(ys), z_center)

def depth_to_meters(raw_rgba: np.ndarray) -> np.ndarray:
    """
    CARLA Depth 카메라(BGRA) 이미지를 거리(m)로 변환합니다.
    """
    bgra = raw_rgba.astype(np.float32) / 255.0
    b, g, r = bgra[:,:,0], bgra[:,:,1], bgra[:,:,2]
    depth = (r + g/256.0 + b/65536.0) * 1000.0
    return depth

def is_occluded(depth_img: Optional[np.ndarray], bxyxy: Tuple[int,int,int,int], z_cam_center: float, w: int, h: int) -> bool:
    """
    Depth 이미지를 사용해 BBox가 가려졌는지 판별합니다.
    """
    if depth_img is None:
        return False
        
    x1,y1,x2,y2 = bxyxy
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return True # BBox 크기가 없음
        
    xs = np.linspace(x1, x2, num=9).astype(int)
    ys = np.linspace(y1, y2, num=9).astype(int)
    occluded = 0; total = 0
    
    for u in xs:
        for v in ys:
            d = float(depth_img[v, u])
            # 뎁스맵의 거리(d)가 객체 중심까지의 거리(z_cam_center)보다
            # 0.3m 이상 가까우면, 가려진 것으로(occluded) 판단
            if d + 0.3 < z_cam_center:
                occluded += 1
            total += 1
            
    # 샘플링한 점의 60% 이상이 가려졌으면 True
    return (occluded / max(1,total)) >= 0.6

def estimate_speed_from_radar(
    cam_actor: carla.Actor, 
    rad_world_history: Deque[List[Tuple[carla.Location, float, float]]],
    bxyxy: Tuple[int,int,int,int], 
    z_cam_center: float,
    w: int, h: int
) -> float:
    """
    BBox 내부로 투영되는 레이더 포인트들의 속도를 집계하여 kph로 반환합니다.
    """
    if cam_actor is None or not rad_world_history:
        return float('nan')

    x1, y1, x2, y2 = bxyxy
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return float('nan')

    depth_tol = max(3.0, 0.15 * float(z_cam_center))
    cand = []

    # 최근 3프레임의 레이더 스캔 사용
    for frame in list(rad_world_history)[-3:]:
        for (wp, vel_ms, r_depth) in frame:
            uv = project_world_to_image(cam_actor, wp, w, h)
            if uv is None:
                continue
            u, v = uv
            
            # BBox 내부 + 거리 근사 일치 체크
            if (x1 <= u <= x2) and (y1 <= v <= y2) and \
               (abs(float(r_depth) - float(z_cam_center)) <= depth_tol):
                cand.append(abs(float(vel_ms))) # 도플러 속도의 절대값

    if len(cand) < 3: # 유효 포인트가 3개 미만이면 실패
        return float('nan')

    # Outlier 제거 (Median Absolute Deviation)
    med = float(np.median(cand))
    mad = float(np.median([abs(x - med) for x in cand])) + 1e-6
    inliers = [x for x in cand if abs(x - med) <= 2.5 * mad]
    
    if len(inliers) < 3:
        inliers = cand # Inlier가 너무 적으면 그냥 원본 사용

    v_ms = float(np.mean(inliers))
    v_kph = v_ms * 3.6
    return v_kph