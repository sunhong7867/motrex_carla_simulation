#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import logging
import random  # <--- Python 기본 random 모듈 사용
from numpy import random as np_random # numpy random은 np_random으로 사용
from typing import List, Set, Tuple, Optional

# --- generate_traffic.py 로직 ---

def get_actor_blueprints(world: carla.World, filter_pattern: str, generation: str):
    """
    필터와 세대에 맞는 액터 블루프린트를 가져옵니다.
    """
    bps = world.get_blueprint_library().filter(filter_pattern)

    if generation.lower() == "all":
        return bps
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print(f"   Warning! Actor Generation '{generation}' is not valid. No actor will be spawned.")
            return []
    except:
        print(f"   Warning! Actor Generation '{generation}' is not valid. No actor will be spawned.")
        return []

# --- VehicleManager (통합) ---

class VehicleManager:
    def __init__(self, client: carla.Client, world: carla.World, tm_port: int = 8000):
        self.client = client
        self.world = world
        self.traffic_manager = client.get_trafficmanager(tm_port)
        self.tm_port = tm_port
        
        # 생성된 차량 ID를 추적
        self.vehicle_ids: Set[int] = set()

        # --- TM 안정화 설정 (generate_traffic.py 기반) ---
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(True)
        # ----------------------------------------------------

    def reset_all_vehicles(self) -> int:
        """
        현재 월드에 있는 *모든* 차량 액터를 파괴합니다.
        (센서 등 다른 액터는 건드리지 않습니다.)
        """
        print("[VehicleManager] Destroying all vehicles...")
        
        # [수정] carla.command.DestroyActor (소문자 c)를 사용합니다.
        DestroyActor = carla.command.DestroyActor
        
        # vehicle_ids에 등록된 액터만 파괴 (더 안전)
        if self.vehicle_ids:
            actors_to_destroy = list(self.vehicle_ids)
            cmds = [DestroyActor(actor_id) for actor_id in actors_to_destroy]
            self.client.apply_batch(cmds)
            self.vehicle_ids.clear()
            return len(cmds)
        
        # 만약 vehicle_ids가 비어있다면, 월드에서 직접 찾아 파괴 (Fallback)
        else:
            vehicles_in_world = self.world.get_actors().filter('vehicle.*')
            if not vehicles_in_world:
                print("[VehicleManager] No vehicles found to destroy.")
                return 0
                
            cmds = [DestroyActor(v.id) for v in vehicles_in_world]
            self.client.apply_batch(cmds)
            print(f"[{__name__}] Destroyed {len(cmds)} vehicles from world fallback.")
            return len(cmds)

    def spawn_vehicles(self, num_vehicles: int, 
                       filter_v: str = 'vehicle.*', gen_v: str = 'All', 
                       safe_spawning: bool = True) -> int:
        """
        [수정됨] world.try_spawn_actor (단일 스폰) 방식으로 안정성을 최우선합니다.
        """
        
        print(f"[VehicleManager] Spawning {num_vehicles} vehicles (single loop)...")
        
        blueprints = get_actor_blueprints(self.world, filter_v, gen_v)
        if safe_spawning:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        
        if num_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points) # Python 기본 random 사용
        elif num_vehicles > number_of_spawn_points:
            logging.warning(f"Requested {num_vehicles} vehicles, but only {number_of_spawn_points} spawn points found.")
            num_vehicles = number_of_spawn_points
            
        spawned_count = 0
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break
            
            blueprint = random.choice(blueprints)
            
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')

            # --- [핵심 수정] try_spawn_actor 사용 ---
            transform.location.z += 0.5 # 스폰 시 충돌 방지를 위해 약간 높임
            veh = self.world.try_spawn_actor(blueprint, transform)
            
            if veh:
                veh.set_autopilot(True, self.tm_port)
                self.vehicle_ids.add(veh.id)
                spawned_count += 1
                
        print(f"[VehicleManager] Successfully spawned {spawned_count} vehicles.")
        return spawned_count

    def reset_and_spawn_vehicles(self, num_vehicles: int) -> Tuple[int, int]:
        """
        [GUI용] 기존 차량을 모두 리셋하고, 새로 'num_vehicles'만큼 스폰합니다.
        """
        # 1. 기존 차량 파괴
        removed_count = self.reset_all_vehicles()
        
        # 2. 동기 모드 tick (파괴 적용)
        try:
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
                self.world.tick() # [수정] 파괴 적용을 위한 1프레임 대기
        except Exception as e:
            print(f"[VehicleManager] Warning: world.tick() after reset failed: {e}")

        # 3. 새 차량 스폰
        spawned_count = self.spawn_vehicles(num_vehicles)
        
        return (removed_count, spawned_count)
        
    def get_vehicle_ids(self) -> Set[int]:
        return self.vehicle_ids