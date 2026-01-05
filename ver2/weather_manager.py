#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import carla
import re
import math
from typing import List, Tuple, Optional

# --- dynamic_weather.py 로직 ---

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)

class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)

class DynamicWeather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

# --- WeatherManager (통합) ---

class WeatherManager:
    def __init__(self, world: carla.World):
        self.world = world
        self.weather_presets = self._find_weather_presets()
        self.dynamic_weather: Optional[DynamicWeather] = None
        self.is_dynamic = False

    def _find_weather_presets(self) -> List[Tuple[carla.WeatherParameters, str]]:
        """
        carla.WeatherParameters에서 프리셋 목록을 찾습니다.
        """
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    def get_preset_names(self) -> List[str]:
        return [name for _, name in self.weather_presets]

    def set_weather_by_preset_name(self, name: str):
        """
        프리셋 이름으로 날씨를 설정합니다. 동적 날씨는 비활성화됩니다.
        """
        self.stop_dynamic_weather()
        for preset, preset_name in self.weather_presets:
            if preset_name == name:
                self.world.set_weather(preset)
                return
        print(f"[WeatherManager] Preset '{name}' not found.")

    def start_dynamic_weather(self, speed_factor: float = 1.0):
        """
        동적 날씨 시뮬레이션을 시작합니다.
        """
        current_weather = self.world.get_weather()
        self.dynamic_weather = DynamicWeather(current_weather)
        self.is_dynamic = True
        self._speed_factor = speed_factor
        print("[WeatherManager] Dynamic weather started.")

    def stop_dynamic_weather(self):
        if self.is_dynamic:
            self.dynamic_weather = None
            self.is_dynamic = False
            print("[WeatherManager] Dynamic weather stopped.")
            
    def toggle_dynamic_weather(self):
        if self.is_dynamic:
            self.stop_dynamic_weather()
        else:
            self.start_dynamic_weather()

    def tick(self, delta_seconds: float):
        """
        동적 날씨가 활성화된 경우, 날씨를 한 스텝 업데이트합니다.
        """
        if self.is_dynamic and self.dynamic_weather:
            self.dynamic_weather.tick(delta_seconds * self._speed_factor)
            self.world.set_weather(self.dynamic_weather.weather)