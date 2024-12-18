from dataclasses import dataclass

@dataclass
class ModelParams:
    year: str
    threshold: float
    solar_absorptance: float
    emissivity: float
    heat_transfer_coefficient: float
    thermal_capacity: float

@dataclass
class WeatherData:
    GHI: list[int]
    DNI: list[int]
    angle: float
    temps: list[float]
    
@dataclass
class PredictionInput:
    state: str
    coordinates: tuple[float]
    weather_data: WeatherData

@dataclass
class PredictionOutput:
    state: str
    coordinates: tuple[float]
    predicted_upper_temps: list[float]
    predicted_lower_temps: list[float]
    n_hours_overtemp: float
    n_hours_undertemp: float
    max_van_temp: int
    min_van_temp: int