"""WOFOST-inspired crop growth simulator.

A simplified but scientifically grounded crop growth model that captures
the key dynamics of the WOFOST (World Food Studies) model used in real
agricultural research. This pure-Python implementation (~200 lines) avoids
external dependencies while preserving the essential physics:

  - Temperature-sum phenology (DVS 0→1→2 progression)
  - Light-use-efficiency (LUE) biomass production
  - Water balance: rainfall + irrigation - evapotranspiration
  - Water stress effects on growth rate
  - Nitrogen response (simplified first-order depletion)
  - DVS-dependent partitioning to storage organs (grain yield)
  - LAI dynamics — growth during vegetative, senescence post grain-fill

The simulator is fully deterministic: same initial conditions + same
weather + same management actions always produce the same output.

Reference: van Diepen et al. (1989), "WOFOST: a simulation model of
crop production", Soil Use and Management, 5(1), 16-24.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Crop parameter library
# ---------------------------------------------------------------------------

@dataclass
class CropParams:
    """Crop-specific growth parameters.

    These correspond to the key WOFOST calibration parameters that control
    phenological development, canopy growth, and biomass production.
    """
    name: str
    display_name: str
    base_temp: float          # Base temperature for development (deg C)
    tsum1: float              # Temp sum sowing -> anthesis (deg C days)
    tsum2: float              # Temp sum anthesis -> maturity (deg C days)
    max_lai: float            # Maximum leaf area index
    sla: float                # Specific leaf area (ha leaf / kg DM)
    lue: float                # Light use efficiency (g DM / MJ PAR)
    initial_lai: float = 0.1
    senescence_dvs: float = 1.5   # DVS at which leaf senescence begins


# Partitioning tables: fraction of daily growth allocated to storage organs (grain).
# Interpolated linearly by DVS.  Before anthesis (DVS<1), virtually nothing
# goes to grain.  After DVS ~1.3, grain filling accelerates rapidly.
WHEAT_PARTITION = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.05),
                   (1.3, 0.40), (1.6, 0.70), (2.0, 0.85)]

# Maize partition table (defined for extensibility — currently unused)
MAIZE_PARTITION = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.10),
                   (1.3, 0.50), (1.6, 0.75), (2.0, 0.90)]

CROP_LIBRARY: dict[str, CropParams] = {
    "wheat": CropParams(
        name="wheat",
        display_name="Winter Wheat",
        base_temp=0.0,
        tsum1=1100.0,
        tsum2=1000.0,
        max_lai=6.0,
        sla=0.0020,
        lue=2.5,
        initial_lai=0.08,
    ),
    "maize": CropParams(
        name="maize",
        display_name="Maize",
        base_temp=10.0,
        tsum1=900.0,
        tsum2=800.0,
        max_lai=5.0,
        sla=0.0025,
        lue=3.0,
        initial_lai=0.05,
    ),
}

PARTITION_TABLES: dict[str, list[tuple[float, float]]] = {
    "wheat": WHEAT_PARTITION,
    "maize": MAIZE_PARTITION,
}


# ---------------------------------------------------------------------------
# Soil parameter library
# ---------------------------------------------------------------------------

@dataclass
class SoilParams:
    """Soil hydraulic properties controlling the water balance.

    These map to standard WOFOST soil input parameters. The rooting depth
    defines the soil volume over which moisture changes are computed.
    """
    name: str
    field_capacity: float     # Volumetric water content at field capacity
    wilting_point: float      # Volumetric water content at wilting point
    initial_sm: float         # Initial soil moisture (start of season)
    rooting_depth_mm: float   # Effective rooting depth (mm)


SOIL_LIBRARY: dict[str, SoilParams] = {
    "clay_loam": SoilParams("Clay Loam", 0.45, 0.20, 0.38, 1000),
    "sandy_loam": SoilParams("Sandy Loam", 0.35, 0.10, 0.28, 800),
    "silt_loam": SoilParams("Silt Loam", 0.40, 0.15, 0.33, 900),
}


# ---------------------------------------------------------------------------
# Crop growth simulator
# ---------------------------------------------------------------------------

class CropSimulator:
    """Simplified WOFOST-style crop growth model."""

    def __init__(
        self,
        crop_params: CropParams,
        soil_params: SoilParams,
        weather_data: list[dict],
        partition_table: list[tuple[float, float]],
    ):
        self.crop = crop_params
        self.soil = soil_params
        self.weather = weather_data
        self.partition_table = partition_table

        # State variables
        self.dvs: float = 0.0
        self.lai: float = crop_params.initial_lai
        self.tagp: float = 0.0      # Total above-ground production (kg/ha)
        self.twso: float = 0.0      # Storage organ (grain) weight (kg/ha)
        self.sm: float = soil_params.initial_sm
        self.current_day: int = 0
        self.n_factor: float = 0.55  # Nitrogen availability factor (0-1)
        self.total_water: float = 0.0
        self.total_n: float = 0.0

    def get_weather(self, day: int) -> dict:
        if 0 <= day < len(self.weather):
            return self.weather[day]
        return self.weather[-1]

    def get_weather_forecast(self, start_day: int, n_days: int = 5) -> list[dict]:
        """Return forecast with slight noise to simulate uncertainty."""
        forecast = []
        for d in range(start_day, start_day + n_days):
            w = self.get_weather(d)
            # Deterministic noise based on day
            noise_t = ((d * 17) % 7 - 3) * 0.4
            noise_r = ((d * 13) % 5 - 2) * 0.2
            noise_rad = ((d * 11) % 5 - 2) * 0.4
            forecast.append({
                "day": d,
                "tmax": round(w["tmax"] + noise_t, 1),
                "tmin": round(w["tmin"] + noise_t * 0.5, 1),
                "rain": round(max(0.0, w["rain"] + noise_r), 2),
                "radiation": round(max(0.5, w["radiation"] + noise_rad), 1),
            })
        return forecast

    def _heat_stress_factor(self, tmax: float) -> float:
        """Temperature penalty on growth during heat-sensitive stages.

        Flowering remains the most heat-sensitive stage due to pollen sterility.
        Grain fill also suffers under sustained heat, but with a milder bounded
        penalty so the benchmark remains trainable and deterministic.
        """
        heat_factor = 1.0

        # Pollen sterility above 35 C during flowering is the strongest penalty.
        if tmax > 35.0 and 0.8 < self.dvs < 1.2:
            heat_factor = min(heat_factor, max(0.3, 1.0 - (tmax - 35.0) * 0.15))

        # Kernel fill also suffers above ~32 C, but with a milder bounded effect.
        if tmax > 32.0 and 1.0 <= self.dvs < 1.5:
            heat_factor = min(heat_factor, max(0.8, 1.0 - (tmax - 32.0) * 0.035))

        return heat_factor

    def advance(self, days: int, irrigation_cm: float = 0.0, n_kg_ha: float = 0.0):
        """Advance the simulation by N days with optional interventions."""
        # Apply nitrogen
        if n_kg_ha > 0:
            # Each kg N pushes n_factor toward 1.0
            increase = n_kg_ha * 0.008
            self.n_factor = min(1.0, self.n_factor + increase)
            self.total_n += n_kg_ha

        # Spread irrigation evenly
        daily_irrig = irrigation_cm / days if days > 0 else 0.0
        self.total_water += irrigation_cm

        for _ in range(days):
            if self.dvs >= 2.0:
                break
            self._simulate_day(daily_irrig)

    def _simulate_day(self, irrigation_cm: float):
        w = self.get_weather(self.current_day)
        tavg = (w["tmax"] + w["tmin"]) / 2.0
        effective_temp = max(0.0, tavg - self.crop.base_temp)

        # --- DVS advancement (temperature sum) ---
        if self.dvs < 1.0:
            self.dvs += effective_temp / self.crop.tsum1
        else:
            self.dvs += effective_temp / self.crop.tsum2
        self.dvs = min(2.0, self.dvs)

        # --- Water balance ---
        rain_cm = w["rain"]
        et = self._evapotranspiration(w)
        depth_cm = self.soil.rooting_depth_mm / 10.0
        delta_sm = (rain_cm + irrigation_cm - et) / depth_cm
        self.sm += delta_sm
        self.sm = max(self.soil.wilting_point,
                      min(self.soil.field_capacity + 0.05, self.sm))

        # --- Stress factors ---
        water_stress = self._water_stress()

        # --- Heat stress (realistic for semi-arid climates like Punjab) ---
        heat_factor = self._heat_stress_factor(w["tmax"])

        # --- Biomass growth ---
        par = 0.5 * w["radiation"]  # PAR approx 50% of total radiation
        light_interception = 1.0 - math.exp(-0.65 * self.lai)
        # g/m2/day -> kg/ha/day (×10)
        potential_growth = par * self.crop.lue * light_interception * 10.0
        actual_growth = potential_growth * water_stress * self.n_factor * heat_factor

        if self.dvs >= 2.0:
            actual_growth = 0.0

        self.tagp += actual_growth

        # --- Partitioning ---
        pf = self._partition_fraction()
        self.twso += actual_growth * pf

        # --- LAI dynamics ---
        if self.dvs < self.crop.senescence_dvs:
            leaf_growth = actual_growth * (1.0 - pf) * self.crop.sla
            self.lai = min(self.crop.max_lai, self.lai + leaf_growth)
        else:
            # Post-senescence: faster rate matches real wheat (7-10 day senescence)
            senescence_rate = 0.10 * (self.dvs - self.crop.senescence_dvs)
            self.lai = max(0.1, self.lai * (1.0 - senescence_rate))

        # N depletion: phenology-aware (slow early, fast post-anthesis)
        if self.dvs < 1.0:
            n_loss = 0.0003   # Pre-anthesis: minimal N loss from soil
        else:
            n_loss = 0.0015   # Post-anthesis: accelerated N depletion
        self.n_factor = max(0.3, self.n_factor - n_loss)

        self.current_day += 1

    def _evapotranspiration(self, w: dict) -> float:
        """Simplified Hargreaves ET estimate (cm/day)."""
        tavg = (w["tmax"] + w["tmin"]) / 2.0
        td = max(0.1, w["tmax"] - w["tmin"])
        # Hargreaves reference ET (mm/day)
        et0 = 0.0023 * (tavg + 17.8) * math.sqrt(td) * w["radiation"] * 0.408
        et0 = max(0.0, et0)
        # Crop coefficient
        kc = min(1.2, 0.3 + 0.15 * self.lai)
        return et0 * kc * 0.1  # mm -> cm

    def _water_stress(self) -> float:
        """Water stress factor: 0.1 (severe) to 1.0 (none)."""
        available = self.sm - self.soil.wilting_point
        total_avail = self.soil.field_capacity - self.soil.wilting_point
        if total_avail <= 0:
            return 0.5
        ratio = available / total_avail
        if ratio > 0.6:
            return 1.0
        elif ratio > 0.2:
            return 0.5 + 0.5 * (ratio - 0.2) / 0.4
        else:
            return max(0.1, ratio / 0.2 * 0.5)

    def _partition_fraction(self) -> float:
        """Fraction of growth allocated to storage organs (grain)."""
        table = self.partition_table
        if self.dvs <= table[0][0]:
            return table[0][1]
        if self.dvs >= table[-1][0]:
            return table[-1][1]
        for i in range(len(table) - 1):
            if table[i][0] <= self.dvs <= table[i + 1][0]:
                frac = (self.dvs - table[i][0]) / (table[i + 1][0] - table[i][0])
                return table[i][1] + frac * (table[i + 1][1] - table[i][1])
        return 0.0

    def growth_stage_name(self) -> str:
        if self.dvs < 0.15:
            return "emergence"
        elif self.dvs < 0.5:
            return "vegetative"
        elif self.dvs < 1.0:
            return "flowering"
        elif self.dvs < 1.5:
            return "grain_fill"
        elif self.dvs < 2.0:
            return "ripening"
        else:
            return "mature"


# ---------------------------------------------------------------------------
# Potential yield computation
# ---------------------------------------------------------------------------

def compute_potential_yield(
    crop_name: str,
    weather_data: list[dict],
    max_days: int = 300,
) -> float:
    """Run simulation with unlimited resources to find potential yield."""
    crop = CROP_LIBRARY[crop_name]
    # Use best soil
    soil = SoilParams("optimal", 0.40, 0.15, 0.38, 900)
    table = PARTITION_TABLES[crop_name]
    sim = CropSimulator(crop, soil, weather_data, table)
    sim.n_factor = 1.0  # Perfect nitrogen

    step = 7
    limit = min(max_days, len(weather_data) - step)
    while sim.dvs < 2.0 and sim.current_day < limit:
        # Keep soil near field capacity
        if sim.sm < soil.field_capacity - 0.03:
            irrig = (soil.field_capacity - sim.sm) * soil.rooting_depth_mm / 100.0
        else:
            irrig = 0.0
        sim.advance(step, irrigation_cm=irrig, n_kg_ha=5.0)

    return round(sim.twso, 1)
