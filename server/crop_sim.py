"""WOFOST-inspired crop growth simulator for precision agriculture.

A simplified but scientifically grounded crop growth model implementing
the core dynamics of WOFOST (WOrld FOod STudies), the standard European
crop simulation model used in agricultural research and policy.

Implemented sub-models and their scientific basis:

  1. **Phenology** — Temperature-sum (thermal-time) approach for DVS
     progression from emergence (0) through anthesis (1) to maturity (2).
     van Diepen et al. (1989), Eq. 2.1.

  2. **Light interception** — Beer–Lambert law with extinction coefficient
     KDIF for diffuse PAR.  Monsi & Saeki (1953).

  3. **Biomass accumulation** — Light-Use Efficiency (LUE) approach,
     converting intercepted PAR to dry matter.
     Gallagher & Biscoe (1978), Field Crops Res. 1: 355–367.

  4. **Reference ET** — Hargreaves & Samani (1985) equation, a
     temperature-based alternative to Penman–Monteith when full met data
     is unavailable.

  5. **Water stress** — Feddes et al. (1978) piecewise-linear reduction
     function relating plant-available water to a stress factor [0–1].

  6. **Heat stress** — Pollen sterility at flowering (Prasad et al. 2006;
     Porter & Gawith 1999) and reduced grain fill under sustained heat
     (Wardlaw & Moncur 1995).

  7. **Nitrogen response** — Simplified first-order depletion with
     injection recovery, after Shibu et al. (2010).

  8. **Grain partitioning** — DVS-dependent FOTB table interpolation,
     following Boogaard et al. (2014), Table 4.7.

Key simplifications vs full WOFOST:
  - Single-layer soil water balance (vs. multi-layer in WOFOST 7.1.7)
  - Linear N response (vs. Michaelis–Menten kinetics / LINTUL-N)
  - Fixed rooting depth (vs. dynamic root growth module)
  - No pest/disease sub-model
  - Deterministic weather noise for forecasts

The simulator is fully deterministic: same initial conditions + same
weather + same management actions always produce the same output.

Full references: see docs/REFERENCES.md
"""
from __future__ import annotations

import math

from server.crop_params import (
    CROP_LIBRARY,
    SOIL_LIBRARY,
    WOFOSTCropParams,
    WOFOSTSoilParams,
)


# ── Legacy compatibility aliases ────────────────────────────────────────
# These allow existing code (scenarios.py, tests) to keep working while we
# migrate to WOFOSTCropParams / WOFOSTSoilParams throughout the codebase.
CropParams = WOFOSTCropParams
SoilParams = WOFOSTSoilParams

# PARTITION_TABLES: legacy compat — extract FOTB from each crop profile.
PARTITION_TABLES: dict[str, tuple] = {
    key: params.FOTB for key, params in CROP_LIBRARY.items()
}


# ---------------------------------------------------------------------------
# Crop growth simulator
# ---------------------------------------------------------------------------

class CropSimulator:
    """Simplified WOFOST-style crop growth model.

    Accepts ``WOFOSTCropParams`` and ``WOFOSTSoilParams`` from
    ``server.crop_params``.  All numeric constants are read from the
    parameter objects — no magic numbers in the simulation loop.
    """

    def __init__(
        self,
        crop_params: WOFOSTCropParams,
        soil_params: WOFOSTSoilParams,
        weather_data: list[dict],
        partition_table: list[tuple[float, float]] | None = None,
    ):
        self.crop = crop_params
        self.soil = soil_params
        self.weather = weather_data
        # Use crop's FOTB if no explicit partition table given
        self.partition_table = (
            list(partition_table) if partition_table is not None
            else list(crop_params.FOTB)
        )

        # ── State variables ──
        self.dvs: float = 0.0                          # Development stage [0–2]
        self.lai: float = crop_params.LAIEM             # Leaf area index (m² m⁻²)
        self.tagp: float = 0.0                          # Total above-ground prod. (kg ha⁻¹)
        self.twso: float = 0.0                          # Storage organ / grain (kg ha⁻¹)
        self.sm: float = soil_params.SM_INIT            # Soil moisture (cm³ cm⁻³)
        self.current_day: int = 0
        self.n_factor: float = crop_params.N_FACTOR_INIT  # Nitrogen availability [0–1]
        self.total_water: float = 0.0                   # Cumulative irrigation (cm)
        self.total_n: float = 0.0                       # Cumulative N applied (kg ha⁻¹)

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

        Flowering is the most heat-sensitive stage due to pollen sterility —
        grain number drops sharply above ~35 °C (Prasad et al. 2006;
        Porter & Gawith 1999).  Grain-fill suffers milder weight loss
        above ~32 °C (Wardlaw & Moncur 1995).

        Thresholds and slopes are read from ``WOFOSTCropParams`` so they
        can be regionally calibrated (e.g. Punjab wheat is more sensitive).
        """
        cp = self.crop
        heat_factor = 1.0

        # Pollen sterility during flowering [Prasad et al. 2006]
        dvs_lo, dvs_hi = cp.HEAT_FLOWER_DVS
        if tmax > cp.HEAT_FLOWER_THRESH and dvs_lo < self.dvs < dvs_hi:
            heat_factor = min(
                heat_factor,
                max(cp.HEAT_FLOWER_FLOOR,
                    1.0 - (tmax - cp.HEAT_FLOWER_THRESH) * cp.HEAT_FLOWER_SLOPE),
            )

        # Kernel weight reduction during grain fill [Wardlaw & Moncur 1995]
        dvs_lo_g, dvs_hi_g = cp.HEAT_GRAIN_DVS
        if tmax > cp.HEAT_GRAIN_THRESH and dvs_lo_g <= self.dvs < dvs_hi_g:
            heat_factor = min(
                heat_factor,
                max(cp.HEAT_GRAIN_FLOOR,
                    1.0 - (tmax - cp.HEAT_GRAIN_THRESH) * cp.HEAT_GRAIN_SLOPE),
            )

        return heat_factor

    def advance(self, days: int, irrigation_cm: float = 0.0, n_kg_ha: float = 0.0):
        """Advance the simulation by *days* with optional interventions."""
        if days <= 0:
            raise ValueError(f"advance() requires days > 0, got {days}")

        # Nitrogen injection [Shibu et al. 2010, simplified linear recovery]
        if n_kg_ha > 0:
            increase = n_kg_ha * self.crop.N_RECOV   # N_RECOV ≈ 0.008
            self.n_factor = min(1.0, self.n_factor + increase)
            self.total_n += n_kg_ha

        # Spread irrigation evenly across the period
        daily_irrig = irrigation_cm / days
        self.total_water += irrigation_cm

        for _ in range(days):
            if self.dvs >= 2.0:
                break
            self._simulate_day(daily_irrig)

    def _simulate_day(self, irrigation_cm: float):
        """Simulate one day of crop growth.

        Implements the WOFOST daily loop: phenology → water balance →
        stress factors → biomass growth → partitioning → LAI dynamics →
        nitrogen depletion.
        """
        cp = self.crop
        w = self.get_weather(self.current_day)
        tavg = (w["tmax"] + w["tmin"]) / 2.0
        effective_temp = max(0.0, tavg - cp.TBASE)   # [van Diepen 1989, Eq. 2.1]

        # ── Phenology: temperature-sum DVS advancement ──
        # DVS 0→1 (vegetative): scaled by TSUM1
        # DVS 1→2 (reproductive): scaled by TSUM2
        if self.dvs < 1.0:
            self.dvs += effective_temp / cp.TSUM1
        else:
            self.dvs += effective_temp / cp.TSUM2
        self.dvs = min(2.0, self.dvs)

        # ── Water balance (single-layer tipping-bucket) ──
        rain_cm = w["rain"]
        et = self._evapotranspiration(w)
        depth_cm = self.soil.RDMSOL                    # already in cm
        delta_sm = (rain_cm + irrigation_cm - et) / depth_cm
        self.sm += delta_sm
        self.sm = max(self.soil.SMW,
                      min(self.soil.SMFCF + 0.05, self.sm))

        # ── Stress factors ──
        water_stress = self._water_stress()            # [Feddes et al. 1978]
        heat_factor = self._heat_stress_factor(w["tmax"])

        # ── Biomass growth (LUE approach) ──
        # PAR ≈ 50 % of total incoming radiation (Szeicz 1974)
        par = 0.5 * w["radiation"]
        # Beer–Lambert light interception [Monsi & Saeki 1953]
        light_interception = 1.0 - math.exp(-cp.KDIF * self.lai)
        # Potential growth: g m⁻² d⁻¹ → kg ha⁻¹ d⁻¹  (×10)
        potential_growth = par * cp.LUE * light_interception * 10.0
        actual_growth = potential_growth * water_stress * self.n_factor * heat_factor

        if self.dvs >= 2.0:
            actual_growth = 0.0

        self.tagp += actual_growth

        # ── Grain partitioning (FOTB table interpolation) ──
        # [Boogaard et al. 2014, Table 4.7]
        pf = self._partition_fraction()
        self.twso += actual_growth * pf

        # ── LAI dynamics ──
        if self.dvs < cp.SENESCENCE_DVS:
            # Vegetative + early reproductive: new leaf growth
            leaf_growth = actual_growth * (1.0 - pf) * cp.SLA0
            self.lai = min(cp.LAI_MAX, self.lai + leaf_growth)
        else:
            # Post-senescence: accelerated decline (7–10 day
            # senescence period realistic for wheat)
            senescence_rate = cp.SENESCENCE_RATE * (self.dvs - cp.SENESCENCE_DVS)
            self.lai = max(0.1, self.lai * (1.0 - senescence_rate))

        # ── Nitrogen depletion [Shibu et al. 2010, simplified] ──
        if self.dvs < 1.0:
            n_loss = cp.N_LOSS_PRE     # Pre-anthesis: minimal soil N loss
        else:
            n_loss = cp.N_LOSS_POST    # Post-anthesis: accelerated depletion
        self.n_factor = max(cp.N_FACTOR_FLOOR, self.n_factor - n_loss)

        self.current_day += 1

    def _evapotranspiration(self, w: dict) -> float:
        """Reference ET via Hargreaves & Samani (1985), then crop-adjusted.

        ET₀ = 0.0023 · (T_avg + 17.8) · √(T_range) · Ra · 0.408
        ET_crop = ET₀ · Kc(LAI)

        where Kc follows a simple LAI-dependent model:
          Kc = min(KC_MAX, KC_BASE + KC_LAI_SLOPE · LAI)

        Allen et al. (1998), FAO Irrigation & Drainage Paper 56, provides
        the radiation unit conversion factor (0.408).
        """
        cp = self.crop
        tavg = (w["tmax"] + w["tmin"]) / 2.0
        td = max(0.1, w["tmax"] - w["tmin"])
        # Hargreaves reference ET (mm d⁻¹)
        et0 = cp.ET_COEFF * (tavg + cp.ET_TCONST) * math.sqrt(td) * w["radiation"] * cp.ET_RAD_CONV
        et0 = max(0.0, et0)
        # Crop coefficient (LAI-dependent)
        kc = min(cp.KC_MAX, cp.KC_BASE + cp.KC_LAI_SLOPE * self.lai)
        return et0 * kc * 0.1           # mm → cm

    def _water_stress(self) -> float:
        """Feddes et al. (1978) piecewise-linear water-stress reduction.

        Returns a factor in [WS_FLOOR, 1.0]:
          - ratio > WS_RATIO_FULL  → 1.0 (no stress)
          - WS_RATIO_MID < ratio ≤ WS_RATIO_FULL → linear ramp 0.5–1.0
          - ratio ≤ WS_RATIO_MID  → linear ramp WS_FLOOR–0.5

        ``ratio`` = plant-available water / total available water
                  = (θ − θ_wp) / (θ_fc − θ_wp)
        """
        cp = self.crop
        available = self.sm - self.soil.SMW
        total_avail = self.soil.SMFCF - self.soil.SMW
        if total_avail <= 0:
            return 0.5
        ratio = available / total_avail
        if ratio > cp.WS_RATIO_FULL:
            return 1.0
        elif ratio > cp.WS_RATIO_MID:
            return 0.5 + 0.5 * (ratio - cp.WS_RATIO_MID) / (cp.WS_RATIO_FULL - cp.WS_RATIO_MID)
        else:
            return max(cp.WS_FLOOR, ratio / cp.WS_RATIO_MID * 0.5)

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
    """Run simulation with unlimited resources to find potential yield.

    Uses the named crop profile from CROP_LIBRARY and an "optimal" soil
    with generous field capacity and deep rooting.
    """
    crop = CROP_LIBRARY[crop_name]
    soil = WOFOSTSoilParams(
        name="optimal", SMFCF=0.40, SMW=0.15, SM_INIT=0.38, RDMSOL=90.0,
    )
    sim = CropSimulator(crop, soil, weather_data, crop.FOTB)
    sim.n_factor = 1.0  # Perfect nitrogen

    step = 7
    limit = min(max_days, len(weather_data) - step)
    while sim.dvs < 2.0 and sim.current_day < limit:
        # Keep soil near field capacity
        if sim.sm < soil.SMFCF - 0.03:
            irrig = (soil.SMFCF - sim.sm) * soil.RDMSOL / 10.0
        else:
            irrig = 0.0
        sim.advance(step, irrigation_cm=irrig, n_kg_ha=5.0)

    return round(sim.twso, 1)
