"""WOFOST crop and soil parameterisation with scientific references.

This module centralises all crop-specific and soil-specific constants used
by the CropSimulator.  Variable names follow the WOFOST naming conventions
established in the official documentation:

  Boogaard, H.L., de Wit, A.J.W., te Roller, J.A., & van Diepen, C.A.
  (2014). *WOFOST Technical Documentation*, Wageningen-UR.

Where we deviate from or simplify the full WOFOST model, the rationale and
source literature are cited inline.

WOFOST variable naming key (used throughout this module):
  TBASE   — Base temperature for phenological development (°C)
  TSUM1   — Temperature sum from sowing/emergence to anthesis (°C·d)
  TSUM2   — Temperature sum from anthesis to maturity (°C·d)
  KDIF    — Extinction coefficient for diffuse PAR (—)
  LUE     — Light-use efficiency (g DM MJ⁻¹ PAR)
  SLA0    — Specific leaf area at emergence (ha leaf kg⁻¹ DM)
  LAIEM   — LAI at emergence (m² m⁻²)
  SPAN    — Life span of leaves growing at 35°C (d)
  SLATB   — SLA table (DVS → SLA); simplified here to a single value
  FOTB    — Fraction of above-ground DM to storage organs by DVS
  SMFCF   — Soil moisture at field capacity (cm³ cm⁻³)
  SMW     — Soil moisture at wilting point (cm³ cm⁻³)
  SM0     — Soil moisture at saturation (cm³ cm⁻³, unused in 1-layer)
  RDMSOL  — Maximum rooting depth of the soil (cm)
  RDMCR   — Maximum rooting depth of the crop (cm)
  CFET    — Correction factor for ET (—); applied to Kc
  N_RECOV — Apparent N recovery fraction (kg uptake kg⁻¹ applied)

All numeric values carry a source tag, e.g. ``# [Boogaard2014]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Crop parameters ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class WOFOSTCropParams:
    """Crop growth parameters following WOFOST naming conventions.

    Parameters are calibrated for the crop variety and region indicated by
    *name* and *region*.  Default values represent NW-European winter wheat
    unless overridden by a regional profile.

    References for default values:
      [1] Boogaard et al. (2014) — WOFOST 7.1.7 Technical Documentation
      [2] de Wit et al. (2019) — 25 years of the WOFOST crop model, Agric. Syst.
      [3] Wolf et al. (2011) — FAO/IIASA GAEZ crop parameters
    """
    # ── Identity ──
    name: str                           # e.g. "wheat"
    display_name: str                   # e.g. "Winter Wheat (NL)"
    region: str = "generic"             # e.g. "NL", "Iowa", "Punjab"

    # ── Phenology ──
    TBASE: float = 0.0                  # °C — base temperature [1] Table 4.1
    TSUM1: float = 1116.0              # °C·d — emergence→anthesis [1][2]
    TSUM2: float = 903.0               # °C·d — anthesis→maturity  [1][2]

    # ── Canopy ──
    SLA0: float = 0.0022               # ha kg⁻¹ — specific leaf area at emergence [1]
    LAIEM: float = 0.048               # m² m⁻² — LAI at emergence [1]
    LAI_MAX: float = 6.0               # m² m⁻² — maximum LAI [1]
    KDIF: float = 0.60                 # — — extinction coeff. for diffuse PAR [1]
                                        #       (Monsi & Saeki 1953; 0.6 for cereals)
    SENESCENCE_DVS: float = 1.5        # DVS at which leaf senescence begins [1]
    SENESCENCE_RATE: float = 0.10      # d⁻¹ — daily LAI decline factor post SENESCENCE_DVS

    # ── Biomass ──
    LUE: float = 2.8                   # g DM MJ⁻¹ PAR — light-use efficiency
                                        #   (Gallagher & Biscoe 1978; 2.7-3.0 for C3)
    # ── Nitrogen ──
    N_RECOV: float = 0.008             # kg factor⁻¹ — apparent N recovery per kg applied
    N_LOSS_PRE: float = 0.0003         # d⁻¹ — N-factor daily depletion pre-anthesis
    N_LOSS_POST: float = 0.0015        # d⁻¹ — N-factor daily depletion post-anthesis
                                        #   (Shibu et al. 2010, simplified linear)
    N_FACTOR_FLOOR: float = 0.30       # — — minimum N availability factor
    N_FACTOR_INIT: float = 0.55        # — — initial N availability at sowing

    # ── N leaching & slow-release parameters ──
    LEACH_RATE: float = 0.35            # fraction of excess-water N lost per cm
    SLOW_RELEASE_IMMEDIATE: float = 0.70 # fraction of slow-release N applied immediately
    SLOW_RELEASE_LEACH_FACTOR: float = 0.30  # leaching multiplier (0.30 = 70% resistant)

    # ── Grain shattering / spoilage ──
    # Post-maturity grain loss: kernels detach from the ear and
    # moisture + microbial activity reduce grain quality.
    # [Paulsen & Shroyer 2008; Gan et al. 2018]
    SHATTER_DVS: float = 1.85          # DVS at which grain shattering begins
    SHATTER_RATE: float = 0.25         # d⁻¹ — daily TWSO loss fraction per DVS above threshold

    # ── Heat stress thresholds ──
    HEAT_FLOWER_THRESH: float = 35.0   # °C — pollen sterility threshold
                                        #   (Prasad et al. 2006; Porter & Gawith 1999)
    HEAT_FLOWER_SLOPE: float = 0.15    # — — growth reduction per °C above threshold
    HEAT_FLOWER_FLOOR: float = 0.30    # — — minimum heat factor during flowering
    HEAT_FLOWER_DVS: tuple[float, float] = (0.8, 1.2)  # DVS window for flowering stress

    HEAT_GRAIN_THRESH: float = 32.0    # °C — grain fill heat threshold
                                        #   (Wardlaw & Moncur 1995)
    HEAT_GRAIN_SLOPE: float = 0.035    # — — growth reduction per °C above threshold
    HEAT_GRAIN_FLOOR: float = 0.80     # — — minimum heat factor during grain fill
    HEAT_GRAIN_DVS: tuple[float, float] = (1.0, 1.5)   # DVS window for grain-fill stress

    # ── Evapotranspiration (Hargreaves & Samani 1985) ──
    ET_COEFF: float = 0.0023           # Hargreaves empirical coefficient
    ET_TCONST: float = 17.8            # °C — Hargreaves temperature constant
    ET_RAD_CONV: float = 0.408         # MJ m⁻² d⁻¹ → mm d⁻¹ conversion
                                        #   (Allen et al. 1998, FAO-56)
    KC_MAX: float = 1.20               # — — maximum crop coefficient
    KC_BASE: float = 0.30              # — — Kc at bare soil / low LAI
    KC_LAI_SLOPE: float = 0.15         # — — Kc increase per unit LAI

    # ── Water stress (Feddes et al. 1978 reduction function) ──
    WS_RATIO_FULL: float = 0.6         # θ ratio above which stress = 1.0
    WS_RATIO_MID: float = 0.2          # θ ratio threshold for mid-range stress
    WS_FLOOR: float = 0.1              # — — minimum water stress factor

    # ── Partitioning table (FOTB) ──
    # DVS → fraction of daily growth allocated to storage organs (grain)
    # Interpolated linearly.  Before anthesis virtually no grain allocation;
    # post DVS 1.3, grain-fill dominates. [1] Table 4.7
    FOTB: tuple[tuple[float, float], ...] = (
        (0.0, 0.00), (0.5, 0.00), (1.0, 0.05),
        (1.3, 0.40), (1.6, 0.70), (2.0, 0.85),
    )


# ── Soil parameters ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class WOFOSTSoilParams:
    """Soil hydraulic properties following WOFOST naming conventions.

    Standard soil-water retention parameters that control the single-layer
    water balance in the CropSimulator.

    References:
      [1] Boogaard et al. (2014) — WOFOST 7.1.7
      [4] Wösten et al. (2001) — Pedotransfer functions, Geoderma 99
    """
    name: str
    display_name: str = ""

    SMFCF: float = 0.43                # cm³ cm⁻³ — field capacity [4]
    SMW: float = 0.20                  # cm³ cm⁻³ — wilting point [4]
    SM_INIT: float = 0.38              # cm³ cm⁻³ — initial soil moisture
    RDMSOL: float = 100.0              # cm — maximum rooting depth of soil [1]

    def __post_init__(self):
        if self.SMW >= self.SMFCF:
            raise ValueError(
                f"Wilting point ({self.SMW}) must be < field capacity ({self.SMFCF})"
            )

    # ── Legacy property aliases (used by environment.py, scenarios.py) ──
    @property
    def field_capacity(self) -> float:
        return self.SMFCF

    @property
    def wilting_point(self) -> float:
        return self.SMW

    @property
    def initial_sm(self) -> float:
        return self.SM_INIT

    @property
    def rooting_depth_mm(self) -> float:
        """Return RDMSOL converted from cm → mm for legacy consumers."""
        return self.RDMSOL * 10.0


# ── Pre-built parameter sets ────────────────────────────────────────────


# --- Winter Wheat (Netherlands) — primary calibration ---
# Source: Boogaard et al. (2014), Table 4.1 & 4.7; TSUM values from
# de Wit et al. (2019) NW-European winter wheat
WHEAT_NL = WOFOSTCropParams(
    name="wheat",
    display_name="Winter Wheat (Netherlands)",
    region="NL",
    TSUM1=1116.0,                       # [2] NL calibration
    TSUM2=903.0,                        # [2] NL calibration
    TBASE=0.0,                          # [1] standard for wheat
    KDIF=0.60,                          # [1] cereals: 0.56–0.65
    LUE=2.8,                            # Gallagher & Biscoe (1978): 2.7–3.0 for C3
    SLA0=0.0022,                        # [1] ha kg⁻¹
    LAIEM=0.048,                        # [1] winter wheat
    LAI_MAX=6.0,                        # [1]
)

# --- Winter Wheat (Iowa, USA) — continental calibration ---
# Slightly longer growing season; drought-prone summers.
# TSUM adjusted from Loomis & Connor (1992), Crop Ecology.
WHEAT_IOWA = WOFOSTCropParams(
    name="wheat",
    display_name="Winter Wheat (Iowa, USA)",
    region="Iowa",
    TSUM1=1150.0,                       # Longer vegetative in continental climate
    TSUM2=950.0,                        # Slightly slower grain fill
    TBASE=0.0,
    KDIF=0.60,
    LUE=2.8,
    SLA0=0.0022,
    LAIEM=0.048,
    LAI_MAX=5.5,                        # Slightly lower due to water limitations
)

# --- Spring Wheat (Punjab, India) — semi-arid calibration ---
# Shorter, hotter growing season.  TSUM from Aggarwal & Kalra (1994),
# "Analyzing constraints limiting wheat yield using WTGROWS model".
WHEAT_PUNJAB = WOFOSTCropParams(
    name="wheat",
    display_name="Spring Wheat (Punjab, India)",
    region="Punjab",
    TSUM1=1050.0,                       # Aggarwal & Kalra (1994)
    TSUM2=850.0,                        # Faster maturity under heat
    TBASE=0.0,
    KDIF=0.60,
    LUE=2.6,                            # Slightly lower due to heat stress
    SLA0=0.0022,
    LAIEM=0.048,
    LAI_MAX=5.0,                        # Heat limits canopy development
    HEAT_FLOWER_THRESH=34.0,            # More sensitive — tropically adapted
    HEAT_GRAIN_THRESH=31.0,
    N_LOSS_POST=0.0020,                 # Faster N depletion in hot soils
)


# --- Soil profiles ---

SOIL_CLAY_LOAM = WOFOSTSoilParams(
    name="clay_loam",
    display_name="Clay Loam",
    SMFCF=0.43,                         # [4] Wösten et al. (2001)
    SMW=0.20,                           # [4]
    SM_INIT=0.38,
    RDMSOL=100.0,                       # 100 cm — typical for NL clay loam
)

SOIL_SANDY_LOAM = WOFOSTSoilParams(
    name="sandy_loam",
    display_name="Sandy Loam",
    SMFCF=0.35,                         # [4]
    SMW=0.10,                           # [4]
    SM_INIT=0.28,
    RDMSOL=80.0,
)

SOIL_SILT_LOAM = WOFOSTSoilParams(
    name="silt_loam",
    display_name="Silt Loam",
    SMFCF=0.40,                         # [4]
    SMW=0.15,                           # [4]
    SM_INIT=0.33,
    RDMSOL=90.0,
)


# ── Lookup helpers ───────────────────────────────────────────────────────

# Legacy-compatible dicts (used by scenarios.py until YAML migration)
CROP_LIBRARY = {
    "wheat_nl": WHEAT_NL,
    "wheat_iowa": WHEAT_IOWA,
    "wheat_punjab": WHEAT_PUNJAB,
}

SOIL_LIBRARY = {
    "clay_loam": SOIL_CLAY_LOAM,
    "sandy_loam": SOIL_SANDY_LOAM,
    "silt_loam": SOIL_SILT_LOAM,
}


def get_crop_params(crop_key: str) -> WOFOSTCropParams:
    """Retrieve crop parameters by key.  Raises KeyError if not found."""
    return CROP_LIBRARY[crop_key]


def get_soil_params(soil_key: str) -> WOFOSTSoilParams:
    """Retrieve soil parameters by key.  Raises KeyError if not found."""
    return SOIL_LIBRARY[soil_key]


# ── YAML loading ─────────────────────────────────────────────────────────

import pathlib

import yaml

_CONFIGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "configs"


def load_profile_from_yaml(
    yaml_path: str | pathlib.Path,
) -> tuple[WOFOSTCropParams, WOFOSTSoilParams]:
    """Load crop + soil parameters from a YAML config file.

    Returns ``(WOFOSTCropParams, WOFOSTSoilParams)`` constructed from the
    ``crop:`` and ``soil:`` sections of the file.  Any field not specified
    in the YAML falls back to the dataclass default.

    Raises ``FileNotFoundError`` if the path does not exist,
    ``ValueError`` if YAML is malformed or missing required keys.
    """
    path = pathlib.Path(yaml_path)
    if not path.is_absolute():
        path = _CONFIGS_DIR / path

    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {path}, got {type(data).__name__}")
    for required in ("crop", "soil"):
        if required not in data:
            raise ValueError(f"Missing required key '{required}' in {path}")

    crop_dict = dict(data["crop"])
    soil_dict = dict(data["soil"])

    # Convert FOTB list-of-lists → tuple-of-tuples for frozen dataclass
    if "FOTB" in crop_dict:
        crop_dict["FOTB"] = tuple(tuple(row) for row in crop_dict["FOTB"])

    # Convert DVS window lists → tuples
    for key in ("HEAT_FLOWER_DVS", "HEAT_GRAIN_DVS"):
        if key in crop_dict:
            crop_dict[key] = tuple(crop_dict[key])

    crop_params = WOFOSTCropParams(**crop_dict)
    soil_params = WOFOSTSoilParams(**soil_dict)
    return crop_params, soil_params


def list_available_configs() -> list[str]:
    """Return basenames of YAML config files in the configs/ directory."""
    if not _CONFIGS_DIR.is_dir():
        return []
    return sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))
