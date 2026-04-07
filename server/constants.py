"""Shared constants for the Crop Management environment.

Centralizes numeric thresholds that must agree across reward, grader,
inference, and environment modules.  Only values referenced by 2+ files
are extracted here.
"""

# ---------------------------------------------------------------------------
# Fertilizer DVS windows and target amounts
# Used by: server/reward.py, agent/inference.py
# CONSTRAINT: FERT_TARGET_DVS_N must lie within FERT_WINDOW_N
# ---------------------------------------------------------------------------
FERT_WINDOW_1 = (0.20, 0.40)       # DVS range for first fertilization
FERT_WINDOW_2 = (0.50, 0.70)       # DVS range for second fertilization
FERT_TARGET_DVS_1 = 0.30           # Optimal DVS for first application
FERT_TARGET_DVS_2 = 0.60           # Optimal DVS for second application
FERT_MAX_KG_PER_STEP = 50.0         # Environment cap on kg N/ha per step
DEFAULT_N_RECOV = 0.008             # Default N recovery factor (kg factor⁻¹)
FERT_HEURISTIC_OFFSET = 0.07       # Greedy heuristic shifts window start to avoid
                                    # applying right at window boundary

# ---------------------------------------------------------------------------
# Soil moisture thresholds
# Used by: server/reward.py, server/environment.py, agent/inference.py
# ---------------------------------------------------------------------------
SM_TARGET_LOW = 0.28               # Lower bound of ideal soil moisture band
SM_TARGET_HIGH = 0.32              # Upper bound of ideal soil moisture band
SM_WATER_DEFICIT = 0.22            # Below this = water deficit flag

# ---------------------------------------------------------------------------
# Grading constants
# Used by: server/grader.py (reference), server/reward.py
# ---------------------------------------------------------------------------
MAX_WATER_CM = 50.0                # Maximum reasonable irrigation for wheat season
HARVEST_DVS_LOW = 1.80             # Start of optimal harvest window
HARVEST_DVS_HIGH = 2.00            # End of optimal harvest window (= sim DVS cap)

# ---------------------------------------------------------------------------
# Grading weights (unified across all tasks)
# Used by: server/grader.py, documented in README
# ---------------------------------------------------------------------------
WEIGHT_YIELD = 0.35
WEIGHT_WATER = 0.20
WEIGHT_COST = 0.18
WEIGHT_TIMING = 0.15
WEIGHT_HARVEST = 0.12

# ---------------------------------------------------------------------------
# Episode limits
# Used by: server/environment.py
# ---------------------------------------------------------------------------
MAX_STEPS = 60                     # Safety cap on episode length (steps)

# ---------------------------------------------------------------------------
# Reward blend weights
# Used by: server/environment.py
# Validated against harvest_hesitation and drought_rescue probes.
# ---------------------------------------------------------------------------
REWARD_INTENT_WEIGHT = 0.4         # Weight for agronomic-intent reward
REWARD_DELTA_WEIGHT = 0.6          # Weight for observed-state-change reward

# ---------------------------------------------------------------------------
# Inspection action costs
# Used by: server/environment.py, agent/inference.py
# ---------------------------------------------------------------------------
INSPECT_SOIL_COST = 10             # Budget cost for inspect_soil action
INSPECT_CROP_COST = 20             # Budget cost for inspect_crop action
INSPECT_MAX_TOTAL = 2              # Max total inspects per episode (any mix)

# ---------------------------------------------------------------------------
# Observability band thresholds
# Used by: server/environment.py (observation coarsening)
# ---------------------------------------------------------------------------
SM_BAND_CRITICAL = 0.18            # SM below this = "critical"
SM_BAND_LOW = 0.22                 # SM below this = "low"
SM_BAND_ADEQUATE = 0.35            # SM below this = "adequate", above = "high"

N_VISUAL_DEFICIENT = 0.4           # n_factor below this = "deficient"
N_VISUAL_ADEQUATE = 0.7            # n_factor below this = "adequate", above = "surplus"

LAI_LOW = 1.5                      # LAI below this = "sparse"
LAI_MODERATE = 3.5                 # LAI below this = "moderate", above = "dense"

# ---------------------------------------------------------------------------
# Growth stage DVS midpoint map (for greedy fallback on hidden-DVS tiers)
# Labels match crop_sim.py growth_stage_name()
# Used by: agent/inference.py
# ---------------------------------------------------------------------------
GROWTH_STAGE_DVS_MAP: dict[str, float] = {
    "emergence": 0.075,
    "vegetative": 0.325,
    "flowering": 0.75,
    "grain_fill": 1.25,
    "ripening": 1.75,
    "mature": 2.0,
}

# ---------------------------------------------------------------------------
# SM band midpoint map (for greedy fallback on hidden-SM tiers)
# Used by: agent/inference.py
# ---------------------------------------------------------------------------
SM_BAND_MIDPOINT: dict[str, float] = {
    "critical": 0.15,
    "low": 0.20,
    "adequate": 0.285,
    "high": 0.40,
}
