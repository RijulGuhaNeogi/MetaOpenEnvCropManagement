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
FERT_TARGET_KG_1 = 18.0            # kg N/ha for first application
FERT_TARGET_KG_2 = 15.0            # kg N/ha for second application
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
HARVEST_DVS_HIGH = 2.05            # End of optimal harvest window

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
