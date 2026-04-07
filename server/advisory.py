"""Deterministic advisory text generator for crop management observations.

Produces a neutral, factual natural-language summary of current field
conditions.  The text is *descriptive only* — it never prescribes actions
(e.g. "you should irrigate").  This gives LLM agents rich contextual
information without biasing their decision-making.

Design rules:
  1. Same state → same text (deterministic, template-based).
  2. Always includes: week, growth stage, DVS, soil moisture, forecast,
     budget status.
  3. Adds ALERT prefix for: water deficit, extreme heat, low budget,
     approaching maturity.
  4. Never prescribes specific actions.
"""
from __future__ import annotations


# ── Growth stage labels ──────────────────────────────────────────────────

# DVS thresholds aligned with WOFOSTCropParams:
#   HEAT_FLOWER_DVS = (0.8, 1.2)  — flowering heat-sensitivity window
#   HEAT_GRAIN_DVS  = (1.0, 1.5)  — grain-fill heat-sensitivity window

def _growth_stage_label(dvs: float) -> str:
    if dvs < 0.01:
        return "Pre-emergence"
    elif dvs < 0.30:
        return "Early vegetative (tillering)"
    elif dvs < 0.65:
        return "Late vegetative (stem elongation)"
    elif dvs < 0.80:
        return "Booting / heading"
    elif dvs < 1.20:
        return "Flowering (anthesis)"
    elif dvs < 1.50:
        return "Grain filling"
    elif dvs < 2.0:
        return "Ripening"
    else:
        return "Mature"


def _moisture_descriptor(sm: float, fc: float, wp: float) -> str:
    """Describe soil moisture relative to field capacity and wilting point."""
    if fc <= wp:
        return "unknown"
    ratio = (sm - wp) / (fc - wp)
    if ratio > 0.8:
        return "adequate"
    elif ratio > 0.5:
        return "moderate"
    elif ratio > 0.25:
        return "low"
    else:
        return "critically low"


# ── Main generator ───────────────────────────────────────────────────────

def generate_advisory(
    *,
    day: int,
    days_remaining: int,
    step_days: int,
    dvs: float,
    lai: float,
    sm: float,
    field_capacity: float,
    wilting_point: float,
    water_stress: float,
    n_availability: float,
    weather_today_tmax: float,
    forecast_rain_3d: float,
    forecast_rain_7d: float,
    total_water_cm: float,
    total_n_kg_ha: float,
    budget_remaining: float,
    budget_total: float,
    location: str,
    tier: int = 1,
    fert_count: int = 0,
    has_crop_report: bool = False,
) -> str:
    """Generate a neutral advisory paragraph from current field state.

    All parameters are plain floats/ints — no model objects required, so the
    function stays decoupled from Pydantic models.
    """
    parts: list[str] = []
    alerts: list[str] = []

    # ── Time context ──
    week = day // step_days + 1
    total_weeks = (day + days_remaining) // step_days
    parts.append(f"Week {week} of {total_weeks}.")

    # ── Growth stage ──
    stage = _growth_stage_label(dvs)
    if tier == 1:
        parts.append(f"{stage} (DVS {dvs:.2f}).")
    else:
        parts.append(f"{stage}.")

    # ── Soil moisture ──
    moisture = _moisture_descriptor(sm, field_capacity, wilting_point)
    if tier == 1:
        sm_pct = sm * 100
        parts.append(f"Soil moisture {moisture} at {sm_pct:.0f}%.")
    else:
        parts.append(f"Soil moisture {moisture}.")

    if water_stress < 0.7:
        if tier == 1:
            alerts.append(f"Water stress factor {water_stress:.2f} — crop transpiration limited.")
        else:
            alerts.append("Water stress detected — crop transpiration limited.")

    # Soil moisture context: reference optimal range
    if sm < 0.28 and forecast_rain_3d < 0.3:
        parts.append("Moisture below optimal range (28-32%) with little rain forecast.")
    elif sm > 0.36:
        parts.append("Soil well above optimal moisture range — no irrigation needed.")

    # ── Nitrogen ──
    if n_availability < 0.45:
        alerts.append(f"Nitrogen availability low ({n_availability:.2f}).")

    # ── Fertilizer window context ──
    if 0.20 <= dvs <= 0.40:
        if tier == 1:
            parts.append(f"Crop is in the first fertilization window (DVS 0.20-0.40, target 0.30, currently {dvs:.2f}).")
        else:
            parts.append("Crop is in the first fertilization window (DVS 0.20-0.40, target 0.30).")
        # Timing guidance
        if dvs < 0.25:
            parts.append("Early in window — waiting closer to target DVS 0.30 improves timing score.")
        elif dvs <= 0.35:
            parts.append("Near optimal fertilization timing (target DVS 0.30).")
        else:
            parts.append("Late in window — fertilize soon before it closes.")
        if tier == 1:
            # T1: agent can see exact n_avail, give dose guidance
            if n_availability < 0.5:
                parts.append("Nitrogen is very low — a large application (~50kg) is recommended.")
            elif n_availability < 0.65:
                parts.append("Nitrogen is low — a substantial application (~45kg) is recommended.")
            elif n_availability < 0.8:
                parts.append("Nitrogen is moderate — fertilization needed (~30kg).")
            elif n_availability < 0.9:
                parts.append("Nitrogen is adequate — a small application (~15kg) may suffice.")
            else:
                parts.append("Nitrogen is surplus — no fertilization needed.")
        else:
            # T2/T3: factual N status only
            if n_availability < 0.5:
                parts.append("Nitrogen is very low.")
            elif n_availability < 0.65:
                parts.append("Nitrogen is low.")
            elif n_availability < 0.8:
                parts.append("Nitrogen is moderate.")
            elif n_availability < 0.9:
                parts.append("Nitrogen is adequate.")
            else:
                parts.append("Nitrogen is surplus.")
        if tier >= 2 and fert_count == 0 and budget_remaining >= 10:
            parts.append("Consider inspect_soil ($10) to check exact nitrogen level.")
    elif 0.50 <= dvs <= 0.70:
        if tier == 1:
            parts.append(f"Crop is in the second fertilization window (DVS 0.50-0.70, target 0.60, currently {dvs:.2f}).")
        else:
            parts.append("Crop is in the second fertilization window (DVS 0.50-0.70, target 0.60).")
        # Timing guidance
        if dvs < 0.55:
            parts.append("Early in window — waiting closer to target DVS 0.60 improves timing score.")
        elif dvs <= 0.65:
            parts.append("Near optimal fertilization timing (target DVS 0.60).")
        else:
            parts.append("Late in window — fertilize soon before it closes.")
        if tier == 1:
            if n_availability < 0.5:
                parts.append("Nitrogen is very low — fertilization needed (~50kg).")
            elif n_availability < 0.65:
                parts.append("Nitrogen is low — a substantial application (~45kg) is recommended.")
            elif n_availability < 0.8:
                parts.append("Nitrogen is moderate — fertilization needed (~30kg).")
            elif n_availability < 0.9:
                parts.append("Nitrogen is adequate — a small application (~15kg) may suffice.")
            else:
                parts.append("Nitrogen is surplus — no fertilization needed.")
        else:
            if n_availability < 0.5:
                parts.append("Nitrogen is very low.")
            elif n_availability < 0.65:
                parts.append("Nitrogen is low.")
            elif n_availability < 0.8:
                parts.append("Nitrogen is moderate.")
            elif n_availability < 0.9:
                parts.append("Nitrogen is adequate.")
            else:
                parts.append("Nitrogen is surplus.")
    elif 0.15 <= dvs < 0.20:
        parts.append("First fertilization window approaching soon.")
    elif 0.42 <= dvs < 0.50:
        parts.append("Second fertilization window approaching soon.")

    # ── Weather ──
    weather_parts = []
    if forecast_rain_3d > 0.05:
        weather_parts.append(f"{forecast_rain_3d:.1f} cm rain expected next 3 days")
    else:
        weather_parts.append("No significant rain forecast next 3 days")

    if forecast_rain_7d > 0.1:
        weather_parts.append(f"{forecast_rain_7d:.1f} cm over 7 days")

    parts.append(". ".join(weather_parts) + ".")

    if weather_today_tmax > 35.0 and 0.8 <= dvs < 1.2:
        alerts.append(f"Extreme heat ({weather_today_tmax:.0f}°C) during flowering — pollen sterility risk.")
    elif weather_today_tmax > 33.0 and 1.0 <= dvs < 1.5:
        alerts.append(f"High temperature ({weather_today_tmax:.0f}°C) during grain fill.")

    # ── Resources ──
    budget_pct = (budget_remaining / max(budget_total, 1.0)) * 100
    parts.append(
        f"Cumulative water: {total_water_cm:.1f} cm, N applied: {total_n_kg_ha:.1f} kg/ha. "
        f"Budget: ${budget_remaining:.0f} remaining ({budget_pct:.0f}%)."
    )

    if budget_pct < 15:
        alerts.append("Budget nearly exhausted.")

    # ── Maturity proximity ──
    if dvs >= 1.85:
        alerts.append("Crop approaching full maturity — harvest window (DVS 1.80-2.00) is open and narrowing.")
    elif dvs >= 1.80:
        alerts.append("Crop has reached maturity — optimal harvest window (DVS 1.80-2.00) is open.")
    elif dvs >= 1.50:
        if tier >= 2:
            if has_crop_report:
                alerts.append("Ripening phase — check your CROP REPORT for exact DVS. Harvest only when DVS >= 1.80.")
            elif budget_remaining >= 20:
                alerts.append("Ripening phase — DVS hidden. Use inspect_crop ($20) to confirm DVS before harvesting. Harvest only when DVS >= 1.80.")
            else:
                alerts.append("Ripening phase — DVS hidden, budget too low for inspect. Harvest when advisory says 'harvest window' is open.")
        else:
            alerts.append("Ripening phase — NOT yet in harvest window. Wait for DVS >= 1.80 before harvesting.")

    # ── Assemble ──
    text = " ".join(parts)
    if alerts:
        alert_str = " ".join(f"ALERT: {a}" for a in alerts)
        text = f"{alert_str} {text}"

    return text


# ── Weather NL ───────────────────────────────────────────────────────────

def _temp_bucket(temp: float) -> str:
    """Quantize temperature into 5°C bucket label."""
    if temp < 5:
        return "cold (below 5°C)"
    elif temp < 10:
        return "cool (5-10°C)"
    elif temp < 15:
        return "mild (10-15°C)"
    elif temp < 20:
        return "mild (15-20°C)"
    elif temp < 25:
        return "warm (20-25°C)"
    elif temp < 30:
        return "warm (25-30°C)"
    elif temp < 35:
        return "hot (30-35°C)"
    else:
        return "very hot (35°C+)"


def _rain_bucket(rain_cm: float) -> str:
    """Classify daily rain into bucket."""
    if rain_cm <= 0:
        return "no rain"
    elif rain_cm < 0.3:
        return "light rain"
    elif rain_cm < 1.0:
        return "moderate rain"
    else:
        return "heavy rain"


def weather_to_nl(forecast_days: list, tier: int) -> str:
    """Convert weather forecast list to deterministic NL string.

    tier 2: exact per-day values.
    tier 3: bucketed per-day values (5°C buckets, rain categories).
    """
    if not forecast_days:
        return "No forecast available."

    parts = []
    for i, day in enumerate(forecast_days):
        # Accept both WeatherDay objects and dicts
        if hasattr(day, "tmax"):
            tmax, tmin, rain = day.tmax, day.tmin, day.rain
        else:
            tmax, tmin, rain = day["tmax"], day["tmin"], day["rain"]

        day_num = i + 1
        if tier <= 2:
            parts.append(f"Day {day_num}: highs {tmax:.0f}°C, lows {tmin:.0f}°C, {rain:.1f}cm rain")
        else:
            parts.append(f"Day {day_num}: {_temp_bucket(tmax)}, {_rain_bucket(rain)}")

    return ". ".join(parts) + "."


# ── Inspection reports ───────────────────────────────────────────────────

def generate_soil_report(
    *,
    sm: float,
    water_deficit: bool,
    field_capacity: float,
    wilting_point: float,
    n_availability: float,
    water_stress: float,
) -> str:
    """Exact soil measurements revealed by inspect_soil action."""
    deficit_str = "significant water deficit" if water_deficit else "no water deficit"
    return (
        f"Soil analysis: moisture at {sm * 100:.1f}%, {deficit_str}. "
        f"Field capacity {field_capacity * 100:.1f}%, wilting point {wilting_point * 100:.1f}%. "
        f"Nitrogen uptake rate {n_availability:.2f}. "
        f"Water stress factor {water_stress:.2f}."
    )


def generate_crop_report(
    *,
    dvs: float,
    lai: float,
    tagp: float,
    twso: float,
    growth_stage: str,
) -> str:
    """Exact crop measurements revealed by inspect_crop action."""
    return (
        f"Crop inspection: Development stage {dvs:.3f} ({growth_stage}). "
        f"Leaf area index {lai:.2f}, total biomass {tagp:.1f} kg/ha, "
        f"grain weight {twso:.1f} kg/ha."
    )
