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

def _growth_stage_label(dvs: float) -> str:
    if dvs < 0.01:
        return "Pre-emergence"
    elif dvs < 0.30:
        return "Early vegetative (tillering)"
    elif dvs < 0.65:
        return "Late vegetative (stem elongation)"
    elif dvs < 1.0:
        return "Booting / heading"
    elif dvs < 1.3:
        return "Flowering (anthesis)"
    elif dvs < 1.7:
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
    parts.append(f"{stage} (DVS {dvs:.2f}).")

    # ── Soil moisture ──
    sm_pct = sm * 100
    moisture = _moisture_descriptor(sm, field_capacity, wilting_point)
    parts.append(f"Soil moisture {moisture} at {sm_pct:.0f}%.")

    if water_stress < 0.7:
        alerts.append(f"Water stress factor {water_stress:.2f} — crop transpiration limited.")

    # ── Nitrogen ──
    if n_availability < 0.45:
        alerts.append(f"Nitrogen availability low ({n_availability:.2f}).")

    # ── Weather ──
    weather_parts = []
    if forecast_rain_3d > 0.05:
        weather_parts.append(f"{forecast_rain_3d:.1f} cm rain expected next 3 days")
    else:
        weather_parts.append("No significant rain forecast next 3 days")

    if forecast_rain_7d > 0.1:
        weather_parts.append(f"{forecast_rain_7d:.1f} cm over 7 days")

    parts.append(". ".join(weather_parts) + ".")

    if weather_today_tmax > 35.0 and 0.7 < dvs < 1.3:
        alerts.append(f"Extreme heat ({weather_today_tmax:.0f}°C) during heat-sensitive stage.")
    elif weather_today_tmax > 33.0 and 1.0 <= dvs < 1.6:
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
        alerts.append("Crop approaching full maturity — harvest window narrowing.")
    elif dvs >= 1.7:
        alerts.append("Ripening phase — grain moisture declining.")

    # ── Assemble ──
    text = " ".join(parts)
    if alerts:
        alert_str = " ".join(f"ALERT: {a}" for a in alerts)
        text = f"{alert_str} {text}"

    return text
