"""LLM-powered aero advisor for setup recommendations."""

from __future__ import annotations

import json
import os
from typing import Optional

import httpx

from backend.models.aero_config import AdvisorRequest, OptimizationResult, SimulationResult

SYSTEM_PROMPT = """You are an expert motorsport aerodynamicist specializing in formula open-wheel cars.
You analyze simulation data and provide actionable setup recommendations.

Focus on:
- Downforce vs drag trade-offs (L/D efficiency)
- Aero balance (front % downforce) and its effect on handling
- Ride height and rake sensitivity for ground effect
- Wing angle and gurney flap adjustments
- Track-specific advice (high-speed vs twisty circuits)

Be concise, technical, and specific. Reference actual numbers from the data provided.
Format your response with clear sections: Summary, Key Changes, Trade-offs, Track Recommendation."""


def _build_context(request: AdvisorRequest) -> str:
    sim = request.simulation
    f = sim.forces
    c = sim.config

    context = f"""Current Setup:
- Speed: {c.speed_kmh} km/h
- Front wing: main {c.front_wing_main.angle_deg}°, flap {c.front_wing_flap.angle_deg}° (gurney {c.front_wing_main.gurney_mm}mm)
- Rear wing: main {c.rear_wing_main.angle_deg}°, flap {c.rear_wing_flap.angle_deg}° (gurney {c.rear_wing_main.gurney_mm}mm)
- Ride height: front {c.ride_height_front_mm}mm, rear {c.ride_height_rear_mm}mm
- Rake: {c.rake_deg}°, Diffuser: {c.diffuser_angle_deg}° at {c.diffuser_ride_height_mm}mm

Simulation Results ({sim.method}):
- Total downforce: {f.downforce_total_N} N (front {f.downforce_front_N} N / rear {f.downforce_rear_N} N)
- Drag: {f.drag_N} N
- Balance: {f.balance_front_pct}% front
- L/D efficiency: {f.efficiency}
- Cl/Cd: {f.cl_total} / {f.cd_total}
- Est. lap time delta: {sim.lap_time_delta_s}s vs baseline
"""

    if request.optimization:
        opt = request.optimization
        bf = opt.best_forces
        bc = opt.best_config
        context += f"""
Optimization Results ({opt.trials_run} trials, {opt.improvement_pct}% improvement):
- Optimized front wing: main {bc.front_wing_main.angle_deg}°, flap {bc.front_wing_flap.angle_deg}°
- Optimized rear wing: main {bc.rear_wing_main.angle_deg}°, flap {bc.rear_wing_flap.angle_deg}°
- Optimized ride height: front {bc.ride_height_front_mm}mm, rear {bc.ride_height_rear_mm}mm
- Optimized rake: {bc.rake_deg}°
- Result: {bf.downforce_total_N} N DF, {bf.drag_N} N drag, {bf.efficiency} L/D, {bf.balance_front_pct}% balance
"""

    if request.question:
        context += f"\nUser question: {request.question}"

    return context


def _fallback_advice(request: AdvisorRequest) -> str:
    """Rule-based advisor when no LLM API key is configured."""
    sim = request.simulation
    f = sim.forces
    c = sim.config

    if request.question:
        q = request.question.lower()
        lines = [f"### Answer to: '{request.question}'\n"]
        if "understeer" in q or "push" in q or "front" in q or "balance" in q:
            lines.append(
                f"Your current aero balance is {f.balance_front_pct}% front. "
                "To resolve understeer, you need more front downforce to turn the car. "
                f"I suggest increasing the Front Wing Flap angle (currently {c.front_wing_flap.angle_deg}°) by 2-3° "
                f"or lowering the Front Ride Height (currently {c.ride_height_front_mm}mm) to increase ground effect suction."
            )
        elif "oversteer" in q or "loose" in q or "rear" in q or "slide" in q:
            lines.append(
                f"Your current aero balance is {f.balance_front_pct}% front ({round(100 - f.balance_front_pct, 1)}% rear). "
                "To resolve oversteer and stabilize the rear, you need more rear load. "
                f"I suggest increasing the Rear Wing Flap angle (currently {c.rear_wing_flap.angle_deg}°) by 2-4° "
                f"or slightly lowering the Rear Ride Height (currently {c.ride_height_rear_mm}mm)."
            )
        elif "drag" in q or "speed" in q or "straight" in q or "monza" in q:
            lines.append(
                f"The current setup produces {f.drag_N} N of drag. To increase top speed, you must reduce aerodynamic drag. "
                f"I recommend reducing the Rear Wing Flap angle (currently {c.rear_wing_flap.angle_deg}°) towards 10° or lower, "
                f"and lowering the Rear Wing Main plane angle (currently {c.rear_wing_main.angle_deg}°)."
            )
        elif "drs" in q:
            lines.append(
                "Drag Reduction System (DRS) flattens the rear wing flap to ~2°, which drops total aerodynamic drag "
                "by approximately 25-30%. This is critical for overtaking on straights, but sacrifices rear downforce."
            )
        elif "ride height" in q or "rake" in q or "ground" in q:
            lines.append(
                f"Your ride heights are {c.ride_height_front_mm}mm front and {c.ride_height_rear_mm}mm rear, giving a rake of {c.rake_deg}°. "
                "Lowering the front ride height maximizes front wing and floor suction. A positive rake angle expands the diffuser, "
                "increasing rear underbody downforce, but going too high can cause diffuser flow stall."
            )
        elif "monaco" in q or "slow" in q or "downforce" in q:
            lines.append(
                "For low-speed tracks like Monaco, drag is not a major penalty because average speed is low. "
                "You should run maximum downforce setup: front wing flap to 18-20° and rear wing flap to 25-30°."
            )
        else:
            lines.append(
                f"As an aerodynamicist, looking at your current setup, we have a total downforce of {f.downforce_total_N} N "
                f"with {f.balance_front_pct}% front balance. If you have handling characteristics you want to improve "
                "(such as understeer or oversteer) or want to optimize for speed, adjust the wings and ground clearance accordingly."
            )
        lines.append("\n---")
    else:
        lines = []

    lines.append("## Summary")
    if f.efficiency < 2.5:
        lines.append(
            f"Efficiency is low ({f.efficiency} L/D). Consider reducing rear wing flap angle "
            "or increasing rake to gain downforce more efficiently."
        )
    else:
        lines.append(f"Good aero efficiency at {f.efficiency} L/D.")

    lines.append("\n## Key Changes")
    if f.balance_front_pct < 40:
        lines.append("- Balance is rear-heavy. Increase front wing flap angle by 1-2° or lower front ride height.")
    elif f.balance_front_pct > 45:
        lines.append("- Balance is front-heavy. Reduce front flap angle or increase rear wing loading.")
    else:
        lines.append(f"- Balance at {f.balance_front_pct}% front is within the optimal window (40-44%).")

    if request.optimization and request.optimization.improvement_pct > 0:
        opt = request.optimization
        bc = opt.best_config
        lines.append("\n## Optimizer Recommendations")
        lines.append(f"- Front wing flap: {bc.front_wing_flap.angle_deg}° (was {sim.config.front_wing_flap.angle_deg}°)")
        lines.append(f"- Rear wing flap: {bc.rear_wing_flap.angle_deg}° (was {sim.config.rear_wing_flap.angle_deg}°)")
        lines.append(f"- Ride height: {bc.ride_height_front_mm}/{bc.ride_height_rear_mm}mm F/R")
        lines.append(f"- Expected improvement: {opt.improvement_pct}%")

    lines.append("\n## Trade-offs")
    lines.append(f"- Current drag: {f.drag_N} N — each +100N downforce costs ~{f.drag_N / max(f.downforce_total_N, 1) * 100:.0f}N drag ratio")
    lines.append(f"- Lap time impact: {sim.lap_time_delta_s:+.4f}s vs baseline")

    lines.append("\n## Track Recommendation")
    if f.downforce_total_N > 4000:
        lines.append("High downforce setup — ideal for twisty circuits (Monaco, Hungaroring).")
    elif f.downforce_total_N < 2500:
        lines.append("Low drag setup — ideal for high-speed circuits (Monza, Spa).")
    else:
        lines.append("Balanced setup — suitable for medium downforce tracks (Silverstone, Barcelona).")

    return "\n".join(lines)


async def get_advice(request: AdvisorRequest) -> dict:
    """Get AI-powered setup advice. Uses OpenAI-compatible API if configured."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    context = _build_context(request)

    if not api_key:
        return {
            "advice": _fallback_advice(request),
            "source": "rule_based",
            "model": None,
        }

    user_message = context
    if request.question:
        user_message += f"\n\nPlease answer: {request.question}"
    else:
        user_message += "\n\nProvide setup optimization advice based on this data."

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.4,
                "max_tokens": 1200,
            },
        )
        response.raise_for_status()
        data = response.json()
        advice = data["choices"][0]["message"]["content"]

    return {
        "advice": advice,
        "source": "llm",
        "model": model,
    }
