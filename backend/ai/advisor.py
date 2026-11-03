"""LLM-powered aero advisor for setup recommendations."""

from __future__ import annotations

import os

import httpx

from backend.models.aero_config import AdvisorRequest, FormulaAeroConfig, TrackContext
from backend.simulation.analytical import (
    DRS_FLAP_ANGLE_DEG,
    diffuser_throat_height_m,
    effective_rear_flap,
    geometric_rake_deg,
    lap_time_delta_s,
    simulate,
)

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

    throat_mm = diffuser_throat_height_m(c) * 1000.0
    throat_note = (
        f"{throat_mm:.2f}mm as the model runs it (raised to the floor line)"
        if throat_mm > c.diffuser_ride_height_mm + 1e-9
        else f"{throat_mm:.2f}mm"
    )

    context = f"""Baseline Setup:
- Speed: {c.speed_kmh} km/h
- Front wing: main {c.front_wing_main.angle_deg}°, flap {c.front_wing_flap.angle_deg}° (gurney {c.front_wing_main.gurney_mm}mm)
- Rear wing: main {c.rear_wing_main.angle_deg}°, flap {c.rear_wing_flap.angle_deg}° (gurney {c.rear_wing_main.gurney_mm}mm)
- Rear flap as the model ran it: {effective_rear_flap(c).angle_deg}°
- DRS: {'open' if c.drs_active else 'closed'}
- Ride height: front {c.ride_height_front_mm}mm, rear {c.ride_height_rear_mm}mm
- Rake model term: {c.rake_deg}° (suction multiplier, not a body angle); body rake implied by those ride heights: {geometric_rake_deg(c):.2f}°
- Diffuser: {c.diffuser_angle_deg}° throat {c.diffuser_ride_height_mm}mm requested -> {throat_note}

Baseline Simulation Results ({sim.method}):
- Total downforce: {f.downforce_total_N} N (front {f.downforce_front_N} N / rear {f.downforce_rear_N} N)
- Drag: {f.drag_N} N
- Balance: {f.balance_front_pct}% front
- L/D efficiency: {f.efficiency}
- Cl/Cd: {f.cl_total} / {f.cd_total}
- Fixed-baseline lap proxy (not a track prediction): {sim.lap_time_delta_s}s
"""

    if request.optimization:
        opt = request.optimization
        bf = opt.best_forces
        bc = opt.best_config
        context += f"""
Optimization Results ({opt.trials_run} trials):
- Optimized front wing: main {bc.front_wing_main.angle_deg}°, flap {bc.front_wing_flap.angle_deg}°
- Optimized rear wing: main {bc.rear_wing_main.angle_deg}°, flap {bc.rear_wing_flap.angle_deg}°
- Optimized ride height: front {bc.ride_height_front_mm}mm, rear {bc.ride_height_rear_mm}mm
- Optimized rake model term: {bc.rake_deg}°; body rake implied by those ride heights: {geometric_rake_deg(bc):.2f}°
- Optimized result: {bf.downforce_total_N} N DF, {bf.drag_N} N drag, {bf.efficiency} L/D, {bf.balance_front_pct}% balance
- Change vs baseline: {bf.downforce_total_N - f.downforce_total_N:+.1f} N downforce, {bf.drag_N - f.drag_N:+.1f} N drag
- Objective score improvement: {opt.improvement_pct}% (normalized score, not a physical or lap-time percentage)
- Balance target: {opt.target_balance_pct}%; sampled achievable range: {opt.achievable_balance_pct}; target reachable: {opt.target_reachable}
"""

    if request.track:
        context += (
            f"\nSelected track: {request.track.name}"
            f"{f' ({request.track.description})' if request.track.description else ''}. "
            f"Lap proxy weighting: drag x{request.track.drag_multiplier:g}, "
            f"downforce x{request.track.downforce_multiplier:g}."
        )

    if request.question:
        context += f"\nUser question: {request.question}"

    return context


WING_ANGLE_MAX_DEG = 35.0
WING_ANGLE_MIN_DEG = -5.0


def _priced_track_advice(
    c: FormulaAeroConfig, track: TrackContext | None, step_deg: float = 3.0
) -> list[str]:
    """Price one-knob flap moves on the selected track's own lap-proxy weights.

    This replaces a string that told every low-speed question to "run maximum
    downforce: front wing flap to 18-20 deg and rear wing flap to 25-30 deg" -
    degrees no tier has ever produced, for a track the model was never shown.
    """
    drag_scale = track.drag_multiplier if track else 1.0
    df_scale = track.downforce_multiplier if track else 1.0

    def lap(f) -> float:
        return lap_time_delta_s(
            f.drag_N,
            f.downforce_total_N,
            drag_scale=drag_scale,
            downforce_scale=df_scale,
        )

    base = simulate(c).forces
    base_lap = lap(base)

    moves: list[tuple[str, float, float, float]] = []
    for name, element in (
        ("Front wing flap", "front_wing_flap"),
        ("Rear wing flap", "rear_wing_flap"),
        ("Rear wing main", "rear_wing_main"),
    ):
        current = getattr(c, element).angle_deg
        for delta in (-step_deg, step_deg):
            angle = min(WING_ANGLE_MAX_DEG, max(WING_ANGLE_MIN_DEG, current + delta))
            if angle == current:
                continue
            cfg = c.model_copy(
                update={
                    element: getattr(c, element).model_copy(update={"angle_deg": angle})
                }
            )
            f = simulate(cfg).forces
            moves.append(
                (
                    f"{name} {current:g} to {angle:g} deg",
                    lap(f) - base_lap,
                    f.downforce_total_N - base.downforce_total_N,
                    f.drag_N - base.drag_N,
                )
            )

    if not moves:
        return ["No wing angle on this setup can move further within the model's bounds."]

    best = min(moves, key=lambda m: m[1])
    worst = max(moves, key=lambda m: m[1])
    head = (
        f"{track.name} weights the lap proxy drag x{drag_scale:g} / downforce x{df_scale:g}"
        if track
        else "No track selected, so both lap-proxy terms carry weight x1"
    )
    return [
        f"{head}. On those weights the model charges these one-knob moves:",
        f"- Fastest: {best[0]} at {best[1]:+.3f} s ({best[2]:+.0f} N downforce, "
        f"{best[3]:+.0f} N drag).",
        f"- Slowest: {worst[0]} at {worst[1]:+.3f} s ({worst[2]:+.0f} N downforce, "
        f"{worst[3]:+.0f} N drag).",
        "This is the linear lap proxy, so it compares two setups of one car - "
        "it is not a lap simulation. Negative means faster than the setup you asked about.",
    ]


def _fallback_advice(request: AdvisorRequest) -> str:
    """Rule-based advisor when no LLM API key is configured."""
    sim = request.simulation
    base_f = sim.forces
    base_c = sim.config
    opt = request.optimization

    # When an optimization is present, the advice must describe the optimized
    # setup. Describing the baseline while showing optimized recommendations was
    # the source of contradictory efficiency/balance/lap text in the UI.
    f = opt.best_forces if opt is not None else base_f
    c = opt.best_config if opt is not None else base_c

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
            closed = simulate(c.model_copy(update={"drs_active": False})).forces
            opened = simulate(c.model_copy(update={"drs_active": True})).forces
            if c.rear_wing_flap.angle_deg <= DRS_FLAP_ANGLE_DEG:
                lines.append(
                    f"DRS folds the rear flap to {DRS_FLAP_ANGLE_DEG}°, but this setup already runs it at "
                    f"{c.rear_wing_flap.angle_deg}°, so opening DRS changes nothing in the model. "
                    f"Drag {closed.drag_N:.0f} N, rear downforce {closed.downforce_rear_N:.0f} N either way."
                )
            else:
                drag_pct = (
                    (opened.drag_N - closed.drag_N) / closed.drag_N * 100.0
                    if closed.drag_N
                    else 0.0
                )
                lines.append(
                    f"DRS folds the rear flap from {c.rear_wing_flap.angle_deg}° to {DRS_FLAP_ANGLE_DEG}°. "
                    f"Analytical model: drag {closed.drag_N:.0f} N -> {opened.drag_N:.0f} N ({drag_pct:.1f}%), "
                    f"rear downforce {closed.downforce_rear_N:.0f} N -> {opened.downforce_rear_N:.0f} N, "
                    f"balance {closed.balance_front_pct:.1f}% -> {opened.balance_front_pct:.1f}% front. "
                    f"Drag falls less than rear downforce, which is why the rear of the car gets lighter. "
                    f"Your DRS is currently {'open' if c.drs_active else 'closed'}."
                )
        elif "ride height" in q or "rake" in q or "ground" in q:
            lines.append(
                f"Your ride heights are {c.ride_height_front_mm}mm front and {c.ride_height_rear_mm}mm rear, "
                f"so the car sits at {geometric_rake_deg(c):.2f}° of rake. The Rake slider value ({c.rake_deg}°) is a "
                "separate model term: it multiplies underbody suction and does not tilt the car. "
                "Lowering the front ride height maximizes front wing and floor suction. Raising the rear expands the diffuser, "
                "increasing rear underbody downforce, but going too high can cause diffuser flow stall."
            )
        elif (
            "monaco" in q
            or "slow" in q
            or "downforce" in q
            or "track" in q
            or (request.track is not None and request.track.name.lower() in q)
        ):
            lines.extend(_priced_track_advice(c, request.track))
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
    if opt is not None:
        lines.append(
            f"Optimized efficiency is {f.efficiency} L/D (baseline {base_f.efficiency} L/D), "
            f"with {f.downforce_total_N:.1f} N downforce (baseline {base_f.downforce_total_N:.1f} N) "
            f"and {f.drag_N:.1f} N drag (baseline {base_f.drag_N:.1f} N)."
        )
    elif f.efficiency < 2.5:
        lines.append(
            f"Efficiency is low ({f.efficiency} L/D). Consider reducing rear wing flap angle "
            "or increasing rake to gain downforce more efficiently."
        )
    else:
        lines.append(f"Good aero efficiency at {f.efficiency} L/D.")

    lines.append("\n## Key Changes")
    if f.balance_front_pct < 40:
        lines.append(
            "- Balance is rear-heavy. Increase front wing flap angle by 1-2° or lower front ride height."
        )
    elif f.balance_front_pct > 45:
        lines.append(
            "- Balance is front-heavy. Reduce front flap angle or increase rear wing loading."
        )
    else:
        lines.append(
            f"- Balance at {f.balance_front_pct}% front is within the optimal window (40-44%)."
        )

    if opt is not None and opt.improvement_pct > 0:
        bc = opt.best_config
        lines.append("\n## Optimizer Recommendations")
        lines.append(
            f"- Front wing flap: {bc.front_wing_flap.angle_deg:.1f}° (was {base_c.front_wing_flap.angle_deg:.1f}°)"
        )
        lines.append(
            f"- Rear wing flap: {bc.rear_wing_flap.angle_deg:.1f}° (was {base_c.rear_wing_flap.angle_deg:.1f}°)"
        )
        lines.append(
            f"- Ride height: {bc.ride_height_front_mm:.1f}/{bc.ride_height_rear_mm:.1f}mm F/R"
        )
        lines.append(
            f"- Objective score improvement: {opt.improvement_pct}% "
            "(normalized score, not a physical or lap-time percentage)"
        )
        if opt.target_reachable is False and len(opt.achievable_balance_pct) == 2:
            lines.append(
                f"- Balance target {opt.target_balance_pct}% is outside the sampled range "
                f"{opt.achievable_balance_pct[0]}-{opt.achievable_balance_pct[1]}%."
            )

    lines.append("\n## Trade-offs")
    lines.append(
        f"- {'Optimized' if opt is not None else 'Current'} drag: {f.drag_N:.1f} N; "
        f"downforce/drag ratio {f.downforce_total_N / max(f.drag_N, 1):.2f}"
    )
    if opt is not None:
        lines.append(
            f"- Change vs baseline: {f.downforce_total_N - base_f.downforce_total_N:+.1f} N downforce, "
            f"{f.drag_N - base_f.drag_N:+.1f} N drag."
        )
        lines.append(
            "- The lap-time proxy is shown in the metric panel relative to the same-fidelity "
            "baseline; the fixed backend proxy is not an optimized lap-time prediction."
        )
    else:
        lines.append(
            f"- Lap-time proxy: {sim.lap_time_delta_s:+.4f}s vs the fixed backend baseline "
            "(not a track-specific prediction)."
        )

    lines.append("\n## Track Recommendation")
    if f.downforce_total_N > 4000:
        lines.append("High downforce setup — ideal for twisty circuits (Monaco, Hungaroring).")
    elif f.downforce_total_N < 2500:
        lines.append("Low drag setup — ideal for high-speed circuits (Monza, Spa).")
    else:
        lines.append(
            "Balanced setup — suitable for medium downforce tracks (Silverstone, Barcelona)."
        )

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
