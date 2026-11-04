"""LLM-powered aero advisor for setup recommendations."""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

from backend.models.aero_config import (
    AdvisorRequest,
    AeroForces,
    FormulaAeroConfig,
    TrackContext,
)
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

WING_KNOBS = (
    ("Front wing flap", "front_wing_flap"),
    ("Rear wing flap", "rear_wing_flap"),
    ("Rear wing main", "rear_wing_main"),
)

# Smallest effect the advisor will state as a fix for a handling goal, per currency.
BALANCE_SHIFT_MIN_PCT = 0.05
DRAG_CUT_MIN_N = 0.5
# Smallest force change that makes a knob worth calling a move at all.
MOVE_EFFECT_MIN_N = 0.5

NO_MOVE_TEXT = (
    "No one-knob flap move from this setup changes the forces the model reports by "
    f"more than {MOVE_EFFECT_MIN_N:g} N - every wing here is either at its angle "
    "bound or carries a planform this speed cannot price."
)


def _lap_s(f: AeroForces, track: TrackContext | None) -> float:
    """The shared lap proxy, weighted by the selected track when there is one."""
    return lap_time_delta_s(
        f.drag_N,
        f.downforce_total_N,
        drag_scale=track.drag_multiplier if track else 1.0,
        downforce_scale=track.downforce_multiplier if track else 1.0,
    )


def _wing_moves(
    c: FormulaAeroConfig, step_deg: float = 3.0
) -> list[tuple[str, AeroForces]]:
    """Each one-knob flap move that changes the forces, solved rather than estimated.

    Every claim the advisor makes about what to change is priced from these, so no
    section has to fall back on a threshold the model cannot reach. A knob at its
    angle bound has one direction blocked, and a wing small enough that this speed
    cannot price it returns the forces the setup already had, so those variants are
    dropped here rather than offered as a fix.
    """
    current_forces = simulate(c).forces
    moves: list[tuple[str, AeroForces]] = []
    for name, element in WING_KNOBS:
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
            forces = simulate(cfg).forces
            if (
                abs(forces.drag_N - current_forces.drag_N) < MOVE_EFFECT_MIN_N
                and abs(
                    forces.downforce_total_N - current_forces.downforce_total_N
                )
                < MOVE_EFFECT_MIN_N
            ):
                continue
            moves.append((f"{name} {current:g} to {angle:g} deg", forces))
    return moves


def _ranked_moves(
    base: AeroForces, moves: list[tuple[str, AeroForces]], track: TrackContext | None
) -> list[tuple[str, float, float, float]]:
    """Moves sorted fastest first, as (label, lap delta, downforce delta, drag delta)."""
    base_lap = _lap_s(base, track)
    scored = [
        (
            label,
            _lap_s(f, track) - base_lap,
            f.downforce_total_N - base.downforce_total_N,
            f.drag_N - base.drag_N,
        )
        for label, f in moves
    ]
    scored.sort(key=lambda s: s[1])
    return scored


def _scored_moves(
    moves: list[tuple[str, AeroForces]],
    gain: Callable[[AeroForces], float],
    minimum: float,
) -> list[tuple[float, str, AeroForces]]:
    """Moves that clear `minimum`, as (gain, label, forces) strongest first.

    A knob at its angle bound only has the other direction left, and that one can
    still land under `minimum`, so moves below it are dropped rather than offered as
    a correction the setup cannot actually make.
    """
    scored = []
    for label, forces in moves:
        value = gain(forces)
        if value > minimum:
            scored.append((value, label, forces))
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored


def _balance_correction(
    base: AeroForces, moves: list[tuple[str, AeroForces]], needed_pct: float | None
) -> tuple[str, float, float] | None:
    """The flap move that best closes a balance gap, or the biggest shifter if none is given.

    A gap is only closed by a move that goes the right way, so a target this setup
    cannot be shifted toward returns None instead of naming the least wrong move.

    Returns (label, change in front share, resulting front share).
    """
    shifts = [
        (label, f.balance_front_pct - base.balance_front_pct, f.balance_front_pct)
        for label, f in moves
    ]
    if not shifts:
        return None
    if needed_pct is None:
        return max(shifts, key=lambda s: abs(s[1]))
    closing = [s for s in shifts if s[1] * needed_pct > 0]
    if not closing:
        return None
    return min(closing, key=lambda s: abs(s[1] - needed_pct))


def _priced_track_advice(
    base: AeroForces, moves: list[tuple[str, AeroForces]], track: TrackContext | None
) -> list[str]:
    """Price the flap moves on the selected track's own lap-proxy weights.

    This replaces a string that told every low-speed question to "run maximum
    downforce: front wing flap to 18-20 deg and rear wing flap to 25-30 deg" -
    degrees no tier has ever produced, for a track the model was never shown.
    """
    if not moves:
        return [NO_MOVE_TEXT]

    scored = _ranked_moves(base, moves, track)
    best, worst = scored[0], scored[-1]
    head = (
        f"{track.name} weights the lap proxy drag x{track.drag_multiplier:g} / "
        f"downforce x{track.downforce_multiplier:g}"
        if track
        else "No track selected, so both lap-proxy terms carry weight x1"
    )
    return [
        f"{head}. The analytical tier prices these one-knob moves on those weights:",
        f"- Fastest: {best[0]} at {best[1]:+.3f} s ({best[2]:+.0f} N downforce, "
        f"{best[3]:+.0f} N drag).",
        f"- Slowest: {worst[0]} at {worst[1]:+.3f} s ({worst[2]:+.0f} N downforce, "
        f"{worst[3]:+.0f} N drag).",
        "This is the linear lap proxy, so it compares two setups of one car - "
        "it is not a lap simulation. Negative means faster than the setup you asked about.",
    ]


def _balance_advice(
    base: AeroForces,
    moves: list[tuple[str, AeroForces]],
    toward_front: bool,
) -> list[str]:
    """Price the balance correction a handling question is asking for.

    This replaces "increase the Front Wing Flap angle by 2-3°" - a magnitude the model
    never derived and never checked against its own sensitivity to that knob.
    """
    axle = "front" if toward_front else "rear"
    sign = 1.0 if toward_front else -1.0
    shifters = _scored_moves(
        moves,
        lambda m: sign * (m.balance_front_pct - base.balance_front_pct),
        BALANCE_SHIFT_MIN_PCT,
    )
    if not shifters:
        verb = "raises" if toward_front else "lowers"
        return [
            f"No one-knob flap move from this setup {verb} the front's share on the "
            f"analytical tier ({base.balance_front_pct:.1f}% front) - nothing this setup "
            f"can price moves the split toward the {axle} by at least "
            f"{BALANCE_SHIFT_MIN_PCT:g} points."
        ]

    _, label, shifted = shifters[0]
    lines = [
        f"The analytical tier prices the largest shift toward the {axle} as {label}: "
        f"{base.balance_front_pct:.1f}% -> {shifted.balance_front_pct:.1f}% front, bought "
        f"with {shifted.downforce_total_N - base.downforce_total_N:+.0f} N total downforce "
        f"and {shifted.drag_N - base.drag_N:+.0f} N drag."
    ]
    additive = max(shifters, key=lambda s: s[2].downforce_total_N)
    if additive[1] != label:
        m = additive[2]
        lines.append(
            f"If you would rather add load than move it: {additive[1]} shifts "
            f"{base.balance_front_pct:.1f}% -> {m.balance_front_pct:.1f}% front while "
            f"gaining {m.downforce_total_N - base.downforce_total_N:+.0f} N of total "
            f"downforce for {m.drag_N - base.drag_N:+.0f} N of drag."
        )
    return lines


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

    # Every "what should I change" line below is priced on the analytical tier from
    # this setup's own one-knob moves. Comparing a number with a threshold the model
    # cannot reach is how the old text called every setup "low" efficiency.
    a_base = simulate(c).forces
    moves = _wing_moves(c)
    track_priced = False

    if request.question:
        q = request.question.lower()
        lines = [f"### Answer to: '{request.question}'\n"]
        if "understeer" in q or "push" in q or "front" in q or "balance" in q:
            lines.append(
                f"Your current aero balance is {f.balance_front_pct}% front on the "
                f"{sim.method} tier; the analytical tier the moves below are priced on "
                f"puts this setup at {a_base.balance_front_pct:.1f}% front. To resolve "
                "understeer, the front needs a larger share of that load."
            )
            lines.extend(_balance_advice(a_base, moves, toward_front=True))
            lines.append(
                f"Lowering the front ride height (currently {c.ride_height_front_mm}mm) is "
                "the other lever the model carries, but it works through ground-effect "
                "suction rather than a wing angle, so it is not in those moves."
            )
        elif "oversteer" in q or "loose" in q or "rear" in q or "slide" in q:
            lines.append(
                f"Your current aero balance is {f.balance_front_pct}% front "
                f"({round(100 - f.balance_front_pct, 1)}% rear) on the {sim.method} tier; "
                f"the analytical tier the moves below are priced on puts this setup at "
                f"{a_base.balance_front_pct:.1f}% front. To resolve oversteer, the rear "
                "needs a larger share of that load."
            )
            lines.extend(_balance_advice(a_base, moves, toward_front=False))
            lines.append(
                f"Lowering the rear ride height (currently {c.ride_height_rear_mm}mm) is "
                "the other lever the model carries, and it changes the diffuser and ground "
                "effect together, so it is not in those moves."
            )
        elif "drag" in q or "speed" in q or "straight" in q or "monza" in q:
            cuts = _scored_moves(
                moves,
                lambda m: a_base.drag_N - m.drag_N,
                DRAG_CUT_MIN_N,
            )
            if not cuts:
                lines.append(
                    f"The current setup produces {f.drag_N} N of drag on the {sim.method} "
                    f"tier ({a_base.drag_N:.0f} N on the analytical tier). No one-knob flap "
                    f"move cuts it there - nothing this setup can price takes "
                    f"{DRAG_CUT_MIN_N:g} N off that figure."
                )
            else:
                _, label, cut = cuts[0]
                lines.append(
                    f"The current setup produces {f.drag_N} N of drag on the {sim.method} "
                    f"tier ({a_base.drag_N:.0f} N on the analytical tier, where the moves "
                    "below are priced). To increase top speed you must reduce it, and the "
                    f"largest cut the model solves is {label}: {a_base.drag_N:.0f} N -> "
                    f"{cut.drag_N:.0f} N, paid for with "
                    f"{cut.downforce_total_N - a_base.downforce_total_N:+.0f} N of total "
                    "downforce."
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
            lines.extend(_priced_track_advice(a_base, moves, request.track))
            track_priced = True
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
    else:
        lines.append(
            f"Efficiency is {f.efficiency} L/D on the {sim.method} tier. The model publishes no "
            "L/D target to grade that against, so it is a ratio, not a verdict."
        )
        best_ld = max(moves, key=lambda m: m[1].efficiency) if moves else None
        if best_ld is None:
            lines.append(f"- {NO_MOVE_TEXT}")
        elif best_ld[1].efficiency > a_base.efficiency:
            lines.append(
                f"- On the analytical tier ({a_base.efficiency} L/D for this setup), the best "
                f"one-knob move for efficiency is {best_ld[0]} at {best_ld[1].efficiency:.2f} L/D "
                f"({best_ld[1].downforce_total_N - a_base.downforce_total_N:+.0f} N downforce, "
                f"{best_ld[1].drag_N - a_base.drag_N:+.0f} N drag)."
            )
        else:
            lines.append(
                f"- No one-knob flap move beats the analytical tier's {a_base.efficiency} L/D "
                "for this setup."
            )

    lines.append("\n## Key Changes")
    target = opt.target_balance_pct if opt is not None else None
    # The gap has to be measured on the tier the correction is priced on, or the
    # advisor reports a move as closing a gap a different solver measured.
    a_balance = a_base.balance_front_pct
    other_tier = (
        ""
        if sim.method == "analytical"
        else f" ({f.balance_front_pct}% on the {sim.method} tier you simulated)"
    )
    if target is None:
        lines.append(
            f"- Balance is {a_balance:.1f}% front on the analytical tier{other_tier}. No "
            "balance target was given, so that is where the load sits rather than a "
            "verdict; the optimizer's target is the only balance number this model "
            "compares against."
        )
        needed = None
    elif abs(target - a_balance) < 0.05:
        lines.append(
            f"- Balance is {a_balance:.1f}% front on the analytical tier{other_tier}, at "
            f"the objective's {target}% target."
        )
        needed = 0.0
    else:
        gap = target - a_balance
        lines.append(
            f"- Balance is {a_balance:.1f}% front on the analytical tier{other_tier} "
            f"against the objective's {target}% target: {abs(gap):.1f} points "
            f"{'short' if gap > 0 else 'over'} at the front."
        )
        needed = gap

    correction = _balance_correction(a_base, moves, needed)
    if correction is not None:
        label, shift, resulting = correction
        why = "closes that gap best" if needed else "moves the split most"
        lines.append(
            f"- {label} {why} on the analytical tier: "
            f"{a_base.balance_front_pct:.1f}% -> {resulting:.1f}% front ({shift:+.1f} points)."
        )
    elif needed:
        lines.append(
            f"- No one-knob flap move on the analytical tier covers the {abs(needed):.1f} "
            f"points that separate its {a_balance:.1f}% front from the objective's "
            f"{target}% target."
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
    if track_priced:
        lines.append(
            "Already priced above - that answer is the recommendation for the selected circuit."
        )
    elif request.track is None:
        lines.append(
            "No track is selected, so both lap-proxy terms carry weight x1 and there is no "
            "circuit for this setup to be recommended to. Pick a track profile for a "
            "weighted answer."
        )
    elif not moves:
        lines.append(
            f"{request.track.name}'s weights cannot be answered from this setup. "
            f"{NO_MOVE_TEXT}"
        )
    else:
        scored = _ranked_moves(a_base, moves, request.track)
        label, dl, ddf, ddrag = scored[0]
        weights = (
            f"drag x{request.track.drag_multiplier:g} / "
            f"downforce x{request.track.downforce_multiplier:g}"
        )
        if dl >= 0:
            lines.append(
                f"On {request.track.name}'s weights ({weights}), no one-knob flap move on the "
                f"analytical tier is faster than this setup; the least costly is {label} at "
                f"{dl:+.3f} s ({ddf:+.0f} N downforce, {ddrag:+.0f} N drag)."
            )
        else:
            lines.append(
                f"On {request.track.name}'s weights ({weights}), the analytical tier's fastest "
                f"one-knob move is {label} at {dl:+.3f} s ({ddf:+.0f} N downforce, "
                f"{ddrag:+.0f} N drag). The lap proxy is a linear sensitivity between two "
                "setups of one car, not a lap simulation."
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
