"""Quasi-steady vehicle model for the lap simulation.

Point mass with:

* load-sensitive tyre friction per axle (``mu`` falls as vertical load rises),
* longitudinal weight transfer from acceleration and braking,
* lateral weight transfer split front/rear by roll stiffness, which reduces
  axle grip through the same load sensitivity (inside wheel can lift),
* aerodynamic downforce and drag from an external map, with the centre of
  pressure split setting the front/rear load distribution,
* power-limited and traction-limited acceleration, and grip-limited braking.

Not modelled: transient tyre dynamics, suspension compliance, differentials,
a driver model, fuel mass, DRS zones, brake lock-up, or aero map variation as
ride height changes over a lap. Those omissions are reported with every result.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.simulation.lap.aero_map import AeroPoint

G = 9.80665


@dataclass(frozen=True)
class Vehicle:
    """Point-mass parameters for a formula car."""

    mass_kg: float = 798.0
    wheelbase_m: float = 3.35
    cg_height_m: float = 0.30
    front_static_pct: float = 0.46
    track_width_m: float = 1.60
    power_kw: float = 750.0
    drivetrain_efficiency: float = 0.90
    tyre_mu: float = 1.60
    tyre_load_sensitivity: float = 0.10
    tyre_reference_load_N: float | None = None
    rolling_resistance: float = 0.012
    roll_stiffness_front_pct: float = 0.55
    max_speed_ms: float = 100.0
    max_tractive_force_N: float | None = None
    rear_wheel_drive: bool = True

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if self.wheelbase_m <= 0 or self.track_width_m <= 0:
            raise ValueError("wheelbase and track width must be positive")
        if not 0.0 < self.front_static_pct < 1.0:
            raise ValueError("front_static_pct must be between 0 and 1")
        if not 0.0 < self.roll_stiffness_front_pct < 1.0:
            raise ValueError("roll_stiffness_front_pct must be between 0 and 1")
        if self.tyre_mu <= 0:
            raise ValueError("tyre_mu must be positive")
        if self.tyre_load_sensitivity < 0:
            raise ValueError("tyre_load_sensitivity must be non-negative")
        if self.power_kw < 0 or self.max_speed_ms <= 0:
            raise ValueError("power_kw must be non-negative and max_speed_ms positive")
        ref_load = self.tyre_reference_load_N
        if ref_load is None:
            ref_load = self.mass_kg * G / 4.0
            object.__setattr__(self, "tyre_reference_load_N", ref_load)
        if ref_load <= 0:
            raise ValueError("tyre_reference_load_N must be positive")

    @property
    def weight_N(self) -> float:
        return self.mass_kg * G

    @property
    def static_front_N(self) -> float:
        return self.weight_N * self.front_static_pct

    @property
    def static_rear_N(self) -> float:
        return self.weight_N * (1.0 - self.front_static_pct)

    @property
    def power_W(self) -> float:
        return self.power_kw * 1000.0 * self.drivetrain_efficiency

    @property
    def reference_load_N(self) -> float:
        """Per-tyre reference load; ``__post_init__`` guarantees it is set."""
        return float(self.tyre_reference_load_N or self.mass_kg * G / 4.0)

    def tyre_mu_at(self, fz_tyre_N: float) -> float:
        """Load-sensitive friction coefficient for one tyre."""
        if fz_tyre_N <= 0.0:
            return 0.0
        ratio = fz_tyre_N / self.reference_load_N
        # float ** float is typed Any by mypy; the arithmetic is unchanged.
        return float(self.tyre_mu * max(0.35, ratio ** (-self.tyre_load_sensitivity)))

    def _axle_capacity(self, fz_axle_N: float, lateral_transfer_N: float) -> float:
        """Combined lateral capacity of one axle's two tyres (N)."""
        inner = max(0.0, 0.5 * (fz_axle_N - lateral_transfer_N))
        outer = max(0.0, 0.5 * (fz_axle_N + lateral_transfer_N))
        return self.tyre_mu_at(inner) * inner + self.tyre_mu_at(outer) * outer

    def axle_loads(self, ax: float, aero: AeroPoint) -> tuple[float, float]:
        """Front and rear axle vertical loads (N) at longitudinal accel ``ax``."""
        transfer = self.mass_kg * ax * self.cg_height_m / self.wheelbase_m
        front = self.static_front_N + aero.downforce_N * aero.front_share - transfer
        rear = self.static_rear_N + aero.downforce_N * (1.0 - aero.front_share) + transfer
        return max(front, 0.0), max(rear, 0.0)

    def lateral_limit(self, speed_ms: float, aero: AeroPoint, ax: float = 0.0) -> float:
        """Maximum lateral acceleration (m/s^2) at the current speed.

        Solved as a fixed point because lateral load transfer depends on the
        lateral acceleration it is limiting. Load sensitivity makes the
        capacity fall as transfer grows, so the iteration is monotone.
        """
        ay = 0.0
        for _ in range(10):
            fz_front, fz_rear = self.axle_loads(ax, aero)
            transfer = self.mass_kg * ay * self.cg_height_m / self.track_width_m
            capacity = self._axle_capacity(
                fz_front, transfer * self.roll_stiffness_front_pct
            ) + self._axle_capacity(
                fz_rear, transfer * (1.0 - self.roll_stiffness_front_pct)
            )
            new_ay = capacity / self.mass_kg
            if abs(new_ay - ay) < 1e-9:
                return max(new_ay, 0.0)
            ay = new_ay
        return max(ay, 0.0)

    def resistance_N(self, aero: AeroPoint) -> float:
        return aero.drag_N + self.rolling_resistance * (self.weight_N + aero.downforce_N)

    def _traction_force(self, ax: float, aero: AeroPoint) -> float:
        fz_front, fz_rear = self.axle_loads(ax, aero)
        if self.rear_wheel_drive:
            capacity = self._axle_capacity(fz_rear, 0.0)
        else:
            capacity = self._axle_capacity(fz_front, 0.0) + self._axle_capacity(
                fz_rear, 0.0
            )
        if self.max_tractive_force_N is not None:
            capacity = min(capacity, self.max_tractive_force_N)
        return capacity

    def _accel_raw(self, speed_ms: float, aero: AeroPoint) -> float:
        """Longitudinal acceleration before the g-g ellipse (may be negative)."""
        speed = max(speed_ms, 0.5)
        engine_force = self.power_W / speed
        ax = 0.0
        for _ in range(6):
            force = min(engine_force, self._traction_force(ax, aero))
            new_ax = (force - self.resistance_N(aero)) / self.mass_kg
            if abs(new_ax - ax) < 1e-9:
                return new_ax
            ax = new_ax
        return ax

    def accel_limit(self, speed_ms: float, aero: AeroPoint) -> float:
        """Grip- and power-limited forward acceleration (m/s^2, >= 0)."""
        return max(0.0, self._accel_raw(speed_ms, aero))

    def brake_limit(self, speed_ms: float, aero: AeroPoint) -> float:
        """Grip-limited deceleration (m/s^2, >= 0, returned as a magnitude)."""
        ax = 0.0
        for _ in range(6):
            fz_front, fz_rear = self.axle_loads(ax, aero)
            capacity = self._axle_capacity(fz_front, 0.0) + self._axle_capacity(
                fz_rear, 0.0
            )
            new_ax = -(capacity + self.resistance_N(aero)) / self.mass_kg
            if abs(new_ax - ax) < 1e-9:
                return max(-new_ax, 0.0)
            ax = new_ax
        return max(-ax, 0.0)

    def top_speed(self, aero_map) -> float:
        """Speed where tractive force balances drag and rolling resistance."""
        low, high = 0.0, self.max_speed_ms
        if self._accel_raw(high, aero_map.at(high)) > 0.0:
            return high
        for _ in range(50):
            mid = 0.5 * (low + high)
            if self._accel_raw(mid, aero_map.at(mid)) > 0.0:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)
