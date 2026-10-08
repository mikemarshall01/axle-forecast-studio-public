"""Scalar reference replay of the physics kernel, for parity tests only.

What this is: one EV, one world, one half-hour at a time, with pandas
timestamps and plain floats.  It is deliberately slow and obvious, so the
vectorised kernel in the model package can be checked against an independent
implementation of the same event order (travel, occupancy, home connection,
home charging).  It came from the removed ``compact_forecast`` module (task
M4-layout) and keeps only what the parity tests read.

What it leaves out: public top-ups (decision 0004 item 32).  Parity tests run
the kernel with a zero top-up threshold and keep every battery above empty,
so no top-up happens; a leg this oracle cannot serve from stock raises
instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.settings import RunSettings

_PATH_IDS = ("normal", "selected")
_TRAVEL_FLOWS = tuple(
    f"{kind}_{leg}_travel_battery_kwh"
    for leg in ("outbound", "return")
    for kind in ("requested", "served", "unserved")
)


@dataclass(frozen=True)
class ScalarResult:
    """Per-EV interval rows, duplicated as identical normal and selected paths."""

    intervals: pd.DataFrame


@dataclass(frozen=True)
class _Journey:
    departure: pd.Timestamp
    outbound_end: pd.Timestamp
    return_start: pd.Timestamp
    return_end: pd.Timestamp
    outbound_energy_kwh: float
    return_energy_kwh: float


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def simulate_explicit_journey_intervals(
    settings: RunSettings,
    units: pd.DataFrame,
    *,
    drives_today: np.ndarray,
    outbound_miles: np.ndarray,
    return_miles: np.ndarray,
    trip_departure_utc: np.ndarray,
    destination_dwell_seconds: np.ndarray,
    drive_speed_mph: np.ndarray,
    connection_session_accepted: np.ndarray,
    connection_start_utc: np.ndarray,
    connection_end_utc: np.ndarray,
    evaluation_effective_efficiency_miles_per_battery_kwh: np.ndarray | None = None,
    **_unused: np.ndarray,
) -> ScalarResult:
    """Replay every EV of every world slot by slot; see the module docstring."""

    boundaries = [
        pd.Timestamp(value)
        for value in utc_half_hour_boundaries(
            settings.start_local_date, settings.warmup_days, settings.study_days
        )
    ]
    study_start = settings.warmup_days * 48
    rows: list[dict[str, object]] = []
    for world in range(settings.evaluation_world_count):
        for ev, unit in enumerate(units.itertuples()):
            capacity = float(unit.physical_capacity_kwh)
            target_stock = capacity * float(unit.preferred_target_soc_fraction)
            stock = capacity * settings.opening_soc_fraction
            journeys = []
            for day in np.flatnonzero(drives_today[world, :, ev]):
                speed = float(drive_speed_mph[world, day, ev])
                efficiency = (
                    float(unit.efficiency_miles_per_battery_kwh)
                    if evaluation_effective_efficiency_miles_per_battery_kwh is None
                    else float(
                        evaluation_effective_efficiency_miles_per_battery_kwh[world, day, ev]
                    )
                )
                departure = _utc(trip_departure_utc[world, day, ev])
                outbound = float(outbound_miles[world, day, ev])
                returning = float(return_miles[world, day, ev])
                outbound_end = departure + pd.Timedelta(seconds=3600.0 * outbound / speed)
                return_start = outbound_end + pd.Timedelta(
                    seconds=float(destination_dwell_seconds[world, day, ev])
                )
                journeys.append(
                    _Journey(
                        departure=departure,
                        outbound_end=outbound_end,
                        return_start=return_start,
                        return_end=return_start + pd.Timedelta(seconds=3600.0 * returning / speed),
                        outbound_energy_kwh=outbound / efficiency,
                        return_energy_kwh=returning / efficiency,
                    )
                )
            sessions = [
                (
                    _utc(connection_start_utc[world, day, ev]),
                    _utc(connection_end_utc[world, day, ev]),
                )
                for day in np.flatnonzero(connection_session_accepted[world, :, ev])
            ]

            for slot, start in enumerate(boundaries[:-1]):
                end = boundaries[slot + 1]
                opening = stock
                overlapping = [j for j in journeys if j.departure < end and j.return_end > start]
                flows = dict.fromkeys(_TRAVEL_FLOWS, 0.0)
                # Outbound leg then return leg, each drawing energy in
                # proportion to the time driven inside this slot.
                for journey in overlapping:
                    for leg, leg_start, leg_end, energy in (
                        (
                            "outbound",
                            journey.departure,
                            journey.outbound_end,
                            journey.outbound_energy_kwh,
                        ),
                        (
                            "return",
                            journey.return_start,
                            journey.return_end,
                            journey.return_energy_kwh,
                        ),
                    ):
                        overlap = _overlap_seconds(start, end, leg_start, leg_end)
                        if overlap <= 0.0:
                            continue
                        requested = energy * overlap / (leg_end - leg_start).total_seconds()
                        if requested > stock:
                            raise ValueError("oracle case runs a battery empty; it has no top-ups")
                        stock -= requested
                        flows[f"requested_{leg}_travel_battery_kwh"] += requested
                        flows[f"served_{leg}_travel_battery_kwh"] += requested

                occupancy = _occupancy(overlapping, start, end, sessions)
                connected_full_slot = occupancy["home_connected_fraction"] == 1.0
                home_import = 0.0
                if connected_full_slot and stock < target_stock:
                    home_import = min(
                        float(unit.home_charger_limit_kw) * 0.5,
                        (target_stock - stock) / settings.home_charge_efficiency,
                    )
                home_added = home_import * settings.home_charge_efficiency
                stock += home_added
                served = (
                    flows["served_outbound_travel_battery_kwh"]
                    + flows["served_return_travel_battery_kwh"]
                )
                residual = stock - (opening + home_added - served)
                if slot < study_start:
                    continue
                rows.append(
                    {
                        "world_id": world,
                        "unit_id": unit.unit_id,
                        "interval_start_utc": start,
                        "interval_end_utc": end,
                        **occupancy,
                        "public_charging_fraction": 0.0,
                        "home_connected_full_slot": connected_full_slot,
                        "opening_battery_kwh": opening,
                        "closing_battery_kwh": stock,
                        "physical_capacity_kwh": capacity,
                        "realised_grid_kw": home_import / 0.5,
                        "home_grid_import_kwh": home_import,
                        "home_battery_added_kwh": home_added,
                        "public_grid_import_kwh": 0.0,
                        "public_battery_added_kwh": 0.0,
                        **flows,
                        "v2g_battery_removed_kwh": 0.0,
                        "conservation_residual_kwh": residual,
                    }
                )
    base = pd.DataFrame(rows)
    paired = [base.assign(path_id=path_id) for path_id in _PATH_IDS]
    return ScalarResult(intervals=pd.concat(paired, ignore_index=True))


def _overlap_seconds(
    start: pd.Timestamp, end: pd.Timestamp, event_start: pd.Timestamp, event_end: pd.Timestamp
) -> float:
    return max(0.0, (min(end, event_end) - max(start, event_start)).total_seconds())


def _occupancy(
    journeys: list[_Journey],
    start: pd.Timestamp,
    end: pd.Timestamp,
    sessions: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float]:
    """Split the slot into away, home unplugged and home connected time.

    Connection counts only up to the first departure in the slot, because
    leaving unplugs the EV.
    """

    slot_seconds = (end - start).total_seconds()
    driving = sum(
        _overlap_seconds(start, end, j.departure, j.outbound_end)
        + _overlap_seconds(start, end, j.return_start, j.return_end)
        for j in journeys
    )
    parked = sum(_overlap_seconds(start, end, j.outbound_end, j.return_start) for j in journeys)
    away = min(slot_seconds, driving + parked)
    home = slot_seconds - away
    connected_until = min([end, *(j.departure for j in journeys)])
    connected = min(
        home,
        sum(_overlap_seconds(start, connected_until, s, e) for s, e in sessions),
    )
    return {
        "away_on_trip_fraction": away / slot_seconds,
        "home_unplugged_fraction": (home - connected) / slot_seconds,
        "home_connected_fraction": connected / slot_seconds,
        "driving_fraction": driving / slot_seconds,
        "parked_away_fraction": parked / slot_seconds,
    }
