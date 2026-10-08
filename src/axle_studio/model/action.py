"""The Axle action: price-optimised smart home charging, and its cost effect.

What this module owns: the charging policy of the selected path (decision
0004 item 38, which replaced the 18:00 import cap of items 2, 8, 25 and 30)
and the illustrative per-world cost effect (items 4, 5 and 13).

The policy.  The normal path charges as soon as the EV is plugged in.  On the
selected path a smart charger plans each home session when it starts, using
only what it knows at that moment:

- the EV's battery stock at plug-in and its preferred target SoC;
- the plug-in half-hour (the first half-hour the EV is plugged in for the
  whole slot, because only whole slots can charge);
- the expected departure: the cohort's typical home departure clock time
  for that date (weekday or weekend window, decision 0004 item 42) minus a
  safety margin (``assumptions.ACTION["departure_margin_hours"]``, an editable
  illustrative assumption, default two hours since decision 0004 item 51), on
  the first London date where that instant is at least one half-hour after
  plug-in (so an evening plug-in expects to leave the next morning);
- its own world's day-ahead prices (decision 0004 item 48), but only those
  already published when it plans (plan B4, decision 0004 item 53): prices
  for London delivery date D are published at 13:00 London on D-1
  (``PRICES["day_ahead_publication_local_hour"]``).  A half-hour not yet
  published is ranked on the expected price shape: the mean published price
  of the same London half-hour over the most recent published days
  (``visible_day_ahead_prices``).  An ordinary evening plug-in sees every
  price to the next morning; only a plan made before 13:00 (an EV still
  plugged in after its expected departure, an always-plugged EV) ranks part
  of its window on the shape.  The realised price adds a forecast error the
  plan never sees.

It plans just enough grid energy to reach the target, ``(target - stock) /
efficiency``, in the cheapest day-ahead half-hours between plug-in and the
expected departure, each capped at home charging power x 0.5 h.  Filling the
cheapest slot first is optimal here: the cost is linear in energy and every
slot has the same cap, so no plan can beat it (an LP would give the same
answer with more machinery).  If the window cannot hold the need, every slot
in it charges at full power, which is exactly "as soon as possible" within
the window.

Three edge rules, each chosen to keep the comparison fair:

- An EV still plugged in when its window ends (it left later than expected,
  or it is always plugged in) starts a new plan for the next expected
  departure, as a real smart charger would for the next night.
- Windows stop at the end of the study horizon.  Energy planned after it
  would count as "not recovered" and be priced at the public rate, a pure
  horizon artefact rather than a cost of smart charging.  The study runs
  noon to noon (decision 0004 item 52), so the expected departures plans
  target fall before the cut for clocked cohorts at the default margin; a
  window running past the end is clipped there and is still long, so the
  item 49 rule that made such sessions charge at once was retired.
- Plans start on the last warm-up night (the last 48 warm-up slots), not at
  the study start, so the first study night opens with batteries a smart
  charger left and is a real smart-charging night (item 52).  Earlier
  warm-up days are shared history on both paths.

How it fits: ``forecast`` builds a ``SmartCharging`` with
``smart_charging_inputs``; the physics kernel calls ``plan_cheapest_slots``
whenever a selected-path EV starts a plan, then executes the plan against the
EV's *actual* session (the evaluation worlds' sampled departures).  If the
EV leaves before the plan finishes, the
missed slots are simply not charged: the EV leaves below target, and any
later public top-up or energy not recovered by the end of the week is priced
below (items 32, 37, 4, 5, 13).  Smart charging draws no random number, so
both paths see the same random futures (common random numbers).

Intraday dispatch (intraday dispatch contract v1, decision 0004 item 62(b)).
With ``trading.intraday_dispatch`` on, a committed share of EVs is locked to
the day-ahead behaviour above (``locked_evs``) and the other, free EVs plan on
the latest intraday price known when they plan (``latest_known_prices``) and
re-plan at every whole hour when the new plan saves more than a threshold
plus the round-trip spread per MWh moved (``replan_when_worth``).
``IntradayDispatch`` carries what the dispatcher knows; the kernel applies
it.  Like the smart charger, it draws no random number.

Every price here is synthetic and every GBP figure illustrative: none is a
bid, settlement or Axle cash flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from axle_studio.model.assumptions import ACTION, COHORT_SHARES, PUBLIC_CHARGING
from axle_studio.model.clock import (
    STUDY_START_LOCAL_HOUR,
    london_wall_time_to_utc,
    session_night_dates,
    utc_half_hour_boundaries,
)
from axle_studio.model.sampling import connection_window_hours, day_ahead_publication_utc_ns
from axle_studio.model.settings import RunSettings

ACTION_ID = "smart_charging_v1"
ACTION_LABEL = "Smart home charging in the cheapest forecast half-hours before expected departure"

_HALF_HOUR_NS = 30 * 60 * 1_000_000_000
_NS_PER_HOUR = 3600 * 1_000_000_000
_LONDON = "Europe/London"
_SHAPE_DAYS = ACTION["expected_price_shape_days"].value

# Decision 0004 items 5 and 13: the reporting tolerance for the not-recovered
# flag and the valued shortfalls.  Early departures use it too, so a plan
# that missed only floating-point noise is not counted as a departure.
NOT_RECOVERED_TOLERANCE_KWH = ACTION["not_recovered_tolerance_kwh"].value

NOT_RECOVERED_MATERIAL_SHARE_PERCENT = ACTION["not_recovered_material_share_percent"].value
"""Default percent of weekly normal home import that makes a week materially not recovered."""

ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH = PUBLIC_CHARGING["public_charge_gbp_per_kwh"].value
"""Default illustrative public charging rate (decision 0004 item 4); not a tariff claim."""

COST_COLUMNS = [
    "world_id",
    "normal_home_import_kwh",
    "selected_home_import_kwh",
    "illustrative_selected_minus_normal_energy_cost_gbp",
    "selected_minus_normal_closing_battery_kwh",
    "selected_minus_normal_public_import_kwh",
    "selected_minus_normal_unserved_travel_kwh",
    "illustrative_selected_minus_normal_public_charge_cost_gbp",
    "illustrative_unrecovered_energy_value_gbp",
    "illustrative_unserved_travel_value_gbp",
    "illustrative_selected_minus_normal_total_gbp",
    "energy_not_recovered",
    "unrecovered_kwh",
    "unrecovered_share",
    "not_recovered_material",
    "evidence_kind",
]
"""``calculate_wholesale_world_cost_effect``'s columns.  ``summaries.build_summaries``
inserts ``sessions_affected_count`` before ``evidence_kind``: it needs per-EV
sessions, which only the chunked per-EV pass has."""


# ============================================================================
# What the smart charger knows, and how it plans one session
# ============================================================================


@dataclass(frozen=True)
class SmartCharging:
    """Decision-time inputs of the smart charger for one run.

    ``visible_price_gbp_per_mwh`` has shape (publication epoch, world, run
    slot): the day-ahead prices (GBP/MWh) a plan made in that epoch can rank,
    from ``visible_day_ahead_prices``.  An epoch is the stretch between two
    13:00 London publications, inside which what the charger can see does
    not change.  ``price_epoch_by_slot`` (run slot,) gives the epoch row for
    a plan made at the start of each run slot (warm-up and study slots, in
    kernel order), or -1 where the charger does not plan (warm-up before the
    last warm-up night).  ``expected_departure_utc_ns`` has shape (date,
    EV): for each London date from the day before the study to two days
    after it, the EV's expected departure as int64 UTC nanoseconds (typical
    departure clock time minus the safety margin).  Rows increase with the
    date, so the next expected departure after any instant in the horizon is
    always present.  None of these holds anything the charger could not
    know when it plans: unpublished day-ahead prices, realised prices and
    actual departures stay unknown to the plan (decision-time cut-off).

    Events and zones (trading contract v1 §2.4, §9.6), all in run-slot
    positions: ``zone_index`` (EV,) is each EV's zone (0 ... zone count - 1);
    ``event_notice_slot`` (request,) and ``event_adjustment_gbp_per_mwh``
    (request, zone, run slot) are ``events.planner_adjustments``: a plan
    made at or after a request's notice slot adds its adjustment for the
    EV's zone.  ``outage_probability`` (zone, run slot) is the control
    outage's non-response share by plug-in slot.

    Non-response (trading contract v1 §4.5, §9.6, §10.1e; decision 0004
    items 57 and 59): ``non_response_uniform`` (world, study night, EV)
    holds one uniform per session night (``None``: no session ignores its
    plan); ``non_response`` (world, study night, EV) is each session's
    chance ``rho`` of ignoring its plan (1 for a control EV or on its
    maker's outage night, else ``1 - r`` of its maker; ``None`` means 0);
    ``base_non_response`` (EV,) is ``1 - r`` alone and ``control_group``
    (EV,) marks the hold-out EVs, both only to label why a session ignores
    its plan (``plan_status``; ``None``: every ``rho`` is base non-response,
    no control group).  ``night_by_slot`` (run slot,) is the study night a
    plan made in each slot belongs to (warm-up slots count to night 0, so a
    session already plugged in at the study start uses night 0).  A plan
    is ignored when the session's uniform is below ``max(rho, outage
    share)``.

    Firm MW (§10.1e, §10.5b): ``blackout`` (run slot,) marks the half-hours
    in which plans may not move charging (``None``: none), and
    ``decision_slot_by_night`` (study night,) is the run slot of each
    night's intraday availability decision ``s_n`` (``None``: no decision
    snapshot, so ``decision_plan_kwh`` is 0).
    """

    visible_price_gbp_per_mwh: np.ndarray
    price_epoch_by_slot: np.ndarray
    expected_departure_utc_ns: np.ndarray
    zone_index: np.ndarray
    event_notice_slot: np.ndarray
    event_adjustment_gbp_per_mwh: np.ndarray
    outage_probability: np.ndarray
    night_by_slot: np.ndarray
    non_response_uniform: np.ndarray | None = None
    non_response: np.ndarray | None = None
    base_non_response: np.ndarray | None = None
    control_group: np.ndarray | None = None
    blackout: np.ndarray | None = None
    decision_slot_by_night: np.ndarray | None = None

    def for_slice(self, world_index: np.ndarray, ev_index: np.ndarray) -> SmartCharging:
        """The same inputs for a subset of worlds and EVs (chunked summaries, one-EV replay)."""

        def per_session(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is None else values[world_index][:, :, ev_index]

        def per_ev(values: np.ndarray | None) -> np.ndarray | None:
            return None if values is None else values[ev_index]

        return SmartCharging(
            self.visible_price_gbp_per_mwh[:, world_index],
            self.price_epoch_by_slot,
            self.expected_departure_utc_ns[:, ev_index].copy(),
            self.zone_index[ev_index],
            self.event_notice_slot,
            self.event_adjustment_gbp_per_mwh,
            self.outage_probability,
            self.night_by_slot,
            per_session(self.non_response_uniform),
            per_session(self.non_response),
            per_ev(self.base_non_response),
            per_ev(self.control_group),
            self.blackout,
            self.decision_slot_by_night,
        )


def smart_charging_inputs(
    settings: RunSettings,
    units: pd.DataFrame,
    day_ahead_price_gbp_per_mwh: np.ndarray,
    *,
    departure_margin_hours: float,
    zone_index: np.ndarray | None = None,
    event_notice_slot: np.ndarray | None = None,
    event_adjustment_gbp_per_mwh: np.ndarray | None = None,
    outage_probability: np.ndarray | None = None,
    non_response_uniform: np.ndarray | None = None,
    non_response: np.ndarray | None = None,
    base_non_response: np.ndarray | None = None,
    control_group: np.ndarray | None = None,
    blackout: np.ndarray | None = None,
    decision_slot_by_night: np.ndarray | None = None,
) -> SmartCharging:
    """Build the smart charger's decision-time inputs.

    ``units`` is the population frame; its weekday and weekend
    ``runtime_*departure_local_hour`` columns are each EV's cohort home
    departure (plug-out) clock hours, the same values the sampler centres home
    sessions on, so they are the charger's best guess of when the car leaves.
    ``day_ahead_price_gbp_per_mwh`` has shape (evaluation world, run slot):
    each world's day-ahead price (GBP/MWh) for every warm-up and study slot.
    ``departure_margin_hours`` (h, >= 0) is subtracted from the typical
    departure so the plan finishes early if the driver leaves early.

    Planning starts on the last warm-up night (the last 48 warm-up slots,
    decision 0004 item 52), so the first study night opens from a smart
    charger's batteries.  For each publication epoch a plan can start in,
    the visible prices are built once here rather than per slot in the
    kernel: they change only at 13:00 London.

    The event inputs are ``SmartCharging``'s (run-slot positions, from
    ``events.planner_adjustments`` and ``events.outage_probability``) and
    ``non_response_uniform`` is ``sampling.sample_non_response``'s draw.
    ``non_response`` (world, study night, EV) is each session's chance of
    ignoring its plan (``forecast``'s ``session_non_response_probability``,
    §10.1e), ``base_non_response`` (EV,) its maker's ``1 - r`` and
    ``control_group`` (EV,) the hold-out EVs.  ``blackout`` (run slot,) bool
    marks the blackout half-hours (§10.5b) and ``decision_slot_by_night``
    (study night,) the run slot of each night's intraday decision (§10.2d).
    Left out, every EV is in zone 0, no request, outage or blackout exists,
    every session follows its plan and no decision snapshot is taken.
    """

    margin = float(departure_margin_hours)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("departure_margin_hours must be a finite non-negative number")
    prices = np.asarray(day_ahead_price_gbp_per_mwh, dtype=np.float64)
    slot_count = 48 * (settings.warmup_days + settings.study_days)
    if (
        prices.shape != (settings.evaluation_world_count, slot_count)
        or not np.isfinite(prices).all()
    ):
        raise ValueError("day-ahead prices must be finite, one per world and run slot")
    if settings.warmup_days < 1:
        raise ValueError("smart charging plans from the last warm-up night: warmup_days >= 1")

    starts = pd.DatetimeIndex(
        utc_half_hour_boundaries(
            settings.start_local_date, settings.warmup_days, settings.study_days
        )[:-1]
    )
    slot_ns = starts.as_unit("ns").asi8
    publications = np.unique(day_ahead_publication_utc_ns(starts))
    epoch = np.searchsorted(publications, slot_ns, side="right")
    plan_start = 48 * (settings.warmup_days - 1)
    used = np.unique(epoch[plan_start:])
    visible = np.stack(
        [
            # A plan made in epoch e sees exactly what was published by the
            # epoch's publication instant, publications[e - 1].  Every slot's
            # own day is published the afternoon before it, so e >= 1.
            visible_day_ahead_prices(prices, starts, publications[e - 1])
            for e in used
        ]
    )
    price_epoch_by_slot = np.full(slot_count, -1, dtype=np.int64)
    price_epoch_by_slot[plan_start:] = np.searchsorted(used, epoch[plan_start:])

    vehicle_count = settings.vehicle_count
    zones = np.zeros(vehicle_count, dtype=np.int64) if zone_index is None else zone_index
    zones = np.asarray(zones, dtype=np.int64)
    zone_count = int(zones.max(initial=0)) + 1
    if outage_probability is not None:
        zone_count = max(zone_count, outage_probability.shape[0])
    if event_adjustment_gbp_per_mwh is not None:
        zone_count = max(zone_count, event_adjustment_gbp_per_mwh.shape[1])
    notice = np.zeros(0, dtype=np.int64) if event_notice_slot is None else event_notice_slot
    adjustment = (
        np.zeros((0, zone_count, slot_count))
        if event_adjustment_gbp_per_mwh is None
        else np.asarray(event_adjustment_gbp_per_mwh, dtype=np.float64)
    )
    outage = (
        np.zeros((zone_count, slot_count))
        if outage_probability is None
        else np.asarray(outage_probability, dtype=np.float64)
    )
    if zones.shape != (vehicle_count,) or zones.min(initial=0) < 0:
        raise ValueError("zone_index must give one zone position per EV")
    if adjustment.shape != (len(notice), zone_count, slot_count) or outage.shape != (
        zone_count,
        slot_count,
    ):
        raise ValueError("event adjustments and outage shares must be per zone and run slot")
    if non_response_uniform is not None and non_response_uniform.shape != (
        settings.evaluation_world_count,
        settings.study_days,
        vehicle_count,
    ):
        raise ValueError("non_response_uniform must be per world, study night and EV")
    session_shape = (settings.evaluation_world_count, settings.study_days, vehicle_count)
    rho = None if non_response is None else np.asarray(non_response, dtype=np.float64)
    if rho is not None and (rho.shape != session_shape or not ((rho >= 0) & (rho <= 1)).all()):
        raise ValueError("non_response must be one probability in 0-1 per world, night and EV")
    base = None if base_non_response is None else np.asarray(base_non_response, np.float64)
    if base is not None and (
        base.shape != (vehicle_count,) or not ((base >= 0) & (base <= 1)).all()
    ):
        raise ValueError("base_non_response must be one probability in 0-1 per EV")
    control = None if control_group is None else np.asarray(control_group, dtype=bool)
    if control is not None and control.shape != (vehicle_count,):
        raise ValueError("control_group must mark each EV")
    if non_response_uniform is None and ((outage > 0).any() or (rho is not None and rho.any())):
        raise ValueError("non-response needs the non-response uniforms")
    blackout_mask = None if blackout is None else np.asarray(blackout, dtype=bool)
    if blackout_mask is not None and blackout_mask.shape != (slot_count,):
        raise ValueError("blackout must mark each run slot")
    decisions = (
        None if decision_slot_by_night is None else np.asarray(decision_slot_by_night, np.int64)
    )
    if decisions is not None and (
        decisions.shape != (settings.study_days,)
        or (decisions < 48 * settings.warmup_days).any()
        or (decisions >= slot_count).any()
    ):
        raise ValueError("decision_slot_by_night must give one study run slot per night")
    # The study night a plan made in each run slot belongs to: London time
    # minus 12 h (session nights, decision 0004 item 52), clipped so warm-up
    # plans count to night 0 and a spring change's stray slots to the last.
    nights = session_night_dates(starts) - np.datetime64(settings.start_local_date, "D")
    night_by_slot = np.clip(nights.astype(np.int64), 0, settings.study_days - 1)
    return SmartCharging(
        visible,
        price_epoch_by_slot,
        expected_departures_utc_ns(settings, units, margin),
        zones,
        np.asarray(notice, dtype=np.int64),
        adjustment,
        outage,
        night_by_slot,
        non_response_uniform,
        rho,
        base,
        control,
        # An all-False mask plans exactly as no mask; None skips the extra work.
        blackout_mask if blackout_mask is not None and blackout_mask.any() else None,
        decisions,
    )


def visible_day_ahead_prices(
    day_ahead_gbp_per_mwh: np.ndarray,
    interval_start_utc: pd.DatetimeIndex,
    decision_utc: pd.Timestamp | int,
) -> np.ndarray:
    """The day-ahead prices a decision at ``decision_utc`` may rank (plan B4).

    ``day_ahead_gbp_per_mwh`` is (world, slot) in GBP/MWh over the slots
    starting at ``interval_start_utc`` (timezone-aware); ``decision_utc`` is
    the decision instant (a UTC ``Timestamp`` or int64 ns).  Returns a
    (world, slot) array:

    - a slot whose price is published at or before the decision
      (``day_ahead_publication_utc_ns``) keeps its day-ahead price;
    - any other slot gets the expected price shape: per world, the mean
      published price of the same London wall-clock half-hour over the
      ``ACTION["expected_price_shape_days"]`` (7) most recent published
      delivery days.  If none of those days has that half-hour (a history
      shorter than a day, or the skipped spring hour), the mean of every
      price in those days is used.

    This is the one visible-price rule: the smart charger and the trading
    overlay's expected plan both call it, so they see the same prices
    (trading contract v1 section 1.4).  It uses only prices published by
    the decision, so changing an unpublished price never changes its
    output.  Why a recent-days mean and not a model shape: it needs nothing
    beyond the prices the charger has already seen, works for any price
    generator, and a week's window covers the weekly cycle.
    """

    starts = pd.DatetimeIndex(interval_start_utc)
    decision_ns = pd.Timestamp(decision_utc).as_unit("ns").value
    published = day_ahead_publication_utc_ns(starts) <= decision_ns
    london = starts.tz_convert(_LONDON)
    delivery_date = london.tz_localize(None).normalize().to_numpy()
    half_hour = np.asarray(london.hour * 2 + london.minute // 30)
    prices = np.asarray(day_ahead_gbp_per_mwh, dtype=np.float64)
    if published.all():
        return prices.copy()

    recent_days = np.unique(delivery_date[published])[-_SHAPE_DAYS:]
    window = published & np.isin(delivery_date, recent_days)
    if not window.any():
        raise ValueError("no day-ahead price is published at the decision instant")
    fallback = prices[:, window].mean(axis=1)
    expected = np.empty((prices.shape[0], 48))
    for index in range(48):
        in_cell = window & (half_hour == index)
        expected[:, index] = prices[:, in_cell].mean(axis=1) if in_cell.any() else fallback
    return np.where(published, prices, expected[:, half_hour])


def expected_departures_utc_ns(
    settings: RunSettings, units: pd.DataFrame, departure_margin_hours: float
) -> np.ndarray:
    """Expected home departures as int64 UTC ns, shape (date, EV).

    One row per London date from the day before the study to two days after
    it, increasing with the date.  Each is the EV's typical departure clock
    hour for that date's day type (``runtime_departure_local_hour`` on
    weekdays, ``runtime_weekend_departure_local_hour`` on Saturday and
    Sunday, decision 0004 item 42) minus ``departure_margin_hours`` (h).  The
    smart charger plans to it, and the flexibility metrics count time left
    until it, so both use one definition.
    """

    # The typical departure is a London clock time, so each date is converted
    # at its own offset: a clock change inside the week moves the UTC instant
    # by an hour, never the wall-clock time (decision 0004 item 1).
    first_date = settings.start_local_date - timedelta(days=1)
    dates = np.datetime64(first_date, "D") + np.arange(settings.study_days + 3)
    # numpy's weekday: 1970-01-01 (day 0) was a Thursday, so Monday = 0.
    weekend = (dates.astype(np.int64) + 3) % 7 >= 5
    weekday_hours, weekend_hours = connection_window_hours(units, "departure")
    hours = np.where(
        weekend[:, np.newaxis], weekend_hours[np.newaxis, :], weekday_hours[np.newaxis, :]
    )
    departures = london_wall_time_to_utc(dates[:, np.newaxis], 60.0 * hours)
    return departures.astype(np.int64) - round(float(departure_margin_hours) * _NS_PER_HOUR)


def timed_start_allowed(settings: RunSettings, start_local_hour: float) -> np.ndarray:
    """Which run half-hours the timed-start rule lets home charging happen in (decision 0007).

    Returns a (run slot,) bool, warm-up then study slots in the kernel's own
    order: ``True`` where the "timed" path may charge, ``False`` in the
    barred stretch. The barred window runs every day from London 12:00
    (``clock.STUDY_START_LOCAL_HOUR``, the same noon the session-night
    convention keys on, ``clock.session_night_dates``) up to
    ``start_local_hour`` (London clock hours, 0-23.5 in half-hour steps): at
    0 (midnight) that is 12:00-24:00 barred, 00:00-12:00 allowed, so an
    18:00 plug-in waits for midnight and a 01:00 plug-in may charge at once.
    The window is read off each slot's own London wall clock
    (``utc_half_hour_boundaries``, converted as ``visible_day_ahead_prices``
    already does), so a clock change inside the run moves no boundary.
    """

    hour = float(start_local_hour)
    if not math.isfinite(hour) or not 0.0 <= hour <= 23.5:
        raise ValueError("timed_start_local_hour must be between 0 and 23.5")
    if not (hour * 2.0).is_integer():
        raise ValueError("timed_start_local_hour must be a half-hour multiple")

    starts = pd.DatetimeIndex(
        utc_half_hour_boundaries(
            settings.start_local_date, settings.warmup_days, settings.study_days
        )[:-1]
    )
    london = starts.tz_convert(_LONDON)
    half_hour = np.asarray(london.hour * 2 + london.minute // 30)
    # Pivot half-hour positions so 0 sits at London noon (the session-night
    # boundary): the barred stretch is the first ``barred_length`` positions
    # from there, wrapping past midnight whenever the start hour is before
    # noon (the usual case), and empty when the start hour is noon itself.
    noon = STUDY_START_LOCAL_HOUR * 2
    start_index = round(hour * 2.0)
    barred_length = (start_index - noon) % 48
    offset = (half_hour - noon) % 48
    return offset >= barred_length


def plan_cheapest_slots(
    need_kwh: np.ndarray, slot_cap_kwh: np.ndarray, window_price: np.ndarray
) -> np.ndarray:
    """Plan grid energy per slot for a batch of sessions, cheapest slot first.

    ``need_kwh`` (session,) is the grid energy to reach the target,
    ``slot_cap_kwh`` (session,) the most one half-hour can import (home
    charging power x 0.5 h) and ``window_price`` (session, slot) the forecast
    price of each slot from plug-in onwards, ``+inf`` outside the session's
    window.  Returns grid kWh per (session, slot).

    Sort each session's slots by price and give the k-th cheapest slot
    ``min(cap, max(0, need - k * cap))``: full slots until the need is met,
    one partial slot, then nothing.  A stable sort keeps equal prices in time
    order, so ties charge earlier (sooner charging is the safer tie-break
    when the driver may leave early).  Slots outside the window get nothing,
    so a need the window cannot hold fills every window slot at full power.
    """

    order = np.argsort(window_price, axis=1, kind="stable")
    already_planned = np.arange(window_price.shape[1]) * slot_cap_kwh[:, np.newaxis]
    planned_sorted = np.clip(
        need_kwh[:, np.newaxis] - already_planned, 0.0, slot_cap_kwh[:, np.newaxis]
    )
    in_window = np.isfinite(np.take_along_axis(window_price, order, axis=1))
    planned = np.empty_like(planned_sorted)
    np.put_along_axis(planned, order, np.where(in_window, planned_sorted, 0.0), axis=1)
    return planned


def plan_around_blackout(
    need_kwh: np.ndarray,
    slot_cap_kwh: np.ndarray,
    window_price: np.ndarray,
    blackout: np.ndarray,
) -> np.ndarray:
    """Plan like ``plan_cheapest_slots`` while leaving blackout half-hours untouched (§10.5b).

    Inputs as ``plan_cheapest_slots``, plus ``blackout`` (session, slot)
    bool: the half-hours in which the aggregator has promised not to move
    charging in either direction.  Returns grid kWh per (session, slot).

    A blackout slot keeps exactly the charging the EV would have done
    unmanaged (full power from plug-in, the normal rule inside the window)
    and takes nothing more; the rest of the need goes to the cheapest slots
    outside the blackout.  So nothing is moved out of a blackout slot and
    nothing into it.  Because the plan never asks for more than the
    unmanaged amount there, the kernel's ``min(plan, normal limit)``
    executes it unchanged.  A need the slots outside cannot hold leaves the
    session short rather than breaching the blackout (lead decision O6).
    Two calls of the existing planner, rather than a constrained solver:
    each call is already optimal for its linear, equally capped problem.
    """

    in_window = np.isfinite(window_price)
    unmanaged = plan_cheapest_slots(need_kwh, slot_cap_kwh, np.where(in_window, 0.0, np.inf))
    fixed = np.where(blackout, unmanaged, 0.0)
    rest = plan_cheapest_slots(
        need_kwh - fixed.sum(axis=1), slot_cap_kwh, np.where(blackout, np.inf, window_price)
    )
    return fixed + rest


# ============================================================================
# Intraday dispatch: locked and free EVs, the latest price, the re-plan rule
# ============================================================================


@dataclass(frozen=True)
class IntradayDispatch:
    """What the intraday dispatcher knows for one run (intraday dispatch contract v1 §5.1).

    It sits beside ``SmartCharging`` (what the day-ahead planner knows).
    ``locked`` (EV,) bool is ``units.dispatch_locked``: a locked EV plans on
    B4-visible day-ahead prices exactly as without dispatch; a free EV plans
    and re-plans on ``latest_known_prices``.  ``intraday_path_gbp_per_mwh``
    (world, run slot, H) is the price generator's intraday path, h whole
    hours before gate closure (h = 0 the close); ``publication_utc_ns`` and
    ``gate_utc_ns`` (run slot,) int64 are each slot's day-ahead publication
    instant and gate closure.  ``decision_slot`` (run slot,) bool marks the
    re-plan instants: slots starting on a whole London hour from the last
    warm-up night on.  ``replan_threshold_gbp_per_mwh`` (theta) and
    ``half_spread_gbp_per_mwh`` (s) set the re-plan bar, ``theta + 2 s``
    GBP per MWh moved.  The kernel reads the path only through
    ``latest_known_prices``, so nothing unknown at a decision is used.
    """

    locked: np.ndarray
    intraday_path_gbp_per_mwh: np.ndarray
    publication_utc_ns: np.ndarray
    gate_utc_ns: np.ndarray
    decision_slot: np.ndarray
    replan_threshold_gbp_per_mwh: float
    half_spread_gbp_per_mwh: float

    def for_slice(self, world_index: np.ndarray, ev_index: np.ndarray) -> IntradayDispatch:
        """The same inputs for a subset of worlds and EVs (chunked summaries, one-EV replay)."""

        return IntradayDispatch(
            self.locked[ev_index],
            self.intraday_path_gbp_per_mwh[world_index],
            self.publication_utc_ns,
            self.gate_utc_ns,
            self.decision_slot,
            self.replan_threshold_gbp_per_mwh,
            self.half_spread_gbp_per_mwh,
        )


def locked_evs(units: pd.DataFrame, commitment_share: float) -> np.ndarray:
    """Which EVs are locked to their day-ahead plans, (EV,) bool (contract §2).

    ``commitment_share`` is ``c`` (``trading.day_ahead_commitment_share``,
    fraction 0-1): the share of the forecast deviation sold day-ahead and
    also the share of EVs locked (Changed by §11.2: one commitment, two
    views of it).  ``K = round-half-up(c x N)`` EVs are locked, split across
    cohorts by largest remainder on ``K x n_c / N`` (ties to the earlier
    cohort in the defaults' order), and the locked EVs of a cohort are its
    first ``k_c`` rows of ``units``.  No draw: the population's cohort
    permutation already made row order within a cohort arbitrary, so the
    locked set is exchangeable with any other subset of the same size, and
    editing ``c`` moves no random channel.  By EV rather than by a fraction
    of each session's energy (rejected in §2): two plans per EV would share
    one charger cap per slot, which the cheapest-slots rule cannot express.
    """

    share = float(commitment_share)
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("commitment_share must be a fraction in 0-1")
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    vehicle_count = len(cohorts)
    locked = np.zeros(vehicle_count, dtype=bool)
    if vehicle_count == 0:
        return locked
    # Half up, as the control group counts (trading contract §9.5).
    size = math.floor(share * vehicle_count + 0.5)
    order = list(COHORT_SHARES)
    present = [c for c in order if (cohorts == c).any()]
    present += [c for c in pd.unique(cohorts) if c not in order]
    counts = np.array([(cohorts == c).sum() for c in present])
    # Largest remainder in whole numbers (K x n_c = quota x N + remainder),
    # so equal remainders tie exactly and the stable sort gives a tie to the
    # earlier cohort.
    per_cohort, remainder = np.divmod(size * counts, vehicle_count)
    for index in np.argsort(-remainder, kind="stable")[: size - per_cohort.sum()]:
        per_cohort[index] += 1
    for cohort_id, k in zip(present, per_cohort, strict=True):
        locked[np.flatnonzero(cohorts == cohort_id)[:k]] = True
    return locked


def intraday_dispatch_inputs(
    settings: RunSettings,
    units: pd.DataFrame,
    intraday_path_gbp_per_mwh: np.ndarray,
    *,
    replan_threshold_gbp_per_mwh: float,
    half_spread_gbp_per_mwh: float,
    gate_closure_minutes: float,
) -> IntradayDispatch:
    """Build the intraday dispatcher's inputs (contract §5.1).

    ``units`` must carry ``dispatch_locked`` (bool, from ``locked_evs``).
    ``intraday_path_gbp_per_mwh`` (evaluation world, run slot, H) is
    ``MarketPrices.intraday_path_gbp_per_mwh`` over the warm-up and study
    slots.  ``replan_threshold_gbp_per_mwh`` (theta, GBP/MWh moved, >= 0)
    and ``half_spread_gbp_per_mwh`` (s, GBP/MWh, >= 0) are the records;
    ``gate_closure_minutes`` (min before the slot start) sets each slot's
    gate closure.  The B4-visible day-ahead prices it also needs are on
    ``SmartCharging`` already.
    """

    slot_count = 48 * (settings.warmup_days + settings.study_days)
    if "dispatch_locked" not in units:
        raise ValueError("units must carry dispatch_locked (action.locked_evs)")
    locked = units["dispatch_locked"].to_numpy()
    if locked.dtype != bool or locked.shape != (settings.vehicle_count,):
        raise ValueError("dispatch_locked must mark each EV with a bool")
    path = np.asarray(intraday_path_gbp_per_mwh, dtype=np.float64)
    if (
        path.ndim != 3
        or path.shape[:2] != (settings.evaluation_world_count, slot_count)
        or path.shape[2] < 1
        or not np.isfinite(path).all()
    ):
        raise ValueError("the intraday path must be finite, per world, run slot and step")
    threshold, half_spread = float(replan_threshold_gbp_per_mwh), float(half_spread_gbp_per_mwh)
    for name, value in (("replan threshold", threshold), ("half spread", half_spread)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"the {name} must be a finite non-negative number")
    starts = pd.DatetimeIndex(
        utc_half_hour_boundaries(
            settings.start_local_date, settings.warmup_days, settings.study_days
        )[:-1]
    )
    slot_ns = starts.as_unit("ns").asi8
    # Re-plans happen at the trader's hourly instants (contract §4.1, lead
    # decision Q6), so the book the trader reads at a whole hour holds the
    # plans the fleet will follow; and only where plans exist at all (from
    # the last warm-up night, decision 0004 item 52).
    whole_hour = np.asarray(starts.tz_convert(_LONDON).minute) == 0
    decision_slot = whole_hour & (np.arange(slot_count) >= 48 * (settings.warmup_days - 1))
    return IntradayDispatch(
        locked.copy(),
        path,
        day_ahead_publication_utc_ns(starts),
        slot_ns - round(float(gate_closure_minutes) * 60 * 1_000_000_000),
        decision_slot,
        threshold,
        half_spread,
    )


def latest_known_prices(
    intraday_path_gbp_per_mwh: np.ndarray,
    visible_day_ahead_gbp_per_mwh: np.ndarray,
    publication_utc_ns: np.ndarray,
    gate_utc_ns: np.ndarray,
    decision_utc_ns: int,
) -> np.ndarray:
    """The latest price of each slot known at a decision, (world, slot) GBP/MWh (contract §3).

    ``intraday_path_gbp_per_mwh`` (world, slot, H) holds the intraday price
    h whole hours before each slot's gate closure (h = 0 the close);
    ``visible_day_ahead_gbp_per_mwh`` (world, slot) the B4-visible day-ahead
    prices at the decision (``visible_day_ahead_prices``);
    ``publication_utc_ns`` and ``gate_utc_ns`` (slot,) each slot's day-ahead
    publication and gate closure; ``decision_utc_ns`` the decision instant.
    All instants are int64 UTC ns.

    A slot whose day-ahead price is not yet published has no intraday
    market either, so it keeps the B4 expected shape.  Otherwise the latest
    update made by the decision is ``h = ceil((gate - tau) / 1 h)``, clipped
    to 0 and H - 1: ceil picks an update that has happened, floor would read
    one from the future (trading review B6).  Past gate closure it is the
    close, fixed at the gate and so known.  Before the first update the path
    is the expected close (day-ahead plus premium), so at publication a free
    EV ranks what a locked EV ranks up to the constant premium.  The one
    rule for the dispatcher, the trader's expected plan and the replay.
    """

    path = intraday_path_gbp_per_mwh
    decision = int(decision_utc_ns)
    # Integer ceil of (gate - tau) / 1 h, as -floor((tau - gate) / 1 h).
    step = np.clip(-((decision - np.asarray(gate_utc_ns)) // _NS_PER_HOUR), 0, path.shape[2] - 1)
    latest = path[:, np.arange(path.shape[1]), step]
    return np.where(
        np.asarray(publication_utc_ns) > decision, visible_day_ahead_gbp_per_mwh, latest
    )


def replan_when_worth(
    plan_kwh: np.ndarray,
    ranking_price: np.ndarray,
    slot_cap_kwh: np.ndarray,
    *,
    replan_threshold_gbp_per_mwh: float,
    half_spread_gbp_per_mwh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-plan a batch of free sessions when it is worth it (contract §4.1 step 0).

    ``plan_kwh`` (session, slot) is each incumbent plan's grid kWh per slot;
    ``ranking_price`` (session, slot) the price each slot ranks at now, GBP/MWh
    (the latest known price plus announced request adjustments), ``+inf``
    where the plan may not move energy (slots already past, outside the
    window, blackout half-hours); ``slot_cap_kwh`` (session,) the most one
    half-hour imports.  Returns the plans to use (session, slot) and whether
    each session was re-planned (session,) bool.

    The incumbent's energy on ``+inf`` slots stays where it is (``fixed``:
    past slots keep their record, blackout slots their unmanaged amount,
    §10.5b); the rest, ``r - sum(fixed)``, is planned again cheapest first,
    so the remaining total is unchanged and the target is never given up for
    a price.  The new plan replaces the incumbent only if

        saving - (theta + 2 s) x moved / 1000 > 1e-9 GBP,

    ``saving = sum((g - n) x price) / 1000`` over finite-price slots (``0 x
    inf`` is skipped, review B5) and ``moved = sum(max(0, g - n))`` kWh: the
    spread is paid twice per MWh moved (sell the turn-down in one slot, buy
    it back in another) and ``theta`` is the policy margin that stops the
    fleet chasing noise.  The 1e-9 GBP floor keeps float noise from counting
    as a saving.  When the window cannot hold the need both plans fill every
    slot, the saving is 0 and the incumbent stands.
    """

    rankable = np.isfinite(ranking_price)
    fixed = np.where(rankable, 0.0, plan_kwh)
    new = fixed + plan_cheapest_slots(
        plan_kwh.sum(axis=1) - fixed.sum(axis=1), slot_cap_kwh, ranking_price
    )
    difference = plan_kwh - new
    saving_gbp = (
        np.where(rankable, difference * np.where(rankable, ranking_price, 0.0), 0.0).sum(axis=1)
        / 1000.0
    )
    moved_kwh = np.maximum(difference, 0.0).sum(axis=1)
    bar_gbp_per_mwh = float(replan_threshold_gbp_per_mwh) + 2.0 * float(half_spread_gbp_per_mwh)
    replan = saving_gbp - bar_gbp_per_mwh * moved_kwh / 1000.0 > 1e-9
    return np.where(replan[:, np.newaxis], new, plan_kwh), replan


# ============================================================================
# The illustrative cost effect in the evaluation worlds
# ============================================================================


def slot_cost_effect_gbp(
    normal_home_kwh: np.ndarray,
    selected_home_kwh: np.ndarray,
    normal_public_kwh: np.ndarray,
    selected_public_kwh: np.ndarray,
    day_ahead_gbp_per_mwh: np.ndarray,
    public_charge_gbp_per_kwh: float,
) -> np.ndarray:
    """The customer leg's selected-minus-normal cost per slot, GBP (illustrative).

    Inputs are paired arrays of one shape, usually (world, slot): fleet (or
    one EV's) home and public grid import on each path in kWh per
    half-hour, and that world's day-ahead price in GBP/MWh.  Returns
    ``(selected - normal) home x P_DA / 1000 + (selected - normal) public x
    public_charge_gbp_per_kwh``: negative when the smart path cost the
    drivers less.  Home import is valued at the day-ahead price, never the
    realised one (plan B3, decision 0004 item 53).

    One formula for the weekly cost effect (``calculate_wholesale_world_cost_effect``
    sums it) and the replay's running saving (replay contract v1 §1.4), so
    the two cannot drift.  The end-of-week shortfall values (items 4 and 13)
    are not flows and are not here.
    """

    home = (selected_home_kwh - normal_home_kwh) * day_ahead_gbp_per_mwh / 1000.0
    return home + (selected_public_kwh - normal_public_kwh) * float(public_charge_gbp_per_kwh)


def calculate_wholesale_world_cost_effect(
    fleet_world_intervals: pd.DataFrame,
    day_ahead_prices: pd.DataFrame,
    *,
    public_charge_gbp_per_kwh: float,
    material_share_percent: float = NOT_RECOVERED_MATERIAL_SHARE_PERCENT,
    selected_path_id: str = "selected",
) -> pd.DataFrame:
    """Value paired ``selected_path_id``-minus-normal outcomes per world, illustratively.

    ``fleet_world_intervals`` holds both paths of every evaluation world with
    the runner's physical column names; ``day_ahead_prices`` each world's
    synthetic day-ahead price (``wholesale_forecast_gbp_per_mwh``, GBP/MWh)
    per slot.  Returns one row per world (``COST_COLUMNS``), still named for
    "selected" even when ``selected_path_id`` is a different path: same
    schema, same prices, same public rate, same shortfall valuation, so two
    calls are directly comparable.

    ``selected_path_id`` (default ``"selected"``) is the path paired against
    normal.  Model step 2 (decision 0007) calls this a second time with
    ``selected_path_id="timed"`` to cost the optional timed path the same
    way, giving ``timed_cost_effect`` beside the ordinary ``cost_effect``;
    the function itself does not change, only which path it is handed.

    Every GBP column is illustrative. None is a bid, settlement, payer/payee
    flow, supplier revenue or Axle cash (decisions 0003 and 0004).

    - ``illustrative_selected_minus_normal_energy_cost_gbp``: home grid import
      difference valued at each world's synthetic day-ahead price (plan B3,
      decision 0004 item 53).  Home import is the customer's tariff leg,
      bought ahead at day-ahead prices; the intraday and imbalance prices
      value only the trading position, which is a separate ledger.  Before
      item 53 this used the realised price, so a forecast error the charger
      could not see moved the saving.
    - ``illustrative_selected_minus_normal_public_charge_cost_gbp``: public
      grid import difference times ``public_charge_gbp_per_kwh`` (signed, so
      less public charging on the selected path is negative).  An EV that
      left before its smart plan finished may need an extra public top-up
      later in the week; this is where it is priced.
    - ``illustrative_unrecovered_energy_value_gbp``: closing battery shortfall
      ``max(0, -selected_minus_normal_closing_battery_kwh)`` times the same
      rate, on the basis that the driver would buy it back at a public
      charger. A closing surplus is not credited: the decision values energy
      not recovered only, so this column is never negative.
    - ``illustrative_unserved_travel_value_gbp``: added unserved travel
      ``max(0, selected_minus_normal_unserved_travel_kwh)`` times the same
      rate, since the driver would need a public charge to make that trip.
      Unpriced, unserved travel would leave energy in the battery and show a
      false saving (decision 0004 item 13). Never negative.
    - ``illustrative_selected_minus_normal_total_gbp``: the illustrative sum
      of the four labelled components above and nothing else.

    The unrecovered and unserved values use the same 1e-6 kWh tolerance as
    the flag: a difference within +-1e-6 kWh is valued at zero.  Both are
    battery-side kWh valued at the grid-side public rate; this simplification
    ignores public charging losses.

    Physical differences are reported alongside:
    ``selected_minus_normal_closing_battery_kwh`` (final slot stock),
    ``selected_minus_normal_public_import_kwh``,
    ``selected_minus_normal_unserved_travel_kwh`` (outbound plus return) and
    ``energy_not_recovered``, which is True when the selected path ends with
    lower battery stock, more public import or more unserved travel than the
    normal path beyond 1e-6 kWh (decision 0004 item 5).

    Magnitude (decision 0004 item 45).  At fleet scale almost every week has
    some EV a half-hour short, so the yes/no flag says little on its own:

    - ``unrecovered_kwh``: the closing battery shortfall plus the extra
      public import plus the added unserved travel, each counted only when
      positive and beyond the tolerance.  It is the same energy the flag
      looks at, summed: what the smart path did not recover at home.  Like
      the valuation it mixes battery-side and grid-side kWh (public losses
      ignored); a closing surplus does not offset it.
    - ``unrecovered_share``: that as a fraction of the world's weekly
      normal-path home import (NaN when the normal path imports nothing).
    - ``not_recovered_material``: ``unrecovered_kwh`` is positive and the
      share is at least ``material_share_percent`` (percent, illustrative
      default 1 %), or the share is undefined.  ``not_recovered_world_count``
      counts these weeks.
    """

    rate = float(public_charge_gbp_per_kwh)
    keys = ["world_id", "interval_start_utc", "interval_end_utc"]
    quantities = [
        "home_grid_import_kwh",
        "public_grid_import_kwh",
        "unserved_travel_kwh",
        "closing_battery_kwh",
    ]
    fleet = fleet_world_intervals.assign(
        unserved_travel_kwh=fleet_world_intervals["unserved_outbound_travel_battery_kwh"]
        + fleet_world_intervals["unserved_return_travel_battery_kwh"]
    )
    normal = fleet.loc[fleet["path_id"].eq("normal"), [*keys, *quantities]]
    selected = fleet.loc[fleet["path_id"].eq(selected_path_id), [*keys, *quantities]]
    # Pair the paths world by world and slot by slot: every difference is
    # taken inside a world before anything is summarised across worlds.
    paired = normal.merge(
        selected, on=keys, how="inner", validate="one_to_one", suffixes=("_normal", "_selected")
    )
    if len(paired) != len(normal) or len(paired) != len(selected):
        raise ValueError("normal and selected paths must have identical world/interval keys")
    prices = day_ahead_prices.loc[:, [*keys, "wholesale_forecast_gbp_per_mwh"]]
    valued = paired.merge(prices, on=keys, how="inner", validate="one_to_one")
    if len(valued) != len(paired):
        raise ValueError("day-ahead prices must match every paired world/interval key")
    # The two flow components, each through the one per-slot formula (the
    # other leg's import set to zero), so the weekly figure is the week's
    # sum of the replay's running figure.
    no_import = np.zeros(len(valued))
    valued["cost_effect_gbp"] = slot_cost_effect_gbp(
        valued["home_grid_import_kwh_normal"].to_numpy(),
        valued["home_grid_import_kwh_selected"].to_numpy(),
        no_import,
        no_import,
        valued["wholesale_forecast_gbp_per_mwh"].to_numpy(),
        rate,
    )
    valued["public_cost_gbp"] = slot_cost_effect_gbp(
        no_import,
        no_import,
        valued["public_grid_import_kwh_normal"].to_numpy(),
        valued["public_grid_import_kwh_selected"].to_numpy(),
        valued["wholesale_forecast_gbp_per_mwh"].to_numpy(),
        rate,
    )
    valued["public_difference_kwh"] = (
        valued["public_grid_import_kwh_selected"] - valued["public_grid_import_kwh_normal"]
    )
    valued["unserved_difference_kwh"] = (
        valued["unserved_travel_kwh_selected"] - valued["unserved_travel_kwh_normal"]
    )
    valued["closing_difference_kwh"] = (
        valued["closing_battery_kwh_selected"] - valued["closing_battery_kwh_normal"]
    )
    valued = valued.sort_values(["world_id", "interval_start_utc"], kind="stable")
    result = valued.groupby("world_id", sort=True, as_index=False).agg(
        normal_home_import_kwh=("home_grid_import_kwh_normal", "sum"),
        selected_home_import_kwh=("home_grid_import_kwh_selected", "sum"),
        illustrative_selected_minus_normal_energy_cost_gbp=("cost_effect_gbp", "sum"),
        selected_minus_normal_closing_battery_kwh=("closing_difference_kwh", "last"),
        selected_minus_normal_public_import_kwh=("public_difference_kwh", "sum"),
        selected_minus_normal_unserved_travel_kwh=("unserved_difference_kwh", "sum"),
        illustrative_selected_minus_normal_public_charge_cost_gbp=("public_cost_gbp", "sum"),
    )
    closing = result["selected_minus_normal_closing_battery_kwh"]
    unserved = result["selected_minus_normal_unserved_travel_kwh"]
    shortfall = -closing
    # Decision 0004 item 13: values within the flag's 1e-6 kWh tolerance are
    # zero, so floating-point noise is neither flagged nor priced and the
    # priced columns always agree with ``energy_not_recovered``.
    tolerance = NOT_RECOVERED_TOLERANCE_KWH
    shortfall_kwh = shortfall.where(shortfall > tolerance, 0.0)
    added_unserved_kwh = unserved.where(unserved > tolerance, 0.0)
    result["illustrative_unrecovered_energy_value_gbp"] = shortfall_kwh * rate
    result["illustrative_unserved_travel_value_gbp"] = added_unserved_kwh * rate
    result["illustrative_selected_minus_normal_total_gbp"] = (
        result["illustrative_selected_minus_normal_energy_cost_gbp"]
        + result["illustrative_selected_minus_normal_public_charge_cost_gbp"]
        + result["illustrative_unrecovered_energy_value_gbp"]
        + result["illustrative_unserved_travel_value_gbp"]
    )
    public_extra = result["selected_minus_normal_public_import_kwh"]
    result["energy_not_recovered"] = (
        (shortfall > tolerance) | (public_extra > tolerance) | (unserved > tolerance)
    )
    result["unrecovered_kwh"] = (
        shortfall_kwh + public_extra.where(public_extra > tolerance, 0.0) + added_unserved_kwh
    )
    normal_import = result["normal_home_import_kwh"]
    result["unrecovered_share"] = (result["unrecovered_kwh"] / normal_import).where(
        normal_import > 0.0
    )
    # Written as "not below the threshold" so an undefined share (no normal
    # home import but energy still short) counts as material, not hidden.
    below_threshold = 100.0 * result["unrecovered_share"] < material_share_percent
    result["not_recovered_material"] = (result["unrecovered_kwh"] > 0.0) & ~below_threshold
    result["evidence_kind"] = "illustrative_synthetic"
    return result.loc[:, COST_COLUMNS]


__all__ = [
    "ACTION_ID",
    "ACTION_LABEL",
    "COST_COLUMNS",
    "ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH",
    "NOT_RECOVERED_MATERIAL_SHARE_PERCENT",
    "NOT_RECOVERED_TOLERANCE_KWH",
    "IntradayDispatch",
    "SmartCharging",
    "calculate_wholesale_world_cost_effect",
    "day_ahead_publication_utc_ns",
    "expected_departures_utc_ns",
    "intraday_dispatch_inputs",
    "latest_known_prices",
    "locked_evs",
    "plan_around_blackout",
    "plan_cheapest_slots",
    "replan_when_worth",
    "slot_cost_effect_gbp",
    "smart_charging_inputs",
    "timed_start_allowed",
    "visible_day_ahead_prices",
]
