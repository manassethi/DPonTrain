"""
Synthetic booking data for train 12138 (Punjab Mail, Firozpur -> Mumbai CSMT), 3A class.

The original project used confidential Indian Railways booking records, which cannot be
published. This module generates a stand-in dataset with the same columns and the same
*kinds* of patterns, so every notebook in this repository runs end to end:

  * daily departures, 1 Aug 2023 - 31 Jul 2025
  * weekly cycle (busy Fridays-Sundays), summer peak, festival rushes and a small trend
  * a yearly pattern that repeats, so "same weekday last year" carries information
  * late booking: a departure-day share, a Tatkal spike the day before, and a long tail
  * the advance booking window shrinking from 120 to 60 days in late 2024
  * origin-destination journeys along the real route, from a simple gravity model
  * group bookings: 1-6 passengers per ticket

The parameters are stylised. They are not estimated from, and do not reproduce, the
confidential data, so numbers from these notebooks will differ from the original study.

    bookings = make_bookings()          # one row per ticket
    daily = daily_totals(bookings)      # one row per departure: total passengers
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROUTE_FILE = Path(__file__).resolve().parents[1] / "data" / "route_12138.csv"
START, END = "2023-08-01", "2025-07-31"
WINDOW_CHANGE = "2024-11-01"           # booking window shortened from 120 to 60 days

# --- Demand level per departure ---------------------------------------------------
BASE_PASSENGERS = 900                  # average 3A passengers per departure
WEEKDAY = np.array([0.96, 0.94, 0.95, 0.93, 1.03, 1.13, 1.06])   # Mon .. Sun
SUMMER_PEAK_DAY, SUMMER_AMPLITUDE = 145, 0.07                    # day of year (late May), +-7%
YEARLY_PATTERN_AMPLITUDE = 0.04        # repeating within-year wiggles (school holidays, weddings, ...)
TREND_PER_YEAR = 0.02
NOISE = 0.06                           # day-to-day randomness (log scale)
FESTIVALS = {                          # main day; travel peaks before and dips on the day itself
    "Diwali": ["2023-11-12", "2024-10-31", "2025-10-20"],
    "Holi": ["2024-03-25", "2025-03-14"],
    "Raksha Bandhan": ["2023-08-30", "2024-08-19"],
}

# --- Booking curve ----------------------------------------------------------------
DEPARTURE_DAY_SHARE = 0.16
TATKAL_SHARE = 0.24                    # booked the day before departure
EARLY_DECAY_DAYS = 10                  # how fast advance bookings thin out
LONG_TAIL_WEIGHT = 0.08                # flat tail of early planners

# --- Origin-destination mix -------------------------------------------------------
NODE_SIZE = {"FZR": 3.0, "BTI": 2.5, "NDLS": 9.0, "AGC": 3.5, "VGLJ": 2.5, "BPL": 5.0,
             "BSL": 3.5, "NK": 3.5, "CSMT": 8.0}
HUB_SHARE = 0.7                        # share of a node's passengers using the node station itself

TICKET_SIZES = np.arange(1, 7)
TICKET_PROBS = np.array([0.42, 0.25, 0.13, 0.11, 0.06, 0.03])


def route():
    """Stops of train 12138 in order, with distance from Firozpur (km). Public timetable data."""
    return pd.read_csv(ROUTE_FILE)


def demand_levels(dates, seed=0):
    """Expected passengers per departure relative to BASE_PASSENGERS (1.0 = average day)."""
    rng = np.random.default_rng([seed, 1])
    dates = pd.DatetimeIndex(dates)
    doy = dates.day_of_year.to_numpy()
    level = WEEKDAY[dates.day_of_week] * (1 + SUMMER_AMPLITUDE * np.cos(2 * np.pi * (doy - SUMMER_PEAK_DAY) / 365.25))
    phases = rng.uniform(0, 2 * np.pi, 3)
    wiggle = sum(np.cos(2 * np.pi * k * doy / 365.25 + p) / k for k, p in zip(range(2, 5), phases))
    level = level * (1 + YEARLY_PATTERN_AMPLITUDE * wiggle / 1.8)
    for days in FESTIVALS.values():
        for day in pd.to_datetime(days):
            gap = (dates - day).days.to_numpy()
            level = level * np.where(gap == 0, 0.6, np.where((gap >= -7) & (gap < 0), 1.12,
                                                            np.where((gap > 0) & (gap <= 3), 1.05, 1.0)))
    years = (dates - pd.Timestamp(START)).days.to_numpy() / 365.25
    level = level * (1 + TREND_PER_YEAR * years) * np.exp(rng.normal(0, NOISE, len(dates)))
    return level / level.mean()


def booking_curve(window, km=1000.0):
    """Share of passengers booking 0..window days before departure. Longer trips book a little earlier."""
    days = np.arange(window + 1)
    advance = np.exp(-days / (EARLY_DECAY_DAYS * (0.7 + km / 2000))) + LONG_TAIL_WEIGHT * (1 + km / 2000)
    advance[:2] = 0
    advance *= (1 - DEPARTURE_DAY_SHARE - TATKAL_SHARE) / advance.sum()
    advance[0], advance[1] = DEPARTURE_DAY_SHARE, TATKAL_SHARE
    return advance


def od_mix():
    """Journeys between stops and their share of passengers, from a gravity model on the key stations."""
    stops = route()
    km = stops.set_index("code")["km"]
    nodes = list(NODE_SIZE)
    node_km = km.loc[nodes].to_numpy()
    node_of = {c: int(np.abs(node_km - d).argmin()) for c, d in km.items()}
    rows = []
    for i, o in enumerate(nodes):
        for j, d in enumerate(nodes):
            if j > i:
                rows.append((o, d, NODE_SIZE[o] * NODE_SIZE[d] / np.sqrt(node_km[j] - node_km[i])))
    pairs = pd.DataFrame(rows, columns=["o", "d", "weight"])
    # spread each node over its own stops: the node station itself plus smaller nearby stations
    members = {n: [c for c in km.index if nodes[node_of[c]] == n] for n in nodes}
    expanded = []
    for o, d, w in pairs.itertuples(index=False):
        for so in members[o]:
            for sd in members[d]:
                wo = HUB_SHARE if so == o else (1 - HUB_SHARE) / max(len(members[o]) - 1, 1)
                wd = HUB_SHARE if sd == d else (1 - HUB_SHARE) / max(len(members[d]) - 1, 1)
                expanded.append((so, sd, w * wo * wd, km[sd] - km[so]))
    out = pd.DataFrame(expanded, columns=["brdpt_code", "resupto_code", "share", "km"])
    out["share"] /= out["share"].sum()
    return out


def make_bookings(seed=0):
    """One row per ticket: booking_date, journey_date, trnno, brdpt_code, resupto_code, cls, PSGN."""
    rng = np.random.default_rng([seed, 2])
    dates = pd.date_range(START, END, freq="D")
    passengers = rng.poisson(BASE_PASSENGERS * demand_levels(dates, seed))
    mean_ticket = (TICKET_SIZES * TICKET_PROBS).sum()
    tickets = rng.poisson(passengers / mean_ticket)
    journey = np.repeat(dates.to_numpy(), tickets)
    n = len(journey)
    psgn = rng.choice(TICKET_SIZES, size=n, p=TICKET_PROBS)
    mix = od_mix()
    pick = rng.choice(len(mix), size=n, p=mix["share"].to_numpy())

    # booking lead time: 120-day window for departures before the rule change took full effect, 60 after
    new_rule = journey >= np.datetime64(pd.Timestamp(WINDOW_CHANGE) + pd.Timedelta(days=60))
    days_out = np.empty(n, dtype=int)
    km_bins = np.digitize(mix["km"].to_numpy()[pick], [300, 700, 1200])
    for window, rule in [(120, ~new_rule), (60, new_rule)]:
        for b, km in enumerate([200, 500, 950, 1600]):
            rows = rule & (km_bins == b)
            curve = booking_curve(window, km)
            days_out[rows] = rng.choice(window + 1, size=rows.sum(), p=curve)

    out = pd.DataFrame({
        "booking_date": journey - days_out.astype("timedelta64[D]"),
        "journey_date": journey,
        "trnno": "12138",
        "brdpt_code": mix["brdpt_code"].to_numpy()[pick],
        "resupto_code": mix["resupto_code"].to_numpy()[pick],
        "cls": "3A",
        "PSGN": psgn,
    })
    return out.sort_values(["journey_date", "booking_date"], ignore_index=True)


def daily_totals(bookings):
    """Total passengers per departure, the target of the demand forecast."""
    daily = bookings.groupby("journey_date", as_index=False)["PSGN"].sum().rename(columns={"PSGN": "total_psgn"})
    daily["journey_date"] = pd.to_datetime(daily["journey_date"])
    return daily
