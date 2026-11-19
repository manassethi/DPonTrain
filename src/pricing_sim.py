"""
Dynamic pricing simulator and pricing policies for one 3A coach.

The simulator is calibrated from booking records (one row per ticket):
  * Booking curve: the share of passengers who book d days before departure,
    measured from bookings made under the 60-day advance reservation rule.
  * Day-to-day demand: how busy each departure is, from the daily passenger
    totals, so weekends, peak season and quiet mid-week days behave as they
    did in the data.

This public version runs on synthetic bookings (see synthetic_data.py); the
original study used confidential Indian Railways records for train 12138.

Price response is an assumption, not something booking data can tell us (fares
never changed). We use a constant-elasticity curve,
D(p) = D(base) * (p / base) ** -elasticity, and test a range of elasticities.

Policies compared on the same random demand:
  * FixedFare          - today's practice: the base fare every day
  * PacingRule         - a simple airline-style rule: raise the fare when selling
                         ahead of plan, lower it when behind
  * QLearningAgent     - tabular reinforcement learning
  * ForecastOptimizer  - forecast the departure's demand, then set fares by
                         dynamic programming (classic revenue management).
                         Optionally re-forecasts every day from bookings so far,
                         caps fares, or values each passenger carried.
  * PerfectInformation - the optimizer told the true demand level
                         (the ceiling; not deployable)

Everything is vectorised: each call simulates many booking windows at once.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- Market assumptions -------------------------------------------------------
CAPACITY = 90                 # seats in one 3A coach
BASE_FARE = 1100.0            # ₹, current fixed fare
HORIZON = 60                  # booking window in days
AVG_DEMAND_AT_BASE = 75       # average bookings per departure at the base fare (team's assumption)
FARE_LEVELS = np.array([0.85, 0.90, 1.00, 1.10, 1.20, 1.30])  # x base fare; +30% cap for fairness
BASE_IDX = int(np.where(FARE_LEVELS == 1.0)[0][0])
DEFAULT_ELASTICITY = 1.2


@dataclass
class Market:
    """Everything the simulator needs to know about one route."""
    booking_curve: np.ndarray      # booking_curve[d] = share of demand arriving d days out (d = 0..HORIZON)
    elasticity: float = DEFAULT_ELASTICITY
    demand_at_base: float = AVG_DEMAND_AT_BASE
    _dp_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.price_effect = FARE_LEVELS ** -self.elasticity
        # share of demand that arrives before / from a given day onward
        self.share_before = np.array([self.booking_curve[d + 1:].sum() for d in range(HORIZON + 1)])
        self.share_from = np.array([self.booking_curve[:d + 1].sum() for d in range(HORIZON + 1)])

    def mean_arrivals(self, demand_level, days_out, fare_idx):
        """Expected bookings on one day (works element-wise on arrays)."""
        return self.demand_at_base * demand_level * self.booking_curve[days_out] * self.price_effect[fare_idx]


@dataclass
class BookingState:
    """What a pricing policy can see on a given day, for n parallel booking windows."""
    days_out: int
    seats_left: np.ndarray        # (n,)
    sold: np.ndarray              # (n,) bookings so far
    exposure: np.ndarray          # (n,) bookings we would have expected so far for demand level 1.0,
                                  #      given the fares actually charged
    forecast: float               # demand level forecast before booking opened (1.0 = average)


# --- Calibration from data ----------------------------------------------------
def booking_curve_from_bookings(bookings: pd.DataFrame) -> np.ndarray:
    """Share of passengers booking 0..HORIZON days before departure."""
    days_out = (pd.to_datetime(bookings["journey_date"]) - pd.to_datetime(bookings["booking_date"])).dt.days
    shares = bookings.groupby(days_out.clip(0, HORIZON))["PSGN"].sum()
    shares = shares.reindex(range(HORIZON + 1), fill_value=0)
    return (shares / shares.sum()).to_numpy()


def demand_levels(daily: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """How busy each departure is, relative to the reference period's average (1.0 = average)."""
    return (daily.set_index("journey_date")["total_psgn"] / reference["total_psgn"].mean()).rename("level")


def forecast_levels(daily: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """
    The forecast available before booking opens: the blend of "weekday average"
    and "same weekday last year" that scored best in DemandForecast.ipynb.
    `history` is every departure known so far (used for the weekday averages and last year).
    """
    history = history.set_index("journey_date")["total_psgn"].sort_index()
    weekday_avg = history.groupby(history.index.day_of_week).mean()
    dates = pd.DatetimeIndex(daily["journey_date"])
    weekday = weekday_avg.reindex(dates.day_of_week).to_numpy()
    last_year = history.reindex(dates - pd.Timedelta(days=364)).to_numpy()
    blend = np.where(np.isnan(last_year), weekday, 0.5 * weekday + 0.5 * last_year)
    return pd.Series(blend / history.mean(), index=dates, name="forecast")


# --- Simulation ---------------------------------------------------------------
def simulate(policy, market, demand_level, forecast_level, n_runs=1, seed=0, uniforms=None):
    """
    Run n booking windows for one departure. Demand is drawn by inverse-CDF from
    `uniforms` (shape (n_runs, HORIZON + 1)), so different policies can be fed
    exactly the same random customers.
    Returns revenue, seats sold, number of fare changes and the fare path, per run.
    """
    if uniforms is None:
        uniforms = np.random.default_rng(seed).random((n_runs, HORIZON + 1))
    n = uniforms.shape[0]
    state = BookingState(HORIZON, np.full(n, CAPACITY), np.zeros(n, int), np.zeros(n), forecast_level)
    revenue = np.zeros(n)
    fares = np.empty((n, HORIZON + 1), dtype=int)
    for step, days_out in enumerate(range(HORIZON, -1, -1)):
        state.days_out = days_out
        a = np.broadcast_to(policy.choose(state, market), (n,))
        lam = market.mean_arrivals(demand_level, days_out, a)
        arrivals = poisson.ppf(uniforms[:, step], lam).clip(min=0).astype(int)   # ppf(0) is -1
        sold = np.minimum(arrivals, state.seats_left)
        revenue += sold * BASE_FARE * FARE_LEVELS[a]
        state.seats_left = state.seats_left - sold
        state.sold = state.sold + sold
        state.exposure = state.exposure + market.mean_arrivals(1.0, days_out, a)
        fares[:, step] = a
    changes = (np.diff(fares, axis=1) != 0).sum(axis=1)
    return revenue, CAPACITY - state.seats_left, changes, BASE_FARE * FARE_LEVELS[fares]


# --- Policies -----------------------------------------------------------------
class FixedFare:
    name = "Fixed fare (today)"

    def __init__(self, fare_idx=BASE_IDX):
        self.fare_idx = fare_idx

    def choose(self, state, market):
        return self.fare_idx


class PacingRule:
    """Start at a fare matching the forecast; step up if 10% ahead of plan, down if 10% behind."""
    name = "Pacing rule"

    def choose(self, state, market):
        start = BASE_IDX + int(np.clip(round((state.forecast - 1) / 0.08), -2, 3))
        plan = state.forecast * state.exposure
        pace = np.where(plan >= 3, state.sold / np.maximum(plan, 1e-9), 1.0)
        step = np.where(pace > 1.1, 1, np.where(pace < 0.9, -1, 0))
        # Protect the last seats: fewer seats left than expected remaining demand -> raise.
        remaining = market.demand_at_base * state.forecast * market.share_from[state.days_out]
        step = step + (state.seats_left < 0.8 * remaining)
        return np.clip(start + step, 0, len(FARE_LEVELS) - 1)


class QLearningAgent:
    """
    Tabular Q-learning. The state is what a revenue manager watches:
    time to departure, seats left, booking pace vs plan, and the demand forecast.
    """
    name = "Q-learning agent"
    DAY_EDGES = [1, 3, 7, 14, 30, 45]          # buckets of days to departure
    SEAT_EDGES = [5, 15, 30, 45, 60, 75]       # buckets of seats left
    PACE_EDGES = [0.85, 1.15]                  # behind / on plan / ahead
    FORECAST_EDGES = [0.93, 1.07]              # quiet / normal / busy departure

    def __init__(self, seed=0):
        shape = (len(self.DAY_EDGES) + 1, len(self.SEAT_EDGES) + 1, len(self.PACE_EDGES) + 1,
                 len(self.FORECAST_EDGES) + 1, len(FARE_LEVELS))
        self.q = np.zeros(shape)
        self.visits = np.zeros(shape)
        self.epsilon = 0.0
        self.rng = np.random.default_rng(seed)

    def _state(self, days_out, seats_left, sold, exposure, forecast, market):
        plan = forecast * exposure
        pace = np.where(plan >= 3, sold / np.maximum(plan, 1e-9), 1.0)
        n = len(seats_left)
        return (np.full(n, np.digitize(days_out, self.DAY_EDGES)), np.digitize(seats_left, self.SEAT_EDGES),
                np.digitize(pace, self.PACE_EDGES), np.digitize(forecast, self.FORECAST_EDGES))

    def choose(self, state, market):
        s = self._state(state.days_out, state.seats_left, state.sold, state.exposure,
                        np.broadcast_to(state.forecast, state.seats_left.shape), market)
        return self.q[s].argmax(axis=-1)

    def train(self, market, levels, forecasts, episodes=60_000, batch=50, seed=1):
        """Learn from simulated booking windows, `batch` departures at a time."""
        rng = np.random.default_rng(seed)
        levels, forecasts = np.asarray(levels), np.asarray(forecasts)
        for start in range(0, episodes, batch):
            self.epsilon = max(0.02, 1.0 - start / (0.7 * episodes))
            pick = rng.integers(len(levels), size=batch)
            level, forecast = levels[pick], forecasts[pick]
            seats, sold, exposure = np.full(batch, CAPACITY), np.zeros(batch, int), np.zeros(batch)
            active = np.ones(batch, bool)
            for days_out in range(HORIZON, -1, -1):
                s = self._state(days_out, seats, sold, exposure, forecast, market)
                a = self._explore(s)
                got = np.minimum(rng.poisson(market.mean_arrivals(level, days_out, a)), seats)
                reward = got * FARE_LEVELS[a]                   # revenue in units of the base fare
                seats, sold = seats - got, sold + got
                exposure = exposure + market.mean_arrivals(1.0, days_out, a)
                done = (days_out == 0) | (seats == 0)
                if days_out > 0:
                    nxt = self.q[self._state(days_out - 1, seats, sold, exposure, forecast, market)].max(axis=-1)
                else:
                    nxt = np.zeros(batch)
                target = reward + np.where(done, 0.0, nxt)
                idx = tuple(x[active] for x in s) + (a[active],)
                np.add.at(self.visits, idx, 1)
                alpha = np.maximum(0.01, self.visits[idx] ** -0.6)
                # several episodes in the batch can hit the same (state, fare): average their updates
                dupes = np.zeros(self.q.shape)
                np.add.at(dupes, idx, 1)
                np.add.at(self.q, idx, alpha * (target[active] - self.q[idx]) / dupes[idx])
                active &= ~done
                if not active.any():
                    break
        self.epsilon = 0.0
        return self

    def _explore(self, s):
        """Epsilon-greedy: mostly the best-known fare, sometimes a random one."""
        greedy = self.q[s].argmax(axis=-1)
        explore = self.rng.random(len(greedy)) < self.epsilon
        return np.where(explore, self.rng.integers(len(FARE_LEVELS), size=len(greedy)), greedy)


def optimal_fares(market, demand_level, max_fare_idx=len(FARE_LEVELS) - 1, seat_value=0.0):
    """
    Best fare for every (days to departure, seats left), by dynamic programming,
    assuming the departure's demand level is `demand_level`.

    max_fare_idx  - highest fare level allowed (a fairness cap)
    seat_value    - extra value (₹) the operator places on each passenger carried,
                    on top of the fare. 0 = pure revenue maximisation.
    """
    key = (round(float(demand_level), 2), int(max_fare_idx), float(seat_value))
    if key in market._dp_cache:
        return market._dp_cache[key]
    demand_level = key[0]
    k = np.arange(CAPACITY + 1)
    s_idx, k_idx = np.meshgrid(k, k, indexing="ij")          # rows: seats left, cols: bookings
    feasible = k_idx <= s_idx
    value = np.zeros(CAPACITY + 1)               # after departure, unsold seats are worth nothing
    best = np.zeros((HORIZON + 1, CAPACITY + 1), dtype=int)
    for days_out in range(HORIZON + 1):          # solve backwards from departure day
        options = np.full((len(FARE_LEVELS), CAPACITY + 1), -np.inf)
        for a in range(max_fare_idx + 1):
            pmf = poisson.pmf(k, market.mean_arrivals(demand_level, days_out, a))
            prob = np.where(feasible, pmf[k_idx], 0.0)
            # demand beyond the seats left is lost: put that probability on "sell out"
            prob[k, k] = 1.0 - (np.cumsum(pmf)[k] - pmf[k])
            payoff = k_idx * (FARE_LEVELS[a] * BASE_FARE + seat_value) + value[np.clip(s_idx - k_idx, 0, None)]
            options[a] = (prob * payoff).sum(axis=1)
        best[days_out] = options.argmax(axis=0)
        value = options.max(axis=0)
    market._dp_cache[key] = (best, value)
    return best, value


class ForecastOptimizer:
    """
    Classic revenue management: forecast the departure's demand, then set the fare
    each day by dynamic programming on that forecast.

    reforecast   - update the demand estimate every day from bookings so far
                   (Bayesian pickup: a Gamma prior centred on the forecast, updated by
                   Poisson bookings). `forecast_cv` is how uncertain the forecast is.
    max_fare     - fare cap as a multiple of the base fare
    seat_value   - ₹ of value placed on each passenger carried (fairness / access goal)
    """

    def __init__(self, reforecast=False, forecast_cv=0.10, max_fare=FARE_LEVELS.max(), seat_value=0.0, name=None):
        self.reforecast = reforecast
        self.prior_strength = 1.0 / forecast_cv ** 2
        self.max_idx = int(np.searchsorted(FARE_LEVELS, max_fare + 1e-9) - 1)
        self.seat_value = seat_value
        self.name = name or ("Forecast + optimizer, re-forecasting daily" if reforecast else "Forecast + optimizer")

    def estimate(self, state, market):
        if not self.reforecast:
            return np.full(len(state.seats_left), state.forecast)
        a = self.prior_strength
        b = a / state.forecast
        return (a + state.sold) / (b + state.exposure)

    def choose(self, state, market):
        level = np.round(self.estimate(state, market), 2)
        out = np.empty(len(level), dtype=int)
        for lv in np.unique(level):
            rows = level == lv
            best, _ = optimal_fares(market, lv, self.max_idx, self.seat_value)
            out[rows] = best[state.days_out, state.seats_left[rows]]
        return out


class PerfectInformation:
    """The optimizer told the true demand level. Not deployable; it is the ceiling."""
    name = "Perfect-information ceiling"

    def __init__(self, demand_level):
        self.demand_level = demand_level

    def choose(self, state, market):
        best, _ = optimal_fares(market, self.demand_level)
        return best[state.days_out, state.seats_left]


# --- Evaluation ---------------------------------------------------------------
def evaluate(policies, market, levels, forecasts, runs_per_departure=20, seed=7, with_ceiling=True):
    """
    Simulate every policy on exactly the same random customers for each test departure.
    Returns one row per (policy, departure, run).
    """
    frames = []
    for i, (date, level) in enumerate(levels.items()):
        forecast = forecasts.iloc[i]
        uniforms = np.random.default_rng([seed, i]).random((runs_per_departure, HORIZON + 1))
        day_policies = list(policies) + ([PerfectInformation(level)] if with_ceiling else [])
        for policy in day_policies:
            revenue, sold, changes, _ = simulate(policy, market, level, forecast, uniforms=uniforms)
            frames.append(pd.DataFrame({"policy": policy.name, "journey_date": date,
                                        "run": np.arange(runs_per_departure), "revenue": revenue,
                                        "seats_sold": sold, "fare_changes": changes, "demand_level": level}))
    return pd.concat(frames, ignore_index=True)


def summarize(results):
    """Average revenue, load factor and fare per departure, with uplift vs today's fixed fare."""
    s = results.groupby("policy").agg(revenue=("revenue", "mean"), seats_sold=("seats_sold", "mean"),
                                      fare_changes=("fare_changes", "mean"))
    base = s.loc[FixedFare.name]
    s["load_factor_%"] = 100 * s["seats_sold"] / CAPACITY
    s["avg_fare"] = s["revenue"] / s["seats_sold"]
    s["uplift_vs_fixed_%"] = 100 * (s["revenue"] / base["revenue"] - 1)
    s["passengers_vs_fixed_%"] = 100 * (s["seats_sold"] / base["seats_sold"] - 1)
    # 95% confidence interval on the uplift, pairing each run with the fixed-fare run on the same customers
    paired = results.pivot_table(index=["journey_date", "run"], columns="policy", values="revenue")
    s["uplift_ci_%"] = pd.Series({p: 1.96 * 100 * (paired[p] / paired[FixedFare.name] - 1).std()
                                  / np.sqrt(len(paired)) for p in s.index})
    return s.sort_values("revenue").round(1)
