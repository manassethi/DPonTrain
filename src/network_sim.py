"""
Network (origin-destination) pricing for one 3A coach of train 12138.

pricing_sim.py treats the coach as a single product. In reality one seat is sold
many times over on one run: Firozpur -> New Delhi overnight, then New Delhi ->
Mumbai the next morning. Airlines price this as a network: every origin-destination
(O&D) itinerary is its own product, and products compete for seats on the legs
they share.

The model:
  * Route: train 12138's stopping pattern (data/route_12138.csv, from the public timetable),
    collapsed to a handful of key stations ("nodes"). Each of the 52 real stops is
    mapped to its nearest node by distance, so the coach has len(nodes) - 1 legs.
  * O&D demand mix: the share of 3A passengers on each node-to-node O&D, measured
    from the booking records (boarding and destination station of every booking).
  * Booking curve per O&D: measured from the bookings, shrunk towards the train-wide
    curve for thin O&Ds.
  * Day-to-day demand: the same daily demand levels (and forecasts) as pricing_sim.py,
    applied to every O&D.
  * Fares: Indian Railways' published distance-based 3A fare for each O&D (see
    fare_for_distance); each O&D can be priced at the same multiples of its base fare
    as in pricing_sim.py (0.85x .. 1.30x).
  * Price response: the same constant-elasticity assumption as pricing_sim.py.

Policies compared on the same random customers:
  * NetworkFixedFare      - today's practice: every O&D at its base fare, first come first served
  * StaticNetworkFares    - one price per O&D for the whole window, chosen by the network LP
                            from the forecast, then first come first served (no seat control)
  * SegmentOptimizer      - pricing_sim.ForecastOptimizer run separately for every O&D, ignoring
                            the other O&Ds that compete for the same seats. Either every O&D
                            sees all seats left on its path ("leg-agnostic"), or each O&D gets
                            a fixed seat quota up front (like Indian Railways' station quotas)
  * BidPriceLP            - network bid prices from a deterministic LP, re-solved every few days
  * BidPriceDecomposition - network bid prices from a dynamic program per leg
                            (DP decomposition); seat values move with every booking
  * PlansWith             - any of the above, planning with a different (wrong) model of demand

Bid-price logic (the displacement cost): selling an O&D uses one seat on each leg it
crosses. The bid price of a leg is what one more seat on that leg is expected to earn
later. An O&D is worth selling only at a fare above the sum of the bid prices of its legs,
and the best fare maximises expected bookings x (fare - that sum).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import poisson

from pricing_sim import (BASE_IDX, CAPACITY, FARE_LEVELS, HORIZON, Market, booking_curve_from_bookings,
                         optimal_fares)

# --- Network assumptions ------------------------------------------------------
NODES = ["FZR", "BTI", "NDLS", "AGC", "VGLJ", "BPL", "BSL", "NK", "CSMT"]   # key stations, in route order
AVG_PEAK_DEMAND_AT_BASE = 75   # average bookings on the busiest leg at base fares, per 90 seats
CLOSED = -1                    # fare index meaning "not for sale today"
CURVE_SHRINKAGE = 500          # passengers' worth of weight the train-wide booking curve gets per O&D


# 3A basic fare of Mail/Express trains (₹) at the top of selected distance slabs, read from the
# IRCTC fare table effective 1 Jul 2025 (published by Indian Railways). 3A charges at least 300 km.
FARE_TABLE_KM = np.array([300, 310, 400, 500, 600, 700, 800, 910, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                          1800, 1900, 2000, 2100])
FARE_TABLE_3A = np.array([446, 498, 604, 718, 854, 954, 1046, 1139, 1216, 1286, 1354, 1424, 1492, 1562, 1630, 1700,
                          1768, 1838, 1908, 1951])
FARE_ADD_ONS = 40 + 45          # reservation fee + superfast surcharge for 3A (GST left out)


def fare_for_distance(km):
    """
    3A fare (₹) for a journey of `km` on train 12138: the published basic fare, interpolated between
    the slabs above, plus reservation fee and superfast surcharge. Short trips cost more per km
    (e.g. ~₹680 Firozpur-New Delhi, ~₹1,700 New Delhi-Mumbai, ~₹1,960 end to end).
    """
    basic = np.interp(np.asarray(km, dtype=float), FARE_TABLE_KM, FARE_TABLE_3A)
    return np.round(basic) + FARE_ADD_ONS


# --- Calibration from data ----------------------------------------------------
def station_order(route_path):
    """Stops in route order, with distance from origin (km), from a CSV with columns code, km."""
    return pd.read_csv(route_path)[["code", "km"]]


def map_to_nodes(stations, nodes=NODES):
    """Map every stop to the nearest node (by distance along the route). Returns {station: node index}."""
    km = stations.set_index("code")["km"]
    node_km = km.loc[nodes].to_numpy()
    return {code: int(np.abs(node_km - d).argmin()) for code, d in km.items()}


def od_bookings(bookings, stations, nodes=NODES):
    """Bookings with origin / destination node indices; trips that start and end at the same node are dropped."""
    to_node = map_to_nodes(stations, nodes)
    out = bookings.assign(o=bookings["brdpt_code"].map(to_node), d=bookings["resupto_code"].map(to_node))
    out = out.dropna(subset=["o", "d"]).astype({"o": int, "d": int})
    return out[out["o"] < out["d"]]


@dataclass
class Network:
    """Everything the simulator needs to know about the route, its legs and its O&D products."""
    nodes: list
    node_km: np.ndarray            # distance of each node from origin
    od: np.ndarray                 # (J, 2) origin and destination node index of each O&D
    share: np.ndarray              # (J,) share of passengers on each O&D at base fares
    curves: np.ndarray             # (J, HORIZON + 1) booking curve of each O&D
    elasticity: float = 1.2
    peak_demand_at_base: float = AVG_PEAK_DEMAND_AT_BASE
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.n_legs, self.n_od = len(self.nodes) - 1, len(self.od)
        legs = np.arange(self.n_legs)
        # incidence[l, j] is True when O&D j uses leg l
        self.incidence = (legs[:, None] >= self.od[:, 0]) & (legs[:, None] < self.od[:, 1])
        self.km = self.node_km[self.od[:, 1]] - self.node_km[self.od[:, 0]]
        self.base_fare = fare_for_distance(self.km)
        self.price_effect = FARE_LEVELS ** -self.elasticity
        # scale O&D shares so the busiest leg expects `peak_demand_at_base` bookings at base fares
        self.demand = self.share * self.peak_demand_at_base / (self.incidence @ self.share).max()
        self.share_from = np.cumsum(self.curves, axis=1)       # demand still to come from day d (incl. d)
        self.leg_km = np.diff(self.node_km)
        self.labels = [f"{self.nodes[o]}-{self.nodes[d]}" for o, d in self.od]

    def mean_arrivals(self, demand_level, days_out, fare_idx):
        """Expected bookings per O&D on one day; fare_idx has shape (..., J), CLOSED means none."""
        fare_idx = np.asarray(fare_idx)
        lam = demand_level * self.demand * self.curves[:, days_out] * self.price_effect[np.maximum(fare_idx, 0)]
        return np.where(fare_idx == CLOSED, 0.0, lam)

    def leg_load(self):
        """Expected bookings on each leg at base fares, average departure."""
        return self.incidence @ self.demand


def network_from_bookings(bookings, stations, nodes=NODES, share_bookings=None, **kwargs):
    """
    Build the Network from 3A booking records.
    bookings        - used for the per-O&D booking curves (choose a period with one booking-window rule)
    share_bookings  - used for the O&D mix (defaults to `bookings`)
    """
    ods = od_bookings(bookings, stations, nodes)
    mix = od_bookings(share_bookings if share_bookings is not None else bookings, stations, nodes)
    pax = mix.groupby(["o", "d"])["PSGN"].sum()
    overall = booking_curve_from_bookings(ods)
    curves, counts = [], ods.groupby(["o", "d"])["PSGN"].sum()
    for o, d in pax.index:
        n = counts.get((o, d), 0)
        own = booking_curve_from_bookings(ods[(ods.o == o) & (ods.d == d)]) if n else overall
        curves.append((n * own + CURVE_SHRINKAGE * overall) / (n + CURVE_SHRINKAGE))
    node_km = stations.set_index("code")["km"].loc[nodes].to_numpy()
    return Network(list(nodes), node_km, np.array(pax.index.tolist()), (pax / pax.sum()).to_numpy(),
                   np.array(curves), **kwargs)


# --- Simulation ---------------------------------------------------------------
@dataclass
class NetworkState:
    """What a pricing policy can see on a given day, for n parallel booking windows."""
    days_out: int
    seats_left: np.ndarray        # (n, L) seats left on each leg
    sold: np.ndarray              # (n, J) bookings so far per O&D
    forecast: float               # demand level forecast before booking opened (1.0 = average)


def random_customers(n_runs, n_od, seed):
    """Uniforms for inverse-CDF demand draws, and the order O&Ds' customers arrive in on each day."""
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_runs, HORIZON + 1, n_od))
    order = rng.permuted(np.tile(np.arange(n_od), (HORIZON + 1, 1)), axis=1)
    return uniforms, order


def simulate_network(policy, net, demand_level, forecast_level, n_runs=1, seed=0, customers=None):
    """
    Run n booking windows for one departure. `customers` = (uniforms, order) from random_customers,
    so different policies can be fed exactly the same random customers. Within a day, O&Ds take
    seats in `order`; a booking needs a free seat on every leg of its O&D.
    Returns revenue (n,), bookings per O&D (n, J) and the fare index path (n, HORIZON + 1, J).
    """
    uniforms, order = customers if customers is not None else random_customers(n_runs, net.n_od, seed)
    n = uniforms.shape[0]
    state = NetworkState(HORIZON, np.full((n, net.n_legs), CAPACITY), np.zeros((n, net.n_od), int), forecast_level)
    revenue = np.zeros(n)
    fares = np.empty((n, HORIZON + 1, net.n_od), dtype=int)
    legs_of = [np.flatnonzero(net.incidence[:, j]) for j in range(net.n_od)]
    for step, days_out in enumerate(range(HORIZON, -1, -1)):
        state.days_out = days_out
        a = np.broadcast_to(policy.choose(state, net), (n, net.n_od))
        lam = net.mean_arrivals(demand_level, days_out, a)
        arrivals = poisson.ppf(uniforms[:, step], lam).clip(min=0).astype(int)    # ppf(0) is -1
        if hasattr(policy, "booking_limit"):          # e.g. seats left in each O&D's quota
            arrivals = np.minimum(arrivals, policy.booking_limit(state, net))
        for j in order[step]:
            sold = np.minimum(arrivals[:, j], state.seats_left[:, legs_of[j]].min(axis=1))
            state.seats_left[:, legs_of[j]] -= sold[:, None]
            state.sold[:, j] += sold
            revenue += sold * net.base_fare[j] * FARE_LEVELS[np.maximum(a[:, j], 0)]
        fares[:, step] = a
    return revenue, state.sold, fares


# --- Network optimisation -----------------------------------------------------
def network_lp(net, demand_level, seats_left, days_out):
    """
    Deterministic LP over the rest of the booking window: treat expected demand as certain and
    choose, for every O&D, what share of the remaining window to sell it at each fare level
    (shares may add to less than 1: the rest of the time the O&D is closed), so that no leg
    sells more seats than it has left.
    Returns (shares (J, fare levels), leg bid prices (L,) in ₹ per seat, expected revenue).
    The bid prices are the LP's shadow prices: the revenue one more seat on that leg would add.
    """
    remaining = demand_level * net.demand * net.share_from[:, days_out]
    sales = remaining[:, None] * net.price_effect[None, :]                 # (J, A) if always offered
    revenue = sales * net.base_fare[:, None] * FARE_LEVELS[None, :]
    n_od, n_fares = sales.shape
    leg_rows = (net.incidence[:, :, None] * sales[None]).reshape(net.n_legs, n_od * n_fares)
    od_rows = np.kron(np.eye(n_od), np.ones(n_fares))
    res = linprog(-revenue.ravel(), A_ub=np.vstack([leg_rows, od_rows]),
                  b_ub=np.concatenate([np.asarray(seats_left, float), np.ones(n_od)]), bounds=(0, None),
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"network LP failed (status {res.status}): {res.message}")
    return res.x.reshape(n_od, n_fares), -res.ineqlin.marginals[:net.n_legs], -res.fun


def best_fares(net, displacement):
    """
    For each O&D, the fare that maximises expected bookings x (fare - displacement cost),
    where the displacement cost is the sum of the bid prices of the legs it uses.
    CLOSED when no fare covers the displacement cost.
    """
    margin = net.price_effect * (net.base_fare[:, None] * FARE_LEVELS - displacement[..., None])
    return np.where(margin.max(axis=-1) > 0, margin.argmax(axis=-1), CLOSED)


NO_SEAT = 1e9   # bid price of a leg with no seats left


def leg_bid_prices(net, demand_level, step_prob=0.05):
    """
    Bid prices by DP decomposition: one dynamic program per leg.
    Each leg is priced as if it were alone, with the other legs' seats valued at the LP's
    (static) bid prices: an O&D's fare is "prorated" to this leg by subtracting the bid prices
    of its other legs. Each day is split into small steps with at most one booking request
    (probability <= step_prob), so the leg's DP is exact for that leg.
    Returns bids[days_out, leg, seats_left] = expected value of the last seat (₹) at the start of the day.
    """
    key = ("leg_dp", round(float(demand_level), 2))
    if key in net._cache:
        return net._cache[key]
    demand_level = key[1]
    _, lp_bids, _ = network_lp(net, demand_level, np.full(net.n_legs, CAPACITY), HORIZON)
    bids = np.full((HORIZON + 1, net.n_legs, CAPACITY + 1), NO_SEAT)
    for leg in range(net.n_legs):
        js = np.flatnonzero(net.incidence[leg])
        others = lp_bids @ net.incidence[:, js] - lp_bids[leg]
        prorated = net.base_fare[js, None] * FARE_LEVELS - others[:, None]            # (J_leg, A)
        value = np.zeros(CAPACITY + 1)                                                # value[x] with x seats left
        for days_out in range(HORIZON + 1):                                           # backwards from departure
            lam = demand_level * net.demand[js, None] * net.curves[js, days_out, None] * net.price_effect
            steps = max(1, int(np.ceil(lam[:, 0].sum() / step_prob)))
            p = (lam / steps)[:, :, None]
            for _ in range(steps):
                seat_value = np.diff(value)                                           # value of the x-th seat
                gain = (p * (prorated[:, :, None] - seat_value)).max(axis=1).clip(min=0).sum(axis=0)
                value[1:] += gain
            bids[days_out, leg, 1:] = np.diff(value)
    net._cache[key] = bids
    return bids


def lp_upper_bound(net, demand_level):
    """Expected revenue if demand were known and certain: an (optimistic) upper bound for any policy."""
    return network_lp(net, demand_level, np.full(net.n_legs, CAPACITY), HORIZON)[2]


# --- Policies -----------------------------------------------------------------
class NetworkFixedFare:
    name = "Fixed fare (today)"

    def choose(self, state, net):
        return np.full(net.n_od, BASE_IDX)


class StaticNetworkFares:
    """
    One price per O&D for the whole window, set by the network LP from the forecast,
    then first come first served: no seat control while bookings come in.
    An O&D the LP would rather not sell at all is priced at the top fare.
    """
    name = "Network LP prices, first come first served"

    def choose(self, state, net):
        key = ("static", round(float(state.forecast), 2))
        if key not in net._cache:
            shares, _, _ = network_lp(net, key[1], np.full(net.n_legs, CAPACITY), HORIZON)
            best = shares.argmax(axis=1)
            net._cache[key] = np.where(shares.sum(axis=1) > 1e-6, best, len(FARE_LEVELS) - 1)
        return net._cache[key]


def segment_quotas(net):
    """
    Split the coach into a fixed seat quota per O&D, in proportion to its expected demand on its
    busiest leg. Seats lost to rounding down go to the largest remainders, as long as every leg
    stays within CAPACITY.
    """
    if "quotas" in net._cache:
        return net._cache["quotas"]
    load = net.leg_load()
    busiest = np.where(net.incidence, load[:, None], 0).max(axis=0)
    exact = CAPACITY * net.demand / busiest
    quotas = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - quotas)):
        if (net.incidence @ quotas + net.incidence[:, j])[net.incidence[:, j]].max() <= CAPACITY:
            quotas[j] += 1
    net._cache["quotas"] = quotas
    return quotas


class SegmentOptimizer:
    """
    pricing_sim.ForecastOptimizer applied to each O&D on its own: a dynamic program on that O&D's
    forecast demand. It never looks at the other O&Ds that use the same seats.

    quotas=False - each O&D's capacity is the seats left on its path: every O&D acts as if the
                   seats were its alone ("leg-agnostic, shared seats").
    quotas=True  - each O&D is given a fixed seat quota up front (segment_quotas, like
                   Indian Railways' station quotas) and prices to fill it.
    """

    def __init__(self, quotas=False, name=None):
        self.quotas = quotas
        self.name = name or ("Per-segment optimizer, fixed seat quotas" if quotas
                             else "Per-segment optimizer, leg-agnostic")

    def choose(self, state, net):
        if "segment_markets" not in net._cache:
            net._cache["segment_markets"] = [Market(net.curves[j], elasticity=net.elasticity,
                                                    demand_at_base=net.demand[j]) for j in range(net.n_od)]
        markets = net._cache["segment_markets"]
        seats = np.where(net.incidence.T[None], state.seats_left[:, None, :], CAPACITY).min(axis=-1)  # (n, J)
        if self.quotas:
            seats = np.minimum(seats, self.booking_limit(state, net))
        out = np.empty(seats.shape, dtype=int)
        for j, market in enumerate(markets):
            best, _ = optimal_fares(market, state.forecast)
            out[:, j] = best[state.days_out, seats[:, j]]
        if self.quotas:
            out = np.where(seats > 0, out, CLOSED)       # quota used up: the O&D shows as sold out
        return out

    def booking_limit(self, state, net):
        """Bookings each O&D may still take: what is left of its quota (no limit without quotas)."""
        if not self.quotas:
            return np.full(state.sold.shape, CAPACITY)
        return np.maximum(segment_quotas(net) - state.sold, 0)


class BidPriceLP:
    """Network bid prices from the deterministic LP, re-solved every `resolve_every` days for each booking window."""

    def __init__(self, resolve_every=7, name=None):
        self.resolve_every = resolve_every
        self.name = name or f"Network bid prices (LP, re-solved every {resolve_every} days)"
        self.bids = None

    def choose(self, state, net):
        if self.bids is None or (HORIZON - state.days_out) % self.resolve_every == 0:
            rows, inverse = np.unique(state.seats_left, axis=0, return_inverse=True)
            solved = np.array([network_lp(net, state.forecast, seats, state.days_out)[1] for seats in rows])
            self.bids = solved[inverse.ravel()]
        bids = np.where(state.seats_left > 0, self.bids, NO_SEAT)
        return best_fares(net, bids @ net.incidence)


class BidPriceDecomposition:
    """Network bid prices from one DP per leg; seat values update with every booking."""

    def __init__(self, true_level=None, name=None):
        self.true_level = true_level
        self.name = name or "Network bid prices (DP by leg)"

    def choose(self, state, net):
        level = state.forecast if self.true_level is None else self.true_level
        bids = leg_bid_prices(net, level)[state.days_out]                     # (L, seats)
        leg_bids = bids[np.arange(net.n_legs), state.seats_left]              # (n, L)
        return best_fares(net, leg_bids @ net.incidence)


class PlansWith:
    """
    Wraps a policy so it plans with a different Network (e.g. a wrong elasticity) from the one
    customers actually follow in the simulation: a test of model misspecification.
    """

    def __init__(self, policy, plan_net, name=None):
        self.policy, self.plan_net = policy, plan_net
        self.name = name or policy.name
        if hasattr(policy, "booking_limit"):
            self.booking_limit = lambda state, net: policy.booking_limit(state, plan_net)

    def choose(self, state, net):
        return self.policy.choose(state, self.plan_net)


# --- Evaluation ---------------------------------------------------------------
def evaluate_network(policies, net, levels, forecasts, runs_per_departure=20, seed=7, with_ceiling=True):
    """
    Simulate every policy on exactly the same random customers for each test departure.
    Returns one row per (policy, departure, run). `lp_bound` is the departure's LP upper bound.
    """
    frames = []
    seat_km = CAPACITY * net.leg_km.sum()
    for i, (date, level) in enumerate(levels.items()):
        forecast = forecasts.iloc[i]
        customers = random_customers(runs_per_departure, net.n_od, [seed, i])
        day_policies = list(policies) + ([BidPriceDecomposition(level, "Bid prices, told true demand")]
                                         if with_ceiling else [])
        bound = lp_upper_bound(net, level)
        for policy in day_policies:
            revenue, sold, fares = simulate_network(policy, net, level, forecast, customers=customers)
            frames.append(pd.DataFrame({"policy": policy.name, "journey_date": date,
                                        "run": np.arange(runs_per_departure), "revenue": revenue,
                                        "passengers": sold.sum(axis=1), "seat_km_load_%": 100 * sold @ net.km / seat_km,
                                        "closed_%": 100 * (fares == CLOSED).mean(axis=(1, 2)),
                                        "demand_level": level, "lp_bound": bound}))
    return pd.concat(frames, ignore_index=True)


def summarize_network(results, baseline=NetworkFixedFare.name):
    """Average revenue, passengers and seat-km load factor per departure, with uplift vs `baseline`."""
    s = results.groupby("policy").agg(revenue=("revenue", "mean"), passengers=("passengers", "mean"),
                                      seat_km_load=("seat_km_load_%", "mean"), closed=("closed_%", "mean"))
    s = s.rename(columns={"seat_km_load": "seat_km_load_%", "closed": "closed_%"})
    base = s.loc[baseline]
    s["revenue_per_pax"] = s["revenue"] / s["passengers"]
    s["uplift_vs_fixed_%"] = 100 * (s["revenue"] / base["revenue"] - 1)
    s["passengers_vs_fixed_%"] = 100 * (s["passengers"] / base["passengers"] - 1)
    paired = results.pivot_table(index=["journey_date", "run"], columns="policy", values="revenue")
    s["uplift_ci_%"] = pd.Series({p: 1.96 * 100 * (paired[p] / paired[baseline] - 1).std()
                                  / np.sqrt(len(paired)) for p in s.index})
    bound = results.groupby("journey_date")["lp_bound"].first().mean()
    s["share_of_lp_bound_%"] = 100 * s["revenue"] / bound
    return s.sort_values("revenue").round(1)
