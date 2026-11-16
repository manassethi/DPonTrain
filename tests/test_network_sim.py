"""Checks the network (O&D) simulator, the network LP and the bid-price policies."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pricing_sim import CAPACITY, FARE_LEVELS, HORIZON, FixedFare, ForecastOptimizer, Market, simulate
from network_sim import (CLOSED, NODES, BidPriceDecomposition, BidPriceLP, Network, NetworkFixedFare, PlansWith,
                         SegmentOptimizer, best_fares, leg_bid_prices, lp_upper_bound, map_to_nodes, network_lp,
                         random_customers, segment_quotas, simulate_network, station_order)


def late_curve():
    curve = np.exp(-np.arange(HORIZON + 1) / 6.0)
    return curve / curve.sum()


def toy_network(peak_demand, elasticity=1.2):
    """Three stations A-B-C: short trips A-B and B-C, and the long trip A-C that needs both legs."""
    return Network(["A", "B", "C"], np.array([0, 400, 1000]), np.array([[0, 1], [1, 2], [0, 2]]),
                   np.array([0.4, 0.3, 0.3]), np.tile(late_curve(), (3, 1)), elasticity=elasticity,
                   peak_demand_at_base=peak_demand)


def one_od_network():
    return Network(["A", "B"], np.array([0, 500]), np.array([[0, 1]]), np.array([1.0]), late_curve()[None],
                   elasticity=1.2, peak_demand_at_base=85)


def same_customers(u):
    return u[:, :, None], np.zeros((HORIZON + 1, 1), dtype=int)


def test_station_order_and_nodes_follow_the_route():
    stations = station_order(ROOT / "data/route_12138.csv")
    assert stations["code"].iloc[0] == "FZR" and stations["code"].iloc[-1] == "CSMT"
    assert stations["km"].is_monotonic_increasing
    to_node = map_to_nodes(stations)
    assert [to_node[n] for n in NODES] == list(range(len(NODES)))
    assert to_node["RKMP"] == NODES.index("BPL")        # Rani Kamalapati is 6 km past Bhopal


def test_network_scaling_and_incidence():
    net = toy_network(peak_demand=90)
    assert net.incidence.tolist() == [[True, False, True], [False, True, True]]
    assert net.leg_load().max() == pytest.approx(90)
    assert (net.base_fare[2] < net.base_fare[0] + net.base_fare[1])   # through fare cheaper than two short fares


def test_single_od_network_matches_single_coach_simulator():
    net, market = one_od_network(), Market(late_curve(), elasticity=1.2, demand_at_base=85)
    u = np.random.default_rng(1).random((500, HORIZON + 1))
    for single, network in [(FixedFare(), NetworkFixedFare()), (ForecastOptimizer(), SegmentOptimizer())]:
        rev, sold, _, _ = simulate(single, market, 1.1, 1.0, uniforms=u)
        net_rev, net_sold, _ = simulate_network(network, net, 1.1, 1.0, customers=same_customers(u))
        assert np.array_equal(net_sold[:, 0], sold)
        assert np.allclose(net_rev / net.base_fare[0], rev / 1100.0)


def test_no_leg_is_ever_oversold():
    net = toy_network(peak_demand=200)
    for policy in [NetworkFixedFare(), SegmentOptimizer(), SegmentOptimizer(quotas=True), BidPriceDecomposition()]:
        _, sold, _ = simulate_network(policy, net, 1.3, 1.0, n_runs=300, seed=2)
        assert (net.incidence @ sold.T).max() <= CAPACITY


def test_quotas_fit_the_coach_and_are_respected():
    net = toy_network(peak_demand=150)
    quotas = segment_quotas(net)
    assert (net.incidence @ quotas).max() <= CAPACITY
    assert (net.incidence @ quotas).max() >= CAPACITY - 1               # rounding loses at most a seat
    _, sold, _ = simulate_network(SegmentOptimizer(quotas=True), net, 1.5, 1.0, n_runs=300, seed=3)
    assert (sold <= quotas).all()


def test_lp_bid_prices_only_where_seats_are_scarce():
    net = toy_network(peak_demand=150)
    _, quiet_bids, _ = network_lp(net, 0.3, np.full(net.n_legs, CAPACITY), HORIZON)
    _, busy_bids, _ = network_lp(net, 1.0, np.full(net.n_legs, CAPACITY), HORIZON)
    assert np.allclose(quiet_bids, 0)
    assert busy_bids[np.argmax(net.leg_load())] > 0


def test_best_fares_uses_displacement_cost():
    net = toy_network(peak_demand=90)
    free = best_fares(net, np.zeros(net.n_od))
    assert (free == 0).all()                       # elastic demand, no seat cost: lowest fare earns most
    blocked = best_fares(net, net.base_fare * FARE_LEVELS.max() + 1)
    assert (blocked == CLOSED).all()               # no fare covers the seats it would use
    mid = best_fares(net, 0.9 * net.base_fare)
    assert (mid > 0).all() and (mid != CLOSED).all()


def test_leg_seat_values_fall_as_seats_rise():
    bids = leg_bid_prices(toy_network(peak_demand=150), 1.0)
    seat_values = bids[:, :, 1:]
    assert (seat_values >= -1e-9).all()
    assert (np.diff(seat_values, axis=-1) <= 1e-6).all()
    assert (seat_values[HORIZON] >= seat_values[0] - 1e-6).all()      # more time left, each seat worth more


def test_lp_bound_is_above_every_policy():
    net = toy_network(peak_demand=130)
    bound = lp_upper_bound(net, 1.0)
    for policy in [NetworkFixedFare(), BidPriceDecomposition(), BidPriceLP()]:
        revenue, _, _ = simulate_network(policy, net, 1.0, 1.0, n_runs=2_000, seed=4)
        assert revenue.mean() <= bound


def test_bid_prices_beat_single_segment_pricing_when_seats_are_scarce():
    net = toy_network(peak_demand=130)
    rng = np.random.default_rng(5)
    customers = (rng.random((2_000, HORIZON + 1, net.n_od)),
                 rng.permuted(np.tile(np.arange(net.n_od), (HORIZON + 1, 1)), axis=1))
    revenue = {p.name: simulate_network(p, net, 1.0, 1.0, customers=customers)[0].mean()
               for p in [NetworkFixedFare(), SegmentOptimizer(), SegmentOptimizer(quotas=True),
                         BidPriceDecomposition()]}
    network = revenue["Network bid prices (DP by leg)"]
    assert network > max(v for k, v in revenue.items() if k != "Network bid prices (DP by leg)")


def test_arrivals_are_never_negative():
    net = toy_network(peak_demand=90)
    customers = (np.zeros((3, HORIZON + 1, net.n_od)), np.tile(np.arange(net.n_od), (HORIZON + 1, 1)))
    revenue, sold, _ = simulate_network(NetworkFixedFare(), net, 1.0, 1.0, customers=customers)
    assert (sold == 0).all() and (revenue == 0).all()


def test_failed_lp_raises():
    net = toy_network(peak_demand=90)
    with pytest.raises(RuntimeError, match="network LP failed"):
        network_lp(net, 1.0, np.full(net.n_legs, -1), HORIZON)      # negative seats: infeasible


def test_planning_with_a_different_model():
    net, wrong = toy_network(peak_demand=130), toy_network(peak_demand=130, elasticity=0.8)
    customers = random_customers(200, net.n_od, 6)
    for policy in [SegmentOptimizer(quotas=True), BidPriceDecomposition()]:
        right = simulate_network(policy, net, 1.0, 1.0, customers=customers)[0]
        same = simulate_network(PlansWith(policy, toy_network(peak_demand=130)), net, 1.0, 1.0, customers=customers)[0]
        assert np.array_equal(right, same)                 # planning with an identical model changes nothing
        misled = simulate_network(PlansWith(policy, wrong), net, 1.0, 1.0, customers=customers)[0]
        assert misled.mean() < right.mean()                # planning with the wrong elasticity costs revenue
