"""Checks that the simulator and the optimizer agree with each other and with basic probability."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pricing_sim import (BASE_IDX, CAPACITY, FARE_LEVELS, HORIZON, BookingState, FixedFare, ForecastOptimizer, PacingRule,
                         Market, PerfectInformation, QLearningAgent, booking_curve_from_bookings, optimal_fares,
                         simulate)


@pytest.fixture
def market():
    # Late-skewed booking curve similar to the real one
    curve = np.exp(-np.arange(HORIZON + 1) / 6.0)
    return Market(curve / curve.sum(), elasticity=1.2, demand_at_base=85)


def test_booking_curve_from_bookings_sums_to_one():
    bookings = pd.DataFrame({"journey_date": ["2025-03-10"] * 3, "booking_date": ["2025-03-10", "2025-03-09", "2024-12-01"],
                             "PSGN": [2, 1, 1]})
    curve = booking_curve_from_bookings(bookings)
    assert curve.shape == (HORIZON + 1,)
    assert curve.sum() == pytest.approx(1.0)
    assert curve[0] == pytest.approx(0.5) and curve[1] == pytest.approx(0.25) and curve[HORIZON] == pytest.approx(0.25)


def test_fixed_fare_matches_poisson_maths(market):
    # Daily Poisson bookings capped by seats left sum to min(Poisson(total), capacity)
    total = market.mean_arrivals(1.0, np.arange(HORIZON + 1), BASE_IDX).sum()
    k = np.arange(CAPACITY + 1)
    expected_sold = (np.minimum(k, CAPACITY) * poisson.pmf(k, total)).sum() + CAPACITY * poisson.sf(CAPACITY, total)
    _, sold, _, _ = simulate(FixedFare(), market, 1.0, 1.0, n_runs=20_000, seed=1)
    assert sold.mean() == pytest.approx(expected_sold, rel=0.01)


def test_optimizer_value_matches_simulation(market):
    _, value = optimal_fares(market, 1.0)
    revenue, _, _, _ = simulate(PerfectInformation(1.0), market, 1.0, 1.0, n_runs=20_000, seed=2)
    assert revenue.mean() == pytest.approx(value[CAPACITY], rel=0.01)


def test_optimizer_never_worse_than_fixed_fare(market):
    for level in [0.5, 1.0, 1.4]:
        _, value = optimal_fares(market, level)
        fixed, _, _, _ = simulate(FixedFare(), market, level, level, n_runs=20_000, seed=3)
        assert value[CAPACITY] >= fixed.mean() * 0.995


def test_fare_cap_is_respected(market):
    capped = ForecastOptimizer(max_fare=1.10)
    _, _, _, fares = simulate(capped, market, 1.5, 1.5, n_runs=200, seed=4)
    assert fares.max() <= 1.10 * 1100.0 + 1e-6


def test_same_random_customers_give_same_result(market):
    u = np.random.default_rng(5).random((50, HORIZON + 1))
    a = simulate(ForecastOptimizer(), market, 1.1, 1.0, uniforms=u)
    b = simulate(ForecastOptimizer(), market, 1.1, 1.0, uniforms=u)
    assert np.array_equal(a[0], b[0])


def test_reforecast_starts_at_forecast_and_learns(market):
    confident = ForecastOptimizer(reforecast=True, forecast_cv=0.1)
    unsure = ForecastOptimizer(reforecast=True, forecast_cv=0.3)
    start = BookingState(HORIZON, np.array([CAPACITY]), np.array([0]), np.array([0.0]), 0.9)
    assert confident.estimate(start, market)[0] == pytest.approx(0.9)
    # Bookings running at twice the forecast pull the estimate up, more so when the forecast is uncertain.
    # Exposure is built the way the simulator builds it: expected bookings so far at demand level 1.0.
    exposure = market.mean_arrivals(1.0, np.arange(11, HORIZON + 1), BASE_IDX).sum()
    busy = BookingState(10, np.array([40]), np.array([round(2 * 0.9 * exposure)]), np.array([exposure]), 0.9)
    assert 0.9 < confident.estimate(busy, market)[0] < unsure.estimate(busy, market)[0]
    assert unsure.estimate(busy, market)[0] > 1.2


def test_reforecasting_recovers_from_a_wrong_forecast(market):
    # True demand is 40% above the forecast: updating from bookings should earn more than trusting the forecast
    u = np.random.default_rng(7).random((3_000, HORIZON + 1))
    static, _, _, _ = simulate(ForecastOptimizer(), market, 1.4, 1.0, uniforms=u)
    updating, _, _, _ = simulate(ForecastOptimizer(reforecast=True, forecast_cv=0.15), market, 1.4, 1.0, uniforms=u)
    assert updating.mean() > static.mean()
    # And when the forecast is right, updating costs (almost) nothing
    static, _, _, _ = simulate(ForecastOptimizer(), market, 1.0, 1.0, uniforms=u)
    updating, _, _, _ = simulate(ForecastOptimizer(reforecast=True, forecast_cv=0.15), market, 1.0, 1.0, uniforms=u)
    assert updating.mean() > 0.99 * static.mean()


def test_pacing_plan_matches_expected_bookings(market):
    # On a departure with exactly forecast demand, the pacing rule should mostly hold the starting fare
    _, _, changes, _ = simulate(PacingRule(), market, 1.0, 1.0, n_runs=2_000, seed=8)
    assert np.median(changes) <= 10


def test_valuing_passengers_carries_more_people(market):
    level = 0.8
    u = np.random.default_rng(6).random((5_000, HORIZON + 1))
    _, sold_rev, _, _ = simulate(PerfectInformation(level), market, level, level, uniforms=u)
    _, sold_soc, _, _ = simulate(ForecastOptimizer(seat_value=500.0), market, level, level, uniforms=u)
    assert sold_soc.mean() > sold_rev.mean()


def test_q_learning_averages_updates_that_share_a_state():
    # All demand books on the first day, so every episode in the batch starts in the same state and
    # its first update's target is just that day's revenue. Duplicates must be averaged, not stacked:
    # no Q value can exceed the most a booking window can earn.
    curve = np.zeros(HORIZON + 1)
    curve[HORIZON] = 1.0
    agent = QLearningAgent(seed=0).train(Market(curve, demand_at_base=85), [1.0], [1.0], episodes=300, batch=300)
    assert agent.q.max() <= CAPACITY * FARE_LEVELS.max()
    assert agent.q.max() > 0


def test_arrivals_are_never_negative(market):
    # poisson.ppf(0, mu) is -1: a uniform of exactly 0 must mean no bookings, not a returned seat
    _, sold, _, _ = simulate(FixedFare(), market, 1.0, 1.0, uniforms=np.zeros((3, HORIZON + 1)))
    assert (sold == 0).all()
