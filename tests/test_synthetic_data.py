"""Checks that the synthetic bookings have the structure and patterns the notebooks rely on."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synthetic_data import booking_curve, daily_totals, make_bookings, od_mix, route


@pytest.fixture(scope="module")
def bookings():
    return make_bookings(seed=0)


def test_columns_and_dates(bookings):
    assert list(bookings.columns) == ["booking_date", "journey_date", "trnno", "brdpt_code", "resupto_code",
                                      "cls", "PSGN"]
    assert bookings.journey_date.min() == pd.Timestamp("2023-08-01")
    assert bookings.journey_date.max() == pd.Timestamp("2025-07-31")
    assert (bookings.booking_date <= bookings.journey_date).all()
    assert bookings.PSGN.between(1, 6).all()


def test_same_seed_same_data():
    a, b = make_bookings(seed=3), make_bookings(seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_journeys_follow_the_route(bookings):
    order = {c: i for i, c in enumerate(route()["code"])}
    assert (bookings.brdpt_code.map(order) < bookings.resupto_code.map(order)).all()
    assert od_mix()["share"].sum() == pytest.approx(1.0)


def test_booking_window_shrinks_to_60_days(bookings):
    days_out = (bookings.journey_date - bookings.booking_date).dt.days
    assert days_out[bookings.journey_date < "2024-08-01"].max() > 60
    assert days_out[bookings.journey_date >= "2025-03-01"].max() <= 60


def test_booking_curve_sums_to_one_with_tatkal_spike():
    for window in (60, 120):
        curve = booking_curve(window)
        assert curve.sum() == pytest.approx(1.0)
        assert curve[1] == curve.max()


def test_weekly_and_yearly_patterns(bookings):
    daily = daily_totals(bookings).set_index("journey_date")["total_psgn"]
    by_weekday = daily.groupby(daily.index.day_of_week).mean()
    assert by_weekday.idxmax() == 5                           # Saturday busiest
    assert daily.corr(daily.shift(364)) > 0.2                 # last year carries information
