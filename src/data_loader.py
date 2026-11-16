"""
Loads the booking records used by the notebooks.

The original analysis used confidential Indian Railways booking records for train 12138
(one row per ticket). They are not part of this repository. If you have them, place them at
data/private/bookings_12138.pkl (the folder is git-ignored) with the columns

    booking_date, journey_date, trnno, brdpt_code, resupto_code, cls, PSGN

Without that file, the loader falls back to synthetic bookings from synthetic_data.py, so
every notebook still runs; its numbers will then differ from the saved outputs.
"""

from pathlib import Path

import pandas as pd

PRIVATE_BOOKINGS = Path(__file__).resolve().parents[1] / "data" / "private" / "bookings_12138.pkl"


def load_bookings(cls=None):
    """Booking records, optionally for one class (e.g. "3A"), with dates parsed."""
    if PRIVATE_BOOKINGS.exists():
        bookings = pd.read_pickle(PRIVATE_BOOKINGS)
    else:
        from synthetic_data import make_bookings
        print("data/private/bookings_12138.pkl not found: using synthetic bookings instead.")
        bookings = make_bookings()
    bookings = bookings.assign(journey_date=pd.to_datetime(bookings["journey_date"]),
                               booking_date=pd.to_datetime(bookings["booking_date"]))
    return bookings[bookings["cls"] == cls] if cls else bookings


def daily_totals(bookings, origin=None, destination=None):
    """Total passengers per departure (optionally for one origin -> destination)."""
    rows = bookings
    if origin:
        rows = rows[rows["brdpt_code"] == origin]
    if destination:
        rows = rows[rows["resupto_code"] == destination]
    daily = rows.groupby("journey_date", as_index=False)["PSGN"].sum().rename(columns={"PSGN": "total_psgn"})
    daily["journey_date"] = pd.to_datetime(daily["journey_date"])
    return daily
