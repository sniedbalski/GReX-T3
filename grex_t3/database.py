import sqlite3
import logging


def connect(path: str) -> sqlite3.Connection:
    """Connect to the SQLite database"""
    con = sqlite3.connect(path)
    logging.info("Successfully connected to SQLite database")
    return con


def is_injection(mjd: float, con: sqlite3.Connection) -> bool:
    """Run a SQL query to see if T0 performed an injection near candidate time"""

    # Not sure why the offset from T0 is this much
    OFFSET = 15 / 86400  # Seconds to days
    cur = con.cursor()
    logging.debug(f"Testing if candidate at {mjd} corresponds to an injection")
    cur.execute(
        "SELECT COUNT(*) FROM injection WHERE mjd BETWEEN ? AND ?",
        (
            mjd - OFFSET / 2,
            mjd + OFFSET / 2,
        ),
    )
    res = cur.fetchone()
    logging.debug(f"SQL Query Result: {res}")
    return res[0] == 1
