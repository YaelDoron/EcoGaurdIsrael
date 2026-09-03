"""SQLite persistence for wildfire reports, with duplicate prevention on source_url."""
import logging
import sqlite3
from pathlib import Path

from models.fire_report import WildfireReport

logger = logging.getLogger(__name__)

# SQLite schema for the wildfire_reports table. The source_url is unique to prevent duplicates.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wildfire_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT UNIQUE NOT NULL,
    source_feed TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    location_name TEXT,
    latitude REAL,
    longitude REAL,
    published_at TEXT,
    fetched_at TEXT NOT NULL
)
"""

# Initialization of the SQLite database and creation of the wildfire_reports table is handled in the StorageManager constructor.
class StorageManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # Check if a report with the given source_url already exists in the database.
    def exists(self, source_url: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM wildfire_reports WHERE source_url = ?", (source_url,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    # Save a new wildfire report to the database. Returns True if the report was saved, False if it was a duplicate.
    def save_report(self, report: WildfireReport) -> bool:
        """Insert a new report. Returns False (without raising) if it's a duplicate."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO wildfire_reports (
                        source_url, source_feed, title, summary,
                        location_name, latitude, longitude, published_at, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report.source_url,
                        report.source_feed,
                        report.title,
                        report.summary,
                        report.location_name,
                        report.latitude,
                        report.longitude,
                        report.published_at,
                        report.fetched_at,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            logger.info("Skipped duplicate report: %s", report.source_url)
            return False
        finally:
            conn.close()
