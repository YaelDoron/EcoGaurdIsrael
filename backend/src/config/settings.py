"""Application configuration, loaded from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_IMS_BASE_URL = "https://api.ims.gov.il/v1/envista"
DEFAULT_IMS_REQUEST_TIMEOUT = 10
DEFAULT_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"
DEFAULT_FIRMS_SOURCE = "VIIRS_NOAA20_NRT"
DEFAULT_FIRMS_DAY_RANGE = 1
DEFAULT_FIRMS_REQUEST_TIMEOUT = 10
DEFAULT_FIRMS_WEST = 34.0
DEFAULT_FIRMS_SOUTH = 29.4
DEFAULT_FIRMS_EAST = 35.9
DEFAULT_FIRMS_NORTH = 33.4


@dataclass(frozen=True)
class Settings:
    """Application configuration values.

    IMS_API_TOKEN may be empty (we are currently waiting for the real IMS API
    token); IMSClient handles that case explicitly when a request is attempted.
    """

    IMS_BASE_URL: str = os.getenv("IMS_BASE_URL", DEFAULT_IMS_BASE_URL)
    IMS_API_TOKEN: str = os.getenv("IMS_API_TOKEN", "")
    IMS_REQUEST_TIMEOUT: int = int(os.getenv("IMS_REQUEST_TIMEOUT", str(DEFAULT_IMS_REQUEST_TIMEOUT)))

    FIRMS_BASE_URL: str = os.getenv("FIRMS_BASE_URL", DEFAULT_FIRMS_BASE_URL)
    FIRMS_MAP_KEY: str = os.getenv("FIRMS_MAP_KEY", "")
    FIRMS_SOURCE: str = os.getenv("FIRMS_SOURCE", DEFAULT_FIRMS_SOURCE)
    FIRMS_DAY_RANGE: int = int(os.getenv("FIRMS_DAY_RANGE", str(DEFAULT_FIRMS_DAY_RANGE)))
    FIRMS_REQUEST_TIMEOUT: int = int(
        os.getenv("FIRMS_REQUEST_TIMEOUT", str(DEFAULT_FIRMS_REQUEST_TIMEOUT))
    )
    FIRMS_WEST: float = float(os.getenv("FIRMS_WEST", str(DEFAULT_FIRMS_WEST)))
    FIRMS_SOUTH: float = float(os.getenv("FIRMS_SOUTH", str(DEFAULT_FIRMS_SOUTH)))
    FIRMS_EAST: float = float(os.getenv("FIRMS_EAST", str(DEFAULT_FIRMS_EAST)))
    FIRMS_NORTH: float = float(os.getenv("FIRMS_NORTH", str(DEFAULT_FIRMS_NORTH)))

    # Raw PostgreSQL/Neon connection string, unmodified. May be empty in
    # environments that don't use the database (e.g. Task 1/2 unit tests).
    # URL normalization (postgresql:// -> postgresql+psycopg://) happens in
    # src.database.connection, not here.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")


settings = Settings()
