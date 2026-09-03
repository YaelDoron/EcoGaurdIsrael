"""Application configuration, loaded from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_IMS_BASE_URL = "https://api.ims.gov.il/v1/envista"
DEFAULT_IMS_REQUEST_TIMEOUT = 10


@dataclass(frozen=True)
class Settings:
    """Configuration values used by the IMS integration.

    IMS_API_TOKEN may be empty (we are currently waiting for the real IMS API
    token); IMSClient handles that case explicitly when a request is attempted.
    """

    IMS_BASE_URL: str = os.getenv("IMS_BASE_URL", DEFAULT_IMS_BASE_URL)
    IMS_API_TOKEN: str = os.getenv("IMS_API_TOKEN", "")
    IMS_REQUEST_TIMEOUT: int = int(os.getenv("IMS_REQUEST_TIMEOUT", str(DEFAULT_IMS_REQUEST_TIMEOUT)))

    # Raw PostgreSQL/Neon connection string, unmodified. May be empty in
    # environments that don't use the database (e.g. Task 1/2 unit tests).
    # URL normalization (postgresql:// -> postgresql+psycopg://) happens in
    # src.database.connection, not here.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")


settings = Settings()
