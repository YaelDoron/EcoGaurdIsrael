"""Application configuration, loaded from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

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
DEFAULT_COPERNICUS_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
DEFAULT_COPERNICUS_STATISTICS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"
DEFAULT_COPERNICUS_LAND_COVER_COLLECTION_ID = "35fecfec-8a73-4723-bb08-b775f283a535"
DEFAULT_COPERNICUS_REQUEST_TIMEOUT = 20
DEFAULT_FRONTEND_ORIGINS = "http://localhost:5173"

# Task 5: runtime ML integration for Fire Detection. Defaults are
# intentionally conservative - SHADOW mode records ML output but never lets
# it change what FireEvent gets created/updated; RULE_ONLY reproduces
# pre-Task-5 behavior exactly. See backend/docs/fire_detection_runtime_ml.md.
DEFAULT_FIRE_DETECTION_DECISION_MODE = "shadow"
DEFAULT_FIRE_DETECTION_ML_MODEL_PATH = str(_BACKEND_ROOT / "models" / "fire_detection" / "fire_detection_logistic_v3.joblib")
DEFAULT_FIRE_DETECTION_ML_METADATA_PATH = str(
    _BACKEND_ROOT / "models" / "fire_detection" / "fire_detection_logistic_v3_metadata.json"
)
DEFAULT_FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD = 0.50
# Selected via scripts.analyze_fire_detection_ml_threshold (out-of-fold CV on
# training_v3.csv): 0.70 is the smallest candidate threshold reaching
# precision >= 0.90 for the specific NO_EVENT -> SUSPECTED HYBRID escalation
# path. Empty string means "no threshold configured" -> escalation disabled.
DEFAULT_FIRE_DETECTION_ML_SUSPECT_THRESHOLD = "0.70"


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

    COPERNICUS_CLIENT_ID: str = os.getenv("COPERNICUS_CLIENT_ID", "")
    COPERNICUS_CLIENT_SECRET: str = os.getenv("COPERNICUS_CLIENT_SECRET", "")
    COPERNICUS_TOKEN_URL: str = os.getenv("COPERNICUS_TOKEN_URL", DEFAULT_COPERNICUS_TOKEN_URL)
    COPERNICUS_STATISTICS_URL: str = os.getenv(
        "COPERNICUS_STATISTICS_URL",
        DEFAULT_COPERNICUS_STATISTICS_URL,
    )
    COPERNICUS_LAND_COVER_COLLECTION_ID: str = os.getenv(
        "COPERNICUS_LAND_COVER_COLLECTION_ID",
        DEFAULT_COPERNICUS_LAND_COVER_COLLECTION_ID,
    )
    COPERNICUS_REQUEST_TIMEOUT: int = int(
        os.getenv("COPERNICUS_REQUEST_TIMEOUT", str(DEFAULT_COPERNICUS_REQUEST_TIMEOUT))
    )

    # Raw PostgreSQL/Neon connection string, unmodified. May be empty in
    # environments that don't use the database (e.g. Task 1/2 unit tests).
    # URL normalization (postgresql:// -> postgresql+psycopg://) happens in
    # src.database.connection, not here.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Comma-separated list of origins allowed to call the API via CORS
    # (React/Vite dev server by default). Parsed once here so callers get a
    # ready-to-use tuple instead of re-splitting a raw string.
    FRONTEND_ORIGINS: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", DEFAULT_FRONTEND_ORIGINS).split(",")
        if origin.strip()
    )

    # Task A1.6: explicit, off-by-default safety gate for DemoStateResetService
    # (backend/src/simulation/demo_state_reset_service.py). This project has
    # no general APP_ENV/environment concept yet, so this flag is the sole
    # gate: reset() refuses to run at all unless the operator has
    # deliberately set this to true in the environment pointed at by
    # DATABASE_URL (intended to be a dedicated demo database/Neon branch,
    # never the shared team development database). Never inferred from the
    # database name/hostname - that would be a heuristic, not a guarantee.
    ENABLE_DEMO_DATA_RESET: bool = os.getenv("ENABLE_DEMO_DATA_RESET", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    # Task A3: explicit, off-by-default gate for the Simulation Control API
    # (POST/GET /api/v1/simulation/...). Deliberately separate from
    # ENABLE_DEMO_DATA_RESET above - "may invoke simulation-control
    # endpoints at all" and "may destructively clear demo runtime state" are
    # two independent permissions a deployment may want to grant separately
    # (e.g. simulation control enabled without reset permission). Never
    # implied by the other flag.
    ENABLE_SIMULATION_CONTROL_API: bool = os.getenv("ENABLE_SIMULATION_CONTROL_API", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    # Task 5: runtime ML integration for Fire Detection.
    FIRE_DETECTION_DECISION_MODE: str = os.getenv("FIRE_DETECTION_DECISION_MODE", DEFAULT_FIRE_DETECTION_DECISION_MODE)
    FIRE_DETECTION_ML_MODEL_PATH: str = os.getenv("FIRE_DETECTION_ML_MODEL_PATH", DEFAULT_FIRE_DETECTION_ML_MODEL_PATH)
    FIRE_DETECTION_ML_METADATA_PATH: str = os.getenv(
        "FIRE_DETECTION_ML_METADATA_PATH", DEFAULT_FIRE_DETECTION_ML_METADATA_PATH
    )
    FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD: float = float(
        os.getenv("FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD", str(DEFAULT_FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD))
    )
    # Empty string means "not configured" -> HYBRID NO_EVENT->SUSPECTED ML
    # escalation is disabled rather than guessing a threshold. Parsed to
    # float | None by callers (see src.calculators.fire_detection.
    # fire_detection_decision_policy), not here, so an invalid/blank value
    # stays a simple, inspectable string at the settings layer.
    FIRE_DETECTION_ML_SUSPECT_THRESHOLD: str = os.getenv(
        "FIRE_DETECTION_ML_SUSPECT_THRESHOLD", DEFAULT_FIRE_DETECTION_ML_SUSPECT_THRESHOLD
    )


settings = Settings()
