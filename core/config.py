
import os
from dotenv import load_dotenv
load_dotenv()
def load_config():
    return {
        "APP_PROJECT_DIR": os.getenv("APP_PROJECT_DIR", "./storage/projects"),
        "APP_LOG_PATH": os.getenv("APP_LOG_PATH", "./storage/logs/app.log"),
        "THRESHOLD_DI": float(os.getenv("THRESHOLD_DI", "0.8")),
        "THRESHOLD_DP_GREEN": float(os.getenv("THRESHOLD_DP_GREEN", "0.1")),
        "THRESHOLD_DP_YELLOW": float(os.getenv("THRESHOLD_DP_YELLOW", "0.2")),
        "THRESHOLD_EO_GREEN": float(os.getenv("THRESHOLD_EO_GREEN", "0.1")),
        "THRESHOLD_EO_YELLOW": float(os.getenv("THRESHOLD_EO_YELLOW", "0.2")),
        "LANG": os.getenv("LANG", "es"),
    }
