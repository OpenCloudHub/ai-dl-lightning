# ==============================================================================
# Serving Configuration
# ==============================================================================
#
# Configuration settings for the serving application.
#
# Settings:
#   - REQUEST_MAX_LENGTH: Maximum batch size for predictions (prevent DOS)
#
# Usage:
#   from src.serving.config import SERVING_CONFIG
#   max_batch = SERVING_CONFIG.request_max_length
#
# ==============================================================================

from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    request_max_length: int = 1000


SERVING_CONFIG = ServingConfig()
