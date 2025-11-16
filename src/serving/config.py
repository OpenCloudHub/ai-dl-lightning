from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    request_max_length: int = 1000


SERVING_CONFIG = ServingConfig()
