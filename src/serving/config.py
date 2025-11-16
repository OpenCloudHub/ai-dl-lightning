from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    expected_num_features: int = 12
    request_max_length: int = 1000


SERVING_CONFIG = ServingConfig()
