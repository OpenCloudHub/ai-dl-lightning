import logging
import sys


def setup_logging(name: str | None = __name__, level=logging.INFO):
    """
    Configure logging for ray workers.
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],  # Force to stdout so Ray captures it
        force=True,  # Reconfigure even if already configured
    )

    ray_data_logger = logging.getLogger("ray.data")
    ray_data_logger.setLevel(logging.WARNING)

    return logging.getLogger(__name__)
