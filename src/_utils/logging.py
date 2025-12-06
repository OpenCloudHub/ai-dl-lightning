import logging
import sys
import warnings
from os import getenv

from loguru import logger
from rich.console import Console

LOG_LEVEL = getenv("LOG_LEVEL", "INFO").upper()

console = Console(force_terminal=True)

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> | <level>{message}</level>",
    colorize=True,
    filter=lambda record: "name" in record["extra"],  # Only for bound loggers
)
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <dim>{name}</dim> | <level>{message}</level>",
    colorize=True,
    filter=lambda record: "name" not in record["extra"],  # Intercepted stdlib
)


# Intercept stdlib → loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

if LOG_LEVEL != "DEBUG":
    # Silence noisy libs
    for name in ["mlflow", "urllib3", "botocore", "boto3", "fsspec", "git", "ray"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore")


def get_logger(name: str):
    """Get a named logger."""
    return logger.bind(name=name)


def log_section(title: str, emoji: str = "📌"):
    """Visual section separator using Rich."""
    console.rule(f"{emoji} {title}", style="bold cyan")
