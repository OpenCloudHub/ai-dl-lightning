import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme
CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "success": "green bold",
        "debug": "blue",
    }
)

# Global console instance
console = Console(
    theme=CUSTOM_THEME,
    file=sys.stdout,
    force_terminal=True,
    force_jupyter=False,
    force_interactive=False,
    color_system="truecolor",  # Use full color support
    legacy_windows=False,
)

# Add custom SUCCESS level
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, message, *args, **kwargs):
    """Log a success message."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = success


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with RichHandler for pretty formatting.

    According to Ray docs, you should configure logging AFTER ray.init()
    but BEFORE creating workers. This function returns a logger that
    will work in both driver and worker processes.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            enable_link_path=False,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def log_section(title: str, emoji: str = "📌") -> None:
    """Print a section header - uses print() for compatibility with Ray."""
    console.rule(f"{emoji} {title}", style="bold cyan")
