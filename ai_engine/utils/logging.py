import logging
import sys

def setup_logger(name: str = "ai_engine") -> logging.Logger:
    """
    Configures and returns a logger instance with formatted console output.
    Uses colorlog if available, falling back to standard logging format.
    """
    logger = logging.getLogger(name)
    
    # If handler is already configured, don't duplicate handlers
    if logger.handlers:
        return logger

    import os
    # Read environment variable directly to bypass circular module imports at startup
    debug_mode = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "yes")
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)

    # Standard formatter fallback
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    try:
        import colorlog
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=date_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            }
        )
    except ImportError:
        formatter = logging.Formatter(format_str, datefmt=date_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Global default logger
logger = setup_logger()
