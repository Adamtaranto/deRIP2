"""
Logging utilities for deRIP2.

This module provides tools for colorized terminal output and configurable logging
including colored log levels and custom text formatting.
"""

import logging
import sys
from typing import Callable, Dict, Optional


# Color formatting class with method-based color access
class ColoredText:
    """
    Utility for applying ANSI color codes to text for terminal output.

    This class provides multiple ways to colorize text:
    - As a function: colored('text', 'green')
    - As methods: colored.green('text'), colored.red('text'), etc.

    Attributes
    ----------
    _colors : Dict[str, str]
        Dictionary mapping color names to ANSI color codes.
    """

    def __init__(self) -> None:
        """Initialize the ColoredText instance with color mappings."""
        # Dictionary mapping color names to ANSI codes
        self._colors: Dict[str, str] = {
            'red': '\033[91m',
            'green': '\033[92m',
            'blue': '\033[94m',
            'yellow': '\033[93m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'bold': '\033[1m',
            'reset': '\033[0m',
        }

        # Create color methods dynamically
        for color_name in self._colors:
            # Skip 'reset' as it's not intended to be a color method
            if color_name != 'reset':
                setattr(self, color_name, self._create_color_method(color_name))

    def _create_color_method(self, color_name: str) -> Callable[[str], str]:
        """
        Create a method that applies a specific color to text.

        Parameters
        ----------
        color_name : str
            Name of the color to apply.

        Returns
        -------
        Callable
            Function that applies the color to provided text.
        """

        def color_method(text: str) -> str:
            """
            Apply a specific color to the given text.

            Parameters
            ----------
            text : str
                The text to colorize.

            Returns
            -------
            str
                Text with ANSI color codes applied.
            """
            return f'{self._colors[color_name]}{text}{self._colors["reset"]}'

        return color_method

    def __call__(self, text: str, color: str) -> str:
        """
        Apply color to text string for terminal output.

        Parameters
        ----------
        text : str
            The text to colorize.
        color : str
            Color name ('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'bold').

        Returns
        -------
        str
            Text with ANSI color codes.
        """
        color_code = self._colors.get(color, '')
        if not color_code:
            return text
        return f'{color_code}{text}{self._colors["reset"]}'


# Create a singleton instance
colored = ColoredText()


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to add color to log messages based on their severity level.

    This formatter applies different ANSI color codes to log messages depending on
    their severity, making it easier to distinguish between different log levels
    in terminal output.

    Parameters
    ----------
    fmt : str
        The format string to use for log messages.

    Attributes
    ----------
    grey : str
        ANSI color code for grey text (used for DEBUG).
    blue : str
        ANSI color code for blue text (used for INFO).
    yellow : str
        ANSI color code for yellow text (used for WARNING).
    red : str
        ANSI color code for red text (used for ERROR).
    bold_red : str
        ANSI color code for bold red text (used for CRITICAL).
    reset : str
        ANSI code to reset text formatting.
    fmt : str
        Format string for log messages.
    FORMATS : dict
        Mapping of log levels to their formatted strings with color codes.
    """

    # ANSI escape codes for colors
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt: str) -> None:
        """
        Initialize the CustomFormatter with a specified format string.

        Parameters
        ----------
        fmt : str
            The format string for log messages.

        Returns
        -------
        None
        """
        super().__init__()
        self.fmt = fmt

        # Map each log level to a colored format string
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with the appropriate color based on its severity level.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log message with appropriate color codes.
        """
        # Get the format string for this record's log level
        log_fmt = self.FORMATS.get(record.levelno)

        # Create a formatter with the color-coded format string
        formatter = logging.Formatter(log_fmt)

        # Apply the formatter to the record
        return formatter.format(record)


def init_logging(loglevel: str = 'DEBUG', logfile: Optional[str] = None) -> None:
    """
    Initialize the logging system with a specified log level and custom formatter.

    This function configures the Python logging system with colored console output
    and optional file output. It sets the global logging configuration that will
    be used throughout the application.

    Parameters
    ----------
    loglevel : str, optional
        The log level to use (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        Default is "DEBUG".
    logfile : str, optional
        The file to which log messages should be written.
        If None, log messages will only be output to stderr.
        Default is None.

    Returns
    -------
    None
        This function configures the global logging system but doesn't return a value.

    Raises
    ------
    ValueError
        If the provided log level string is not a valid logging level.
    """
    # Convert log level string to numeric value
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {loglevel}')

    # Define log message format including timestamp, level, module, function, line number
    fmt = '%(asctime)s | %(levelname)s | %(module)s | %(funcName)s | %(lineno)d | %(message)s'

    # Create a StreamHandler to output colored log messages to stderr
    handler_sh = logging.StreamHandler(sys.stderr)
    handler_sh.setFormatter(CustomFormatter(fmt))

    # Set up handlers for the root logger
    handlers = [handler_sh]

    # If a log file was specified, add a file handler
    if logfile is not None:
        # Create a FileHandler with plain (non-colored) formatter
        handler_fh = logging.FileHandler(logfile)
        handler_fh.setFormatter(logging.Formatter(fmt))
        handlers.append(handler_fh)

    # Configure the root logger with the specified level and handlers
    logging.basicConfig(
        format=fmt,  # This format is used for non-custom formatters
        level=numeric_level,
        handlers=handlers,
    )
