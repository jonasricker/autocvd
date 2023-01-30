"""Various utility functions."""

import argparse
import sys
from typing import Any


def print_info(msg: str, overwrite: bool = False) -> None:
    """Print message to stderr."""
    if overwrite:
        end = "\r"
    else:
        end = None
    print(f"autocvd: {msg}", file=sys.stderr, end=end)


class Spinner:
    """Simple progress spinner."""

    chars = "/-\\|"

    def __init__(self) -> None:
        """Create new Spinner instance."""
        self.state = 0

    def __repr__(self) -> str:
        """Return current character and increment state."""
        out = self.chars[self.state]
        self.state = (self.state + 1) % len(self.chars)
        return out


def positive_int(value: Any) -> int:
    """Convert value to positive integer."""
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive int value: '{value}'")
    return integer
