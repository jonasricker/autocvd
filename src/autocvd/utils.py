"""Various utility functions."""

import argparse
from typing import Any


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
