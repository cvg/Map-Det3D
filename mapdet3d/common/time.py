"""This module contains utilities for tracking execution time."""

from __future__ import annotations

from time import perf_counter
from typing import no_type_check


@no_type_check
def timeit(func):
    """Function to be used as decorator to time a function."""

    def timed(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        toc = perf_counter()
        print(f"{func.__name__}  {(toc - tic) * 1000:.2f} ms")
        return result

    return timed


class Timer:  # pragma: no cover
    """Timer class based on perf_counter."""

    def __init__(self) -> None:
        """Creates an instance of the class."""
        self._tic = perf_counter()
        self._toc: None | float = None
        self.paused = False

    def reset(self) -> None:
        """Reset timer."""
        self._tic = perf_counter()
        self._toc = None
        self.paused = False

    def pause(self) -> None:
        """Pause function."""
        if self.paused:
            raise ValueError("Timer already paused!")
        self._toc = perf_counter()
        self.paused = True

    def resume(self) -> None:
        """Resume function."""
        if not self.paused:
            raise ValueError("Timer is not paused!")
        assert self._toc is not None
        self._tic = perf_counter() - (self._toc - self._tic)
        self._toc = None
        self.paused = False

    def time(self, milliseconds: bool = False) -> float:
        """Return elapsed time."""
        if not self.paused:
            self._toc = perf_counter()
        assert self._toc is not None
        time_elapsed = self._toc - self._tic
        if milliseconds:
            return time_elapsed * 1000
        return time_elapsed
