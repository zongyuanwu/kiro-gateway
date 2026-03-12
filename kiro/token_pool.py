"""Token pool for distributing requests across multiple Kiro credentials."""

import asyncio
import itertools
import random
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from loguru import logger

from kiro.auth import KiroAuthManager


class TokenPool:
    """
    Manages multiple KiroAuthManager instances and distributes requests.

    Strategies:
    - round_robin: cycle through tokens sequentially (default)
    - least_used: pick the token with fewest in-flight requests
    - random: pick a random token
    """

    def __init__(self, managers: List[KiroAuthManager], strategy: str = "round_robin"):
        if not managers:
            raise ValueError("TokenPool requires at least one KiroAuthManager")

        self._managers = managers
        self._strategy = strategy
        self._cycle = itertools.cycle(range(len(managers)))
        self._lock = asyncio.Lock()
        self._in_flight = [0] * len(managers)

        logger.info(f"Token pool initialized: {len(managers)} token(s), strategy={strategy}")

    @property
    def managers(self) -> List[KiroAuthManager]:
        return self._managers

    @property
    def size(self) -> int:
        return len(self._managers)

    async def get_auth_manager(self) -> KiroAuthManager:
        """Pick the next auth manager based on the configured strategy."""
        async with self._lock:
            if self._strategy == "least_used":
                idx = self._in_flight.index(min(self._in_flight))
            elif self._strategy == "random":
                idx = random.randrange(len(self._managers))
            else:  # round_robin
                idx = next(self._cycle)

            self._in_flight[idx] += 1
            logger.debug(f"Token pool: selected token #{idx} (strategy={self._strategy}, in_flight={self._in_flight})")
            return self._managers[idx]

    async def release(self, manager: KiroAuthManager) -> None:
        """Decrement in-flight count when a request completes."""
        async with self._lock:
            try:
                idx = self._managers.index(manager)
                self._in_flight[idx] = max(0, self._in_flight[idx] - 1)
            except ValueError:
                logger.warning("Attempted to release an unknown auth manager")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[KiroAuthManager]:
        """Context manager that acquires and automatically releases an auth manager."""
        manager = await self.get_auth_manager()
        try:
            yield manager
        finally:
            await self.release(manager)
