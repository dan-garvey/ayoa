"""Random number generator with seed control."""

import random

from core.config import engine_config


class GameRNG:
    """Seeded random number generator for deterministic behavior."""

    def __init__(self, seed: int = None):
        """
        Initialize RNG.

        Args:
            seed: Random seed (uses config default if not provided)
        """
        self.seed = seed if seed is not None else engine_config.rng_seed
        self.rng = random.Random(self.seed)

    def choice(self, items: list):
        """Random choice from list."""
        return self.rng.choice(items)

    def shuffle(self, items: list) -> list:
        """Shuffle list (returns new list)."""
        shuffled = items.copy()
        self.rng.shuffle(shuffled)
        return shuffled

    def randint(self, a: int, b: int) -> int:
        """Random integer in range [a, b]."""
        return self.rng.randint(a, b)


# Global RNG instance
game_rng = GameRNG()
