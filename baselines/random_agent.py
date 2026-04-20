import numpy as np


class RandomLegalAgent:
    """Selects a random legal action using the provided mask."""

    def act(self, observation: dict) -> int:
        mask = observation["action_mask"]
        legal = np.nonzero(mask)[0]
        return int(np.random.choice(legal))
