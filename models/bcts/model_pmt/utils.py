from enum import Enum

class DiseaseState(Enum):
    """An enum for different disease states."""
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTED    = 2
    RECOVERED   = 3
