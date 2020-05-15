from . import BaseMetric
import numpy as np

class TravelTimeMetric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """
    def __init__(self, world):
        self.world = world

    def update(self, done=False):
        return self.world.eng.get_average_travel_time()
        
