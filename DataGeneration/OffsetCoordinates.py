from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.feeder.base import DetectionFeeder, GroundTruthFeeder
from stonesoup.reader.generic import CSVGroundTruthReader
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.simulator.platform import PlatformDetectionSimulator
from stonesoup.types.state import GaussianState, State, StateVector
from stonesoup.types.update import GaussianStateUpdate
import numpy as np

from datetime import datetime, timedelta
from functools import partial
from typing import Tuple

class OffsetCoordinates(DetectionFeeder, GroundTruthFeeder):
    """ Custom class that allows manipulation of the cartesian
        co-ordinates in the ground-truth file. This allows the
        flight location to be manipulated with respect to the
        radar, which is located at x,y position (0,0). """
    offsets: Tuple[float, float] = Property(
        doc="(offset_x, offset_y)")
    mapping: Tuple[int, int] = Property(
        default=(0, 2),
        doc="Indexes of x and y in state vector. Default (0, 2)")

    @BufferedGenerator.generator_method
    def data_gen(self):
        # iterate through all states and offset co-ordinates accordingly
        for time, states in self.reader:
            for state in states:
                new_coord = (state.state_vector[self.mapping[0], 0] + self.offsets[0],
                             state.state_vector[self.mapping[1], 0] + self.offsets[1])
                state.state_vector[self.mapping, 0] = new_coord
            yield time, states