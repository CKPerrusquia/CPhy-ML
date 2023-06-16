from stonesoup.types.update import GaussianStateUpdate
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.track import Track
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
import numpy as np

class Initiator(SimpleMeasurementInitiator):
    """ Basic Initiator for simulating radar measurements """
    def initiate(self, detections, timestamp, **kwargs):
        MAX_DEV = 400.
        tracks = set()
        measurement_model = self.measurement_model
        for detection in detections:
            state_vector = measurement_model.inverse_function(detection)
            model_covar = measurement_model.covar()

            el_az_range = np.sqrt(np.diag(model_covar)) #elev, az, range
            
            std_pos = detection.state_vector[2, 0]*el_az_range[1]
            stdx = np.abs(std_pos*np.sin(el_az_range[1]))
            stdy = np.abs(std_pos*np.cos(el_az_range[1]))
            stdz = np.abs(detection.state_vector[2, 0]*el_az_range[0])
            if stdx > MAX_DEV:
                print('Warning - X Deviation exceeds limit!!')
            if stdy > MAX_DEV:
                print('Warning - Y Deviation exceeds limit!!')
            if stdz > MAX_DEV:
                print('Warning - Z Deviation exceeds limit!!')
            C0 = np.diag(np.array([stdx, 30.0, stdy, 30.0, stdz, 30.0])**2)

            # add track to our total tracks
            tracks.add(Track([GaussianStateUpdate(state_vector, C0,
                                                  SingleHypothesis(None, detection),
                                                  timestamp=detection.timestamp)]))
        return tracks