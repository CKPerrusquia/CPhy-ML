from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.reader.generic import CSVGroundTruthReader
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.array import CovarianceMatrix
from stonesoup.platform.base import FixedPlatform
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.simulator.platform import PlatformDetectionSimulator
from stonesoup.types.state import GaussianState, State, StateVector
from stonesoup.measures import Euclidean
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.tracker.simple import SingleTargetTracker
from Framework.DataGeneration.Initiator import Initiator
from Framework.DataGeneration.MyDeleter import MyDeleter
from Framework.DataGeneration.OffsetCoordinates import OffsetCoordinates

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import os
import seaborn as sns

from datetime import datetime, timedelta
from functools import partial
from typing import Tuple
from mpl_toolkits import mplot3d

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = True
sns.set_style('white')

class FlightSimulation:
    def simulate_flight(self, flight_filename, data_dir, noise_factor=1.0, velocity_coef=0.5, 
                    offset_x=0, offset_y=0, show_radar=False, figsize=(10,6)):
        
        """ Simulate radar measurements based on ground-truth
        UAV GPS measured trajectories """
        transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(velocity_coef), 
             ConstantVelocity(velocity_coef), 
             ConstantVelocity(velocity_coef/10)])

        # Model coords = elev, bearing, range. Angles in radians
        # this radar measures range with an accuracy of +/- 3.14m, 
        # elevation accuracy +/- sqrt(10) rad  
        # bearing accuracy of +/- 0.6 rad
        meas_covar = np.diag([np.radians(np.sqrt(1.0))**2,
                              np.radians(0.6)**2,
                              3.14**2])

        meas_covar_trk = CovarianceMatrix(1.0*meas_covar)
        meas_model = CartesianToElevationBearingRange(
                        ndim_state=6,
                        mapping=np.array([0, 2, 4]),
                        noise_covar=1.0*meas_covar_trk)
        predictor = ExtendedKalmanPredictor(transition_model)
        updater = ExtendedKalmanUpdater(measurement_model=None)
    
        # read ground truth file according to filename given
        ground_truth_reader = CSVGroundTruthReader(
        path=os.path.join(data_dir, flight_filename),
        state_vector_fields=['x', 'vel_x', 'y', 'vel_y', 'z'],
        time_field='time',
        path_id_field='track_id')
    
        # modify our ground truths by selected offsets
        offsets = [offset_x, offset_y]
        ground_truth_reader = OffsetCoordinates(ground_truth_reader, 
                                                offsets, [0, 2])
    
        # sensor simulated measurements - with desired noise factor
        radar_sensor = RadarElevationBearingRange(
            position_mapping=[0, 2, 4],
            noise_covar=noise_factor*meas_covar,
            ndim_state=6,
            mounting_offset=StateVector([0.0, 0.0, 0.0]))

        # create platform, with sensor at reference point, zero velocity
        platform = FixedPlatform(
            State([0, 0, 0, 0, 0, 0]),
            position_mapping=[0, 2, 4],
            sensors=[radar_sensor])
    
        # Create the detector and initialize it.
        detector = PlatformDetectionSimulator(ground_truth_reader, [platform])
    
        # define prior state for initiator
        prior_state = GaussianState(
            np.array([[0], [0], [0], [0], [0], [0]]),
            np.diag([0, 30.0, 0, 30.0, 0, 30.0])**2)

        # initiate initiator with prior & measurement model
        initiator = Initiator(prior_state, meas_model)
    
        # custom deleter - never delete track (since we know we only have one)
        deleter = MyDeleter()
    
        # we know there is only one measurement per scan, and so use
        # nearest neighbour associator
        meas = Euclidean()
        hypothesiser = DistanceHypothesiser(predictor, updater, meas)
        associator = NearestNeighbour(hypothesiser)

        # create tracker to track measurements and make state estimates
        tracker = SingleTargetTracker(initiator,
                                      deleter,
                                      detector,
                                      associator,
                                      updater)
    
        # create simulated measurements, performing tracking and estimation
        est_X = []
        est_Y = []
        est_Z = []
        meas_X = []
        meas_Y = []
        meas_Z = []
        true_X = []
        true_Y = []
        true_Z = []
        for time, tracks in tracker:
            for ground_truth in ground_truth_reader.groundtruth_paths:
                true_X.append(ground_truth.state_vector[0])
                true_Y.append(ground_truth.state_vector[2])
                true_Z.append(ground_truth.state_vector[4])

            # Because this is a single target tracker, I know there is only 1 track.
            for track in tracks:

                # Get the corresponding measurement
                detection = track.states[-1].hypothesis.measurement
                # Convert measurement into xy (compensate for radar posn offset)
                xyz = meas_model.inverse_function(detection)
                meas_X.append(xyz[0])
                meas_Y.append(xyz[2])
                meas_Z.append(xyz[4])

                vec = track.states[-1].state_vector
                est_X.append(vec[0])
                est_Y.append(vec[2])
                est_Z.append(vec[4])

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1,1,1)
        #ax1 = fig.add_subplot(projection = '3d')
        plt.plot(meas_X, meas_Y, 'xb', label=r'Measurements')
        ax1.plot(true_X, true_Y, 'd-k', label=r'Truth', markerfacecolor='None')
        #ax1.plot(meas_X, meas_Y, 'b', alpha=0.3)
        ax1.set_xlabel(r"$X$ (m)", fontsize=16)
        ax1.set_ylabel(r"$Y$ (m)", fontsize=16)
        ax1.tick_params(axis='x', labelsize = 16)
        ax1.tick_params(axis='y', labelsize = 16)
        if show_radar:
            ax1.scatter(0.0, 0.0, marker='*', s=250, 
                    color='purple', label='Radar')
            ax1.set_xlim(min([-10.0, min(meas_X) - 20.0]))
            ax1.set_ylim(min([-10.0, min(meas_Y) - 20.0]))
        ax1.legend()
        ax1.grid(0.5)

        fig = plt.figure(figsize=figsize)
        ax2 = fig.add_subplot(1,1,1)
        ax2.plot(true_X, true_Y, 'd-k', label = r"Truth", markerfacecolor='None')
        ax2.plot(est_X, est_Y, 'r.', label = r"Estimates")
        ax2.plot(est_X, est_Y, 'r', alpha=0.3)
        ax2.set_xlabel(r"$X$ (m)", fontsize=16)
        ax2.set_ylabel(r"$Y$ (m)", fontsize=16)
        ax2.tick_params(axis='x', labelsize = 16)
        ax2.tick_params(axis='y', labelsize = 16)
        if show_radar:
            ax2.scatter(0.0, 0.0, marker='*', s=250, 
                    color='purple', label=r"Radar")
            ax2.set_xlim(min([-10.0, min(meas_X) - 20.0]))
            ax2.set_ylim(min([-10.0, min(meas_Y) - 20.0]))
        ax2.legend()
        ax2.grid(0.5)
        plt.show()
    
        fig = plt.figure(figsize=figsize)
        ax2 = fig.add_subplot(1, 1, 1)
        ax2.plot(true_Z, 'd-k', label='True z', markerfacecolor='None')
        ax2.plot(est_Z, 'r.', label='Est z', markerfacecolor='None')
        ax2.tick_params(axis='x', labelsize = 16)
        ax2.tick_params(axis='y', labelsize = 16)
        plt.show()
        
    def get_simulated_flight_df(self, flight_filename, data_dir, 
                                noise_factor=1.0, velocity_coef=0.1, 
                                offset_x=0, offset_y=0,
                                elevation_accuracy=0.5,
                                bearing_accuracy=0.6,
                                range_accuracy=3.14):
        """ Simulate radar measurements based on ground-truth
            UAV GPS measured trajectories and return results in 
            DataFrame format. """
        # create transition model (lower noise diff for altitude dim)
        transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(velocity_coef), 
             ConstantVelocity(velocity_coef), 
             ConstantVelocity(velocity_coef/5.0)])

        # Model coords = elev, bearing, range. Angles in radians
        # this radar measures range with an accuracy of +/- 3.14m, 
        # elevation accuracy +/- sqrt(10) rad  
        # bearing accuracy of +/- 0.6 rad
        meas_covar = np.diag([np.radians(np.sqrt(elevation_accuracy))**2,
                              np.radians(bearing_accuracy)**2,
                              range_accuracy**2])
    
        meas_covar_trk = CovarianceMatrix(1.0*meas_covar)
        meas_model = CartesianToElevationBearingRange(
                        ndim_state=6,
                        mapping=np.array([0, 2, 4]),
                        noise_covar=1.0*meas_covar_trk)
        predictor = ExtendedKalmanPredictor(transition_model)
        updater = ExtendedKalmanUpdater(measurement_model=None)
    
        # read ground truth file according to filename given
        ground_truth_reader = CSVGroundTruthReader(
        path=os.path.join(data_dir, flight_filename),
        state_vector_fields=['x', 'vel_x', 'y', 'vel_y', 'z'],
        time_field='time',
        path_id_field='track_id')
    
        # modify our ground truths by selected offsets
        offsets = [offset_x, offset_y]
        ground_truth_reader = OffsetCoordinates(ground_truth_reader, 
                                                offsets, [0, 2])
    
        # sensor simulated measurements - with desired noise factor
        radar_sensor = RadarElevationBearingRange(
            position_mapping=[0, 2, 4],
            noise_covar=noise_factor*meas_covar,
            ndim_state=6,
            mounting_offset=StateVector([0.0, 0.0, 0.0]))

        # create platform, with sensor at reference point, zero velocity
        platform = FixedPlatform(
            State([0, 0, 0, 0, 0, 0]),
            position_mapping=[0, 2, 4],
            sensors=[radar_sensor])
    
        # Create the detector and initialize it.
        detector = PlatformDetectionSimulator(ground_truth_reader, [platform])
    
        # define prior state for initiator
        prior_state = GaussianState(
            np.array([[0], [0], [0], [0], [0], [0]]),
            np.diag([0, 30.0, 0, 30.0, 0, 30.0])**2)

        # initiate initiator with prior & measurement model
        initiator = Initiator(prior_state, meas_model)
    
        # custom deleter - never delete track (since we know we only have one)
        deleter = MyDeleter()
    
        # we know there is only one measurement per scan, and so use
        # nearest neighbour associator
        meas = Euclidean()
        hypothesiser = DistanceHypothesiser(predictor, updater, meas)
        associator = NearestNeighbour(hypothesiser)

        # create tracker to track measurements and make state estimates
        tracker = SingleTargetTracker(initiator,
                                      deleter,
                                      detector,
                                      associator,
                                      updater)
    
        # create simulated measurements, performing tracking and estimation
        est_X=[]
        est_Y=[]
        est_Z=[]
        est_vel_X=[]
        est_vel_Y=[]
        est_vel_Z=[]
        meas_X=[]
        meas_Y=[]
        meas_Z=[]
        meas_vel_X=[]
        meas_vel_Y=[]
        meas_range=[]
        meas_elevation=[]
        meas_bearing=[]
        true_X = []
        true_Y = []
        true_Z = []
        true_vel_X=[]
        true_vel_Y=[]
        times=[]
    
        for time, tracks in tracker:
        
            times.append(time)
        
            for ground_truth in ground_truth_reader.groundtruth_paths:
                true_X.append(ground_truth.state_vector[0])
                true_vel_X.append(ground_truth.state_vector[1])
                true_Y.append(ground_truth.state_vector[2])
                true_vel_Y.append(ground_truth.state_vector[3])
                true_Z.append(ground_truth.state_vector[4])

            # Because this is a single target tracker, I know there is only 1 track.
            for track in tracks:

                # Get the corresponding measurement
                detection = track.states[-1].hypothesis.measurement
            
                # Convert measurement into xy (compensate for radar posn offset)
                xyz = meas_model.inverse_function(detection)
                meas_X.append(xyz[0])
                meas_vel_X.append(xyz[1])
                meas_Y.append(xyz[2])
                meas_vel_Y.append(xyz[3])
                meas_Z.append(xyz[4])
            
                # also add radar range, bearing & elevation measurements
                meas_range.append(detection.state_vector[2])
                meas_bearing.append(detection.state_vector[1])
                meas_elevation.append(detection.state_vector[0])

                vec = track.states[-1].state_vector
                est_X.append(vec[0])
                est_vel_X.append(vec[1])
                est_Y.append(vec[2])
                est_vel_Y.append(vec[3])
                est_Z.append(vec[4])
                est_vel_Z.append(vec[5])
    
        # return results as a dataframe
        return pd.DataFrame({'time' : times, 
                             'true_x' : true_X, 'true_y' : true_Y, 'true_z' : true_Z,
                             'true_vel_x' : true_vel_X, 'true_vel_y' : true_vel_Y,
                             'meas_x' : meas_X, 'meas_y' : meas_Y, 'meas_z' : meas_Z,
                             'meas_vel_x' : meas_vel_X, 'meas_vel_y' : meas_vel_Y,
                             'meas_range' : meas_range, 'meas_bearing' : meas_bearing,
                             'meas_elevation' : meas_elevation,
                             'est_x' : est_X, 'est_y' : est_Y, 'est_z' : est_Z,
                             'est_vel_x' : est_vel_X, 'est_vel_y' : est_vel_Y, 'est_vel_z' : est_vel_Z})
