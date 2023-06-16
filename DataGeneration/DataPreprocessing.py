import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit

class Preprocessing:
    def form_time_sequences(self, data_df, cols, window_size=40, overlap_factor=2):
        """ Produce overlapping time-sequences from our data for use 
            in a time-sequencing deep learning model, e.g. LSTM / CNN """
        # keep only desired columns
        lean_df = data_df.loc[:, cols].copy()
        output_data = []
        for i in range(0, lean_df.shape[0] - window_size, 
                       int(window_size / overlap_factor)):
            output_data.append(lean_df.iloc[i : (i + window_size)])
        return np.stack(output_data)
    
    def get_class_trajectories(self, class_name, data_dir, col_names, window_size, 
                               overlap_factor=2):
        """ Helper function for creating a dataset of sub-trajectories
            from the chosen class directory and higher-level data
            directory. Each sub-trajectory has a window size equal
            to the given window_size argument. 
        """
        flt_filenames = [x for x in os.listdir(os.path.join(data_dir, class_name))
                    if x.endswith('.csv')]
    
        if 'est_z' in col_names:
            altitude = True
        else:
            altitude = False

        subtrajs = []
        labels = []
        flight_refs = []

        for filename in flt_filenames:
            flt_df = pd.read_csv(os.path.join(data_dir, 
                                              class_name,
                                              filename))
    
            # if window size too large, skip flight
            if flt_df.shape[0] <= window_size + 1:
                continue
    
            # get sub-trajectories for this flight
            sub_trajs = self.form_time_sequences(flt_df, col_names, 
                                            window_size, 
                                            overlap_factor)
    
            # append to all of our sub-trajectories
            subtrajs.append(sub_trajs)
        
            # get original flight filename for reference
            original_ref = "_".join(filename.split('_')[:-1])
        
            # give each sub-traj the same filename reference
            flt_refs = np.array([original_ref for _ 
                                 in range(sub_trajs.shape[0])])
            flight_refs.append(flt_refs)
    
            # also add correct number of labels to align with sub-trajs
            sub_traj_labels = np.array([class_name for _ 
                                        in range(sub_trajs.shape[0])])
            labels.append(sub_traj_labels)

        # convert data into numpy arrays
        subtrajs = np.concatenate(subtrajs)
        labels = np.concatenate(labels)
        flight_refs = np.concatenate(flight_refs)
        trajs_std = subtrajs.copy()

        # get first x,y,z co-ordinates of each sub-traj
        n_rows = subtrajs.shape[0]
        x0 = subtrajs[:, 0, 0].reshape((n_rows, 1))
        y0 = subtrajs[:, 0, 1].reshape((n_rows, 1))
        if altitude:
            z0 = subtrajs[:, 0, 2].reshape((n_rows, 1))

        # centralise sub-trajectories to start at (0,0,0)
        trajs_std[:, :, 0] = subtrajs[:, :, 0] - x0
        trajs_std[:, :, 1] = subtrajs[:, :, 1] - y0
        if altitude:
            trajs_std[:, :, 2] = subtrajs[:, :, 2] - z0
    
        # re-order results into convenient order for processing later
        # (row, features, timestep)
        trajs_std = np.transpose(trajs_std, (0, 2, 1))
    
        return trajs_std, labels, flight_refs
    
    def OutliersRemoval(self, UAV_TYPE_IDX, X, y, flight_refs, ANOMALOUS_THRESHOLD = 11000):
        anomalous_mask = X[:,:UAV_TYPE_IDX,:].max(axis=(1,2))>ANOMALOUS_THRESHOLD
        print(flight_refs[anomalous_mask])
        
        X = X[~anomalous_mask]
        y = y[~anomalous_mask]
        flight_refs = flight_refs[~anomalous_mask]
        
        return X, y, flight_refs
    
    def Train_Val_Test_Split(self, X, y, flight_refs, random_state, n_splits = 1, test_size = 0.10, val_size = 0.20):
        train_test_splitter = GroupShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = random_state)
        train_test_split = train_test_splitter.split(X, y, groups = flight_refs)
        train_idx, test_idx = next(train_test_split)

        X_temp = X[train_idx]
        y_temp = y[train_idx]
        flight_refs_temp = flight_refs[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # split again to get validation data (20% of remaining training data)
        train_val_splitter = GroupShuffleSplit(n_splits = n_splits, test_size = val_size, random_state = random_state)
        train_val_split = train_val_splitter.split(X_temp, y_temp, groups = flight_refs_temp)
        train_idx, val_idx = next(train_val_split)

        X_train = X_temp[train_idx]
        y_train = y_temp[train_idx]
        X_val = X_temp[val_idx]
        y_val = y_temp[val_idx]
        
        train_flight_refs = flight_refs_temp[train_idx]
        val_flight_refs = flight_refs_temp[val_idx]
        test_flight_refs = flight_refs[test_idx]

        print(X_train.shape, y_train.shape, X_test.shape, 
              y_test.shape, X_val.shape, y_val.shape,
             np.unique(train_flight_refs).shape, np.unique(val_flight_refs).shape, np.unique(test_flight_refs).shape)
        return X_train, y_train, X_test, y_test, X_val, y_val, train_flight_refs, val_flight_refs, test_flight_refs
    
    def time_seqs_with_lookahead(self, data_df, cols, window_size=20, 
                                  overlap_factor=10, lookahead=20):
        """ Produce overlapping time-sequences from our data for use 
            in a time-sequencing deep learning model, e.g. LSTM / CNN """
        # keep only desired columns
        lean_df = data_df.loc[:, cols].copy()
        
        max_idx = lean_df.shape[0]
        subtrajs = []
        lookahead_seqs = []
        
        for i in range(0, max_idx - (lookahead + window_size), 
                       int(max(1, window_size / overlap_factor))):
            # append subtrajectory
            subtrajs.append(lean_df.iloc[i : (i + window_size)])
            
            # also get sequence ahead of current sub-trajectory
            lookahead_seqs.append(lean_df.iloc[i + window_size : 
                                               (i + window_size + lookahead)])
            
        return np.stack(subtrajs), np.stack(lookahead_seqs)
    
    def get_min_max_differences(self, sub_traj, lookahead_traj):
        """ Determine the minimum and maximum difference between
        the las timestep on the given sub-trajectory and the
        co-ordinates in the lookahead trajectory """
        # get first timestep co-ordinates (x,y,z) for lookahead
        last_timestep = sub_traj[:, -1, :3]
    
        # get maximum bounds for lookahead seq
        max_lookaheads = lookahead_traj[:, :, :3].max(axis=1)
        min_lookaheads = lookahead_traj[:, :, :3].min(axis=1)
    
        # find maximum bounds relative to last input timestep
        max_diffs = max_lookaheads - last_timestep
        min_diffs = min_lookaheads - last_timestep
    
        # combine into one set of labels (xmin,ymin,zmin,xmax,ymax,zmax)
        min_max_diffs = np.hstack([min_diffs, max_diffs])
    
        return min_max_diffs
    
    def get_class_reg_trajectories(self, class_name, data_dir, 
                                   col_names, window_size=10,
                                   overlap_factor=5, lookahead=30,
                                   sample_times=[10, 20, 30, 40]):
        """ Helper function for creating a dataset of sub-trajectories
            from the chosen class directory and higher-level data
            directory for maximum / minimum bounds regression. 
        
            Labels are created from the lookahead sequences corresponding
            to maximum bounds (in 3D) for each of the given time intervals
            in the argument (sample_times)
        """
        flt_filenames = [x for x in os.listdir(os.path.join(data_dir, class_name))
                         if x.endswith('.csv')]
    
        all_subtrajs = []
        all_lookaheads = []
        flight_refs = []

        for filename in flt_filenames:
            flt_df = pd.read_csv(os.path.join(data_dir, 
                                              class_name,
                                              filename))
    
            # if window size too large, skip flight
            if flt_df.shape[0] <= window_size + lookahead + 1:
                continue
            
            # add intent type to data
            flt_df['uav_intent'] = class_name
    
            # get sub-trajectories for this flight
            sub_trajs, lookaheads = self.time_seqs_with_lookahead(flt_df, col_names, 
                                                                  window_size, 
                                                                  overlap_factor,
                                                                  lookahead)
    
            # append to all of our sub-trajectories & lookaheads
            all_subtrajs.append(sub_trajs)
            all_lookaheads.append(lookaheads)
        
            # get original flight filename for reference
            original_ref = "_".join(filename.split('_')[:-1])
        
            # give each sub-traj the same filename reference
            flt_refs = np.array([original_ref for _ 
                                 in range(sub_trajs.shape[0])])
            flight_refs.append(flt_refs)

        # convert data into numpy arrays
        all_subtrajs = np.concatenate(all_subtrajs)
        all_lookaheads = np.concatenate(all_lookaheads)
        flight_refs = np.concatenate(flight_refs)
    
        # standardise all sub-trajs to start at (0,0,0)
        trajs_std = all_subtrajs.copy()
        lookahead_std = all_lookaheads.copy()
    
        # get first x,y,z co-ordinates of each sub-traj
        n_rows = all_subtrajs.shape[0]
        x0 = all_subtrajs[:, 0, 0].reshape((n_rows, 1))
        y0 = all_subtrajs[:, 0, 1].reshape((n_rows, 1))
        z0 = all_subtrajs[:, 0, 2].reshape((n_rows, 1))

        # centralise sub-trajectories to start at (0,0,0)
        trajs_std[:, :, 0] = all_subtrajs[:, :, 0] - x0
        trajs_std[:, :, 1] = all_subtrajs[:, :, 1] - y0
        trajs_std[:, :, 2] = all_subtrajs[:, :, 2] - z0
    
        # also re-centre lookahead trajectories using above translations
        lookahead_std[:, :, 0] = all_lookaheads[:, :, 0] - x0
        lookahead_std[:, :, 1] = all_lookaheads[:, :, 1] - y0
        lookahead_std[:, :, 2] = all_lookaheads[:, :, 2] - z0
    
        # get min/max bound labels for each sample time
        min_max_labels = []
        for sample_time in sample_times:
            sample_min_maxes = self.get_min_max_differences(trajs_std, 
                                                            lookahead_std[:, :sample_time, :])
            min_max_labels.append(sample_min_maxes)
    
        # horizontally stack labels into array of labels for each input
        min_max_labels = np.hstack(min_max_labels)
    
        # re-order results into more convenient order:
        # (row, features, timestep)
        trajs_std = np.transpose(trajs_std, (0, 2, 1))
        lookahead_std = np.transpose(lookahead_std, (0, 2, 1))
    
        return trajs_std, lookahead_std, min_max_labels, flight_refs
    
    def DataSplitter(self, X, y, y_lookaheads, flight_refs,
                     val_size = 0.20, test_size = 0.10,
                     n_splits = 1, random_state = 18):
        train_test_splitter = GroupShuffleSplit(test_size = test_size, n_splits = 1, random_state = random_state )
        train_test_split = train_test_splitter.split(X, y, groups=flight_refs)
        train_idx, test_idx = next(train_test_split)

        X_temp = X[train_idx].copy()
        y_temp = y[train_idx].copy()
        y_temp_traj = y_lookaheads[train_idx].copy()
        flight_refs_temp = flight_refs[train_idx].copy()
        X_test = X[test_idx].copy()
        y_test = y[test_idx].copy()
        y_test_traj = y_lookaheads[test_idx].copy()
        y_test_refs = flight_refs[test_idx].copy()

        # split again to get validation data (val_size of remaining training data)
        train_val_splitter = GroupShuffleSplit(test_size = val_size, n_splits = 1, random_state = random_state)
        train_val_split = train_val_splitter.split(X_temp, y_temp, groups=flight_refs_temp)
        train_idx, val_idx = next(train_val_split)

        X_train = X_temp[train_idx].copy()
        y_train = y_temp[train_idx].copy()
        y_train_traj = y_temp_traj[train_idx].copy()
        y_train_refs = flight_refs_temp[train_idx].copy()

        X_val = X_temp[val_idx].copy()
        y_val = y_temp[val_idx].copy()
        y_val_traj = y_temp_traj[val_idx].copy()
        y_val_refs = flight_refs_temp[val_idx].copy()

        print(X_train.shape, y_train.shape, y_train_traj.shape)
        print(X_val.shape, y_val.shape, y_val_traj.shape)
        print(X_test.shape, y_test.shape, y_test_traj.shape)
        return (X_train, y_train, y_train_traj,
                X_val, y_val, y_val_traj,
                X_test, y_test, y_test_traj) 
