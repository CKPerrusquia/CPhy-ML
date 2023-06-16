import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class TrajectoryStandardiser(object):
    """ Basic class for fitting and transforming trajectory
        sequenced data so that it is standardised & encoded. 
    """
    def __init__(self, num_upper_idx=None, 
                 cat_idx=None, cat_mappers=None,
                 summary_feats=True):
        self.means = []
        self.std_devs = []
        self.num_upper_idx = num_upper_idx
        self.cat_idx = cat_idx
        self.cat_mappers = cat_mappers
        self.one_hot_encoder = None
        self.summary_feats = summary_feats
        
    def fit(self, X):
        # if no upper idx given, assume no categoricals
        if self.num_upper_idx is None:
            self.num_upper_idx = X.shape[1]
        
        # encode categoricals and get seperately from trajectories
        X_cat_std = self.get_onehot_cats(X)
        
        means = X[:, :self.num_upper_idx, 
                  :].astype('float').mean(axis=(2,0)).reshape(-1, 1)
        stds = X[:, :self.num_upper_idx, 
                 :].astype('float').std(axis=(2,0)).reshape(-1, 1)
        self.means = means
        self.std_devs = stds
        return
        
    def fit_transform(self, X):
        """ Standardise numerical trajectories and one-hot 
            encode categorical variables. The trajectories and
            categorical features are then returned seperately.
            """
        if self.num_upper_idx is None:
            self.num_upper_idx = X.shape[1]
        
        # encode categoricals and get seperately from trajectories
        X_cat_std = self.get_onehot_cats(X)
        
        means = X[:, :self.num_upper_idx, 
                  :].astype('float').mean(axis=(2,0)).reshape(-1, 1)
        stds = X[:, :self.num_upper_idx, 
                 :].astype('float').std(axis=(2,0)).reshape(-1, 1)
        self.means = means
        self.std_devs = stds
        X_std = X.copy()
        X_std[:, :self.num_upper_idx, :] = ((X[:, :self.num_upper_idx, :] 
                                         - means) / stds)
        # keep only numerical features for trajectories
        X_num_std = X_std[:, :self.num_upper_idx, :].astype('float32')
        
        if self.summary_feats:
            # get summary trajectory features 
            X_seq_sum = self.get_seq_summary_feats(X_num_std)
        
            # combine categorical and summary feats into meta feats
            X_meta = np.hstack([X_cat_std, X_seq_sum])
        else:
            X_meta = X_cat_std.copy()
        
        return X_num_std, X_meta
    
    def transform(self, X):
        X_std = X.copy()
        if len(self.means) == 0:
            raise ValueError("Scaler not yet fitted. "
                    "You must call .fit(X) on training data first.")
        X_std[:, :self.num_upper_idx, :] = ((X[:, :self.num_upper_idx, :] 
                                         - self.means) / self.std_devs)
        
        # keep only numerical features for trajectories
        X_num_std = X_std[:, :self.num_upper_idx, :].astype('float32')
        
        # get categorical features one hot encoded
        X_cat_std = self.get_onehot_cats(X)
        
        if self.summary_feats:
            # get summary trajectory features 
            X_seq_sum = self.get_seq_summary_feats(X_num_std)
        
            # combine categorical and summary feats into meta feats
            X_meta = np.hstack([X_cat_std, X_seq_sum])
        else:
            X_meta = X_cat_std.copy()
        
        return X_num_std, X_meta
    
    def inverse_transform(self, X):
        pass
    
    def get_onehot_cats(self, X):
        """ One hot encode chosen feature """
        # iterate through each categorical feat and one-hot encode
        feat_encodings = []
        for feat_idx, cat_map in zip(self.cat_idx, self.cat_mappers):
            # encode feat (take only first timestep (all same))
            feat_encs = np.vectorize(
                cat_map.get)(X[:, feat_idx, 0]).reshape(-1, 1)
            feat_encodings.append(feat_encs)
        # combine encodings into single array
        feat_encodings = np.hstack(feat_encodings)
        
        # one hot encode categorical features
        if self.one_hot_encoder is not None:
            X_cat_oh = self.one_hot_encoder.transform(
                                feat_encodings).todense()
        else:
            self.one_hot_encoder = OneHotEncoder(handle_unknown='error')
            X_cat_oh = self.one_hot_encoder.fit_transform(
                                feat_encodings).todense()
        return X_cat_oh
    
    def get_seq_summary_feats(self, X):
        """ Function to generate a range of simple features from
            a given trajectory sequence. 
        """
        means = X.mean(axis=1)
        std_devs = X.std(axis=1)
        maxes = X.max(axis=1)
        mins = X.min(axis=1)
        X_simplified = np.hstack([means, std_devs, maxes, mins])
        return X_simplified
    
    def preprocess_labels(self, y, class_mappings, one_hot=True):
        """ Helper function for preprocessing labels """
        y_proc = pd.Series(y).apply(lambda x : class_mappings.get(x)).values
        if one_hot:
            y_proc = pd.get_dummies(y_proc)
        return y_proc
    
    def obtain_trajectory_features(self, X):
        """ Function to generate a range of simple features from
            a given trajectory sequence. """
        means = X.mean(axis=2)
        X_simplified = means
        return X_simplified
    
    def summarise_feature(self, df, y_msk_dict, label_name, feat_name):
        mean = df.loc[y_msk_dict[label_name], [feat_name]].mean().values[0]
        std = df.loc[y_msk_dict[label_name], [feat_name]].std().values[0]
        return mean, std