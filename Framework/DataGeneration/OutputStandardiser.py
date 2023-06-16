import numpy as np
import pandas as pd

class OutputStandardiser(object):
    """ Basic class for fitting and transforming trajectory
        output regression labels so that they are standardised.
    """
    def __init__(self):
        self.means = []
        self.std_devs = []
        
    def fit(self, y):
        means = y.astype('float').mean(axis=0)
        std_devs = y.astype('float').std(axis=0)
        self.means = means
        self.std_devs = std_devs
        return
        
    def fit_transform(self, y):
        means = y.astype('float').mean(axis=0)
        std_devs = y.astype('float').std(axis=0)
        self.means = means
        self.std_devs = std_devs
        y_std = (y - means) / std_devs
        return y_std
    
    def transform(self, y):
        y_std = y.copy()
        if len(self.means) == 0:
            raise ValueError("Scaler not yet fitted. "
                    "You must call .fit(y) on training outputs first.")
        y_std = (y - self.means) / self.std_devs
        return y_std
    
    def inverse_transform(self, y_std):
        y = (y_std * self.std_devs) + self.means
        return y
    
class OutputNormaliser(object):
    """ Basic class for fitting and transforming trajectory
        output regression labels so that they are Min-Max scaled.
    """
    def __init__(self):
        self.mins = []
        self.maxes = []
        
    def fit(self, y):
        mins = y.astype('float').min(axis=0)
        maxes = y.astype('float').max(axis=0)
        self.mins = mins
        self.maxes = maxes
        return
        
    def fit_transform(self, y):
        mins = y.astype('float').min(axis=0)
        maxes = y.astype('float').max(axis=0)
        self.mins = mins
        self.maxes = maxes
        y_norm = (y - mins) / (maxes - mins)
        return y_norm
    
    def transform(self, y):
        if len(self.mins) == 0:
            raise ValueError("Normaliser not yet fitted. "
                    "You must call .fit(y) on training outputs first.")
        y_norm = (y - self.mins) / (self.maxes - self.mins)
        return y_norm
    
    def inverse_transform(self, y_norm):
        y = ((self.maxes - self.mins) * y_norm) + self.mins
        return y