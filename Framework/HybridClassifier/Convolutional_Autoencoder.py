from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class Convolutional_Autoencoder(object):
    
    def __init__(self, layer_filters = [32, 16], 
                 activation = 'relu', 
                 dropout_rate = 0.4, kernel_size = 8,
                 optimizer = 'adam',
                 epochs = 200,
                 batch_size = 256):
        
        self.model = []
        self.layer_filters = layer_filters
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        
    def conv_autoencoder_model(self, Xtrain):
        
        """ One-Dimensional CNN Autoencoder Architecture, with dropout 
        """
        Xtrain = np.transpose(Xtrain, (0, 2, 1)).copy()
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape = Xtrain.shape[1:]))
        model.add(keras.layers.Conv1D(filters = self.layer_filters[0], kernel_size = self.kernel_size, 
                                      padding = 'same', strides = 2, activation = self.activation))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.Conv1D(filters = self.layer_filters[1], kernel_size = self.kernel_size, 
                                      padding = 'same', strides = 2, activation = self.activation))
    
        model.add(keras.layers.Conv1DTranspose(filters = self.layer_filters[1], kernel_size = self.kernel_size, 
                                      padding = 'same', strides = 2, activation = self.activation))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.Conv1DTranspose(filters = self.layer_filters[0], kernel_size = self.kernel_size, 
                                      padding = 'same', strides = 2, activation = self.activation))
    
        model.add(keras.layers.Conv1DTranspose(filters = Xtrain.shape[2], 
                                               kernel_size = self.kernel_size, padding = 'same', 
                                               activation = None))
    
        model.compile(optimizer = self.optimizer, loss='mse', metrics='mse')
        self.model = model
    
        return model
    
    def train(self, Xtrain, Xval):
        # create an early stopper callback
        Xtrain = np.transpose(Xtrain, (0, 2, 1)).copy()
        Xval = np.transpose(Xval, (0, 2, 1)).copy()
        
        early_stopper = keras.callbacks.EarlyStopping(patience = 10, 
                                              restore_best_weights=True,
                                              monitor='val_loss')

        # use only our early stopper callback
        ae_callbacks = [early_stopper]
        
        ca_history = self.model.fit(Xtrain, Xtrain, epochs = self.epochs, 
                                       batch_size = self.batch_size, callbacks = ae_callbacks,
                                       validation_data = (Xval, Xval))
        return ca_history

    def prediction(self, X):
        # Transform the data in a suitable representation
        X = np.transpose(X, (0, 2, 1)).copy()
        
        # Initialize the plotter
        Plots = Results()
        
        # Prediction under the X dataset
        recons = self.model.predict(X)
        recon_mses, recon_stds = Plots.get_avg_recon_results(recons , X)
        print(f"- Mean: {recon_mses}\n - Std Dev: {recon_stds}\n") 
        return recons, recon_mses, recon_stds