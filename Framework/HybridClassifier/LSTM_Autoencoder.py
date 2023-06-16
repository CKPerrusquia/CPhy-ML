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

class LSTM_Autoencoder(object):
    
    def __init__(self, codings_size=8, 
                 lstm_layers=[32, 16],
                 rnn_dropout=0.4,
                 lr=1e-3,
                 epochs = 200,
                 batch_size = 256):
        
        self.model = []
        self.codings_size = codings_size
        self.lstm_layers = lstm_layers
        self.rnn_dropout = rnn_dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    def lstm_autoencoder_model(self, Xtrain):
        
        """ LSTM autoencoder defined using functional API """
        # define encoder component
        inputs = keras.layers.Input(shape=(Xtrain.shape[1], 
                                           Xtrain.shape[2]))      
        
        z = keras.layers.LSTM(self.lstm_layers[0], dropout = self.rnn_dropout,
                              return_sequences = True)(inputs)
        z = keras.layers.LSTM(self.lstm_layers[1], dropout = self.rnn_dropout,
                              return_sequences = True)(z)
        codings = keras.layers.LSTM(self.codings_size, return_sequences = False)(z)
        encoder = keras.models.Model(inputs=[inputs], outputs=[codings])
    
        # define decoder component
        decoder_inputs = keras.layers.Input(shape=[self.codings_size])
        x = keras.layers.RepeatVector(Xtrain.shape[1], 
                                      input_shape = [self.codings_size])(decoder_inputs)
       
        x = keras.layers.LSTM(self.lstm_layers[1], return_sequences = True)(x)
        x = keras.layers.LSTM(self.lstm_layers[0], return_sequences = True)(x)
        decoder_output = keras.layers.TimeDistributed(keras.layers.Dense(
                                        Xtrain.shape[2]))(x)
        decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[decoder_output])
    
        # obtain codings and use these to build reconstructions using decoder
        codings = encoder(inputs)
        reconstructions = decoder(codings)
    
        # build overall autoencoder
        lstm_ae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])
    
        # compile model
        lstm_ae.compile(loss="mse", 
                        optimizer=keras.optimizers.Adam(learning_rate = self.lr), 
                        metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
        self.model = lstm_ae
    
        return lstm_ae
    
    def train(self, Xtrain, Xval):
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience=25, 
                                              restore_best_weights=True,
                                              monitor='val_loss')

        # use only our early stopper callback
        ae_callbacks = [early_stopper]
        
        clstma_history = self.model.fit(Xtrain, Xtrain, epochs = self.epochs, 
                                       batch_size = self.batch_size, callbacks = ae_callbacks,
                                       validation_data = (Xval, Xval))
        return clstma_history
    
    def prediction(self, X):
        # Initialize the plotter
        Plots = Results()
        
        # Predict under the X dataset
        recons = self.model.predict(X)
        
        # Reconstruction results for saving
        recon_mses, recon_stds = Plots.get_avg_recon_results(recons , X)
        print(f"- Mean: {recon_mses}\n - Std Dev: {recon_stds}\n") 
        return recons, recon_mses, recon_stds
  