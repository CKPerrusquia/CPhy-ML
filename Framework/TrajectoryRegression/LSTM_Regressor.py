from Framework.TrajectoryRegression.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from numpy.random import seed
seed(14)
from tensorflow.random import set_seed
set_seed(14)

class MultiInputLSTM:
    
    def __init__(self, lstm_layers = [16], 
                 cat_dense_units = 16, dense_units = 64, 
                 rnn_dropout = 0.4, dense_dropout = 0.4, 
                 dense_activation = 'relu', n_outputs = 6,
                 epochs = 75, batch_size = 256):
        
        self.model = []
        self.lstm_trg_time = []
        self.train_rmse = []
        self.train_mae = []
        self.train_r2 = []
        self.lstm_layers = lstm_layers
        self.cat_dense_units = cat_dense_units
        self.dense_units = dense_units
        self.rnn_dropout = rnn_dropout
        self.dense_dropout = dense_dropout 
        self.dense_activation = dense_activation
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
    
    def MultiInputLSTM(self, input_seqs, input_cats):
        """ Basic LSTM model for trajectory classification.
            Uses several LSTM layers, followed by final 
            Dense network to classify input sequences. """
    
        seq_input = keras.layers.Input(shape = (input_seqs.shape[1], 
                                              input_seqs.shape[2]))
    
        # lstm layers to process sequential inputs
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                z = keras.layers.LSTM(self.lstm_layers[i], dropout = self.rnn_dropout,
                              return_sequences = True)(seq_input)
            else:
                z = keras.layers.LSTM(self.lstm_layers[i], dropout = self.rnn_dropout,
                              return_sequences = True)(z)
    
        # flatten and apply dense layer
        rnn_dense = keras.layers.Flatten()(z)
    
        # define categorical inputs
        cat_input = keras.layers.Input(shape = (input_cats.shape[1]))
    
        # simple dense layer to process categorical inputs
        cat_dense = keras.layers.Dense(self.cat_dense_units, 
                    activation = self.dense_activation)(cat_input)
    
        # combine embeddings from both rnn and categorical components
        concat = keras.layers.Concatenate()([cat_dense, rnn_dense])
    
        # add dropout for regularisation
        dense = keras.layers.Dropout(self.dense_dropout)(concat)
    
        # dense layer to process combined embeddings
        dense = keras.layers.Dense(self.dense_units, 
                        activation = self.dense_activation)(concat)
    
        # define dense output (no activation) for regression
        outputs = keras.layers.Dense(self.n_outputs)(dense)
    
        # build overall model with inputs and outputs
        model = keras.models.Model(inputs = [seq_input, cat_input], outputs=[outputs])
    
        # compile model with desired settings
        model.compile(loss = "mean_squared_error", 
                      optimizer = "adam", 
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        
        self.model = model
        
        return model
    
    def train(self, Xtrain_seq, Xtrain_meta, ytrain,
              Xval_seq, Xval_meta, yval):
        
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience = 10, 
                                                      restore_best_weights = True,
                                                      monitor = 'val_loss')

        # use only our early stopper callback
        trg_callbacks = [early_stopper]
        
        lstm_start_time = time()

        lstm_history = self.model.fit([Xtrain_seq, Xtrain_meta], ytrain,
                                      epochs = self.epochs, batch_size = self.batch_size,
                                      validation_data=([Xval_seq, Xval_meta], yval),
                                      callbacks = trg_callbacks)

        # get total training time
        lstm_trg_time = time() - lstm_start_time
        self.lstm_trg_time = lstm_trg_time
        
        train_preds = self.model.predict([Xtrain_seq, Xtrain_meta])
        train_rmse = mean_squared_error(ytrain, train_preds, squared = False)
        train_mae = mean_absolute_error(ytrain, train_preds)
        train_r2 = r2_score(ytrain, train_preds)
        
        self.train_rmse = train_rmse
        self.train_mae = train_mae
        self.train_r2 = train_r2
        
        return lstm_history
    
    def prediction(self, Xval_seq, Xval_meta, yval,
                   Xtest_seq, Xtest_meta, ytest,
                   WINDOW_SIZE):
        
        # Initialize the plotter
        Plots = Results()
        val_preds = self.model.predict([Xval_seq, Xval_meta])
        
        # time how long it takes to make predictions on test set
        start_pred_time = time()
        test_preds = self.model.predict([Xtest_seq, Xtest_meta])
        lstm_test_pred_time = time() - start_pred_time
        lstm_avg_pred_time = lstm_test_pred_time / Xtest_seq.shape[0]

        print(f"LSTM Test Inference Time: {lstm_test_pred_time}")
        print(f"LSTM Individual Pred time: {lstm_avg_pred_time}")

        # add our regression metric results to final dataframe
        model_results = Plots.get_model_results(yval, val_preds, 
                                                ytest, test_preds,
                                                "LSTM Regressor")
        model_results['Training Time'] = self.lstm_trg_time
        model_results['Avg Pred Time'] = lstm_avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        # Compute the metrics
        val_rmse = mean_squared_error(yval, val_preds, squared=False)
        test_rmse = mean_squared_error(ytest, test_preds, squared=False)
        
        val_mae = mean_absolute_error(yval, val_preds)
        test_mae = mean_absolute_error(ytest, test_preds)
        
        val_r2 = r2_score(yval, val_preds)
        test_r2 = r2_score(ytest, test_preds)
        
        print("Training Data:")
        print(f"    - RMSE: {self.train_rmse:.4f}")
        print(f"    - MAE: {self.train_mae:.4f}")
        print(f"    - R^2: {self.train_r2:.4f}")

        print("\nValidation Data:")
        print(f"    - RMSE: {val_rmse:.4f}")
        print(f"    - MAE: {val_mae:.4f}")
        print(f"    - R^2: {val_r2:.4f}")

        print("\nTest Data:")
        print(f"    - RMSE: {test_rmse:.4f}")
        print(f"    - MAE: {test_mae:.4f}")
        print(f"    - R^2: {test_r2:.4f}")
        
        return model_results, val_preds, test_preds
        