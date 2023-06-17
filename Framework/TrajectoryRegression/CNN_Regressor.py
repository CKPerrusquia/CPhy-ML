from Framework.TrajectoryRegression.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
from Framework.HybridClassifier.Attention_Layer import AttentionLayer
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


class CNN_Regressor:
    
    def __init__(self, conv_filters = 20, kernel_size = 8,
                 cat_dense_units = 64, dense_units = 64, 
                 dense_dropout = 0.4, activation = 'relu',
                 loss = 'mean_squared_error', lr = 1e-3, 
                 n_outputs = 6, epochs = 75, batch_size = 256):
        
        self.model = []
        self.cnn_trg_time = []
        self.train_rmse = []
        self.train_mae = []
        self.train_r2 = []
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.cat_dense_units = cat_dense_units
        self.dense_units = dense_units
        self.dense_dropout = dense_dropout 
        self.activation = activation
        self.lr = lr
        self.loss = loss
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        
    def MultiInputConv(self, input_seqs, input_cats):
        """ Convolutional model for trajectory classification.
            Uses multiple inputs - both sequences and categorical features.
            Uses a Conv layer and several LSTM layers, followed by final 
            Dense network to classify input sequences. """
    
        seq_input = keras.layers.Input(shape = (input_seqs.shape[1], 
                                               input_seqs.shape[2]))
    
        # Conv layers for sequential inputs
        cnn_z = keras.layers.Conv1D(filters = self.conv_filters, kernel_size = self.kernel_size, 
                                    padding = 'same', strides = 2, 
                                    activation = self.activation)(seq_input)
    
        # add dropout regularisation to prevent overfitting
        z = keras.layers.Dropout(self.dense_dropout, name = 'cnn_dropout')(cnn_z)
    
        # follow-up convolutional layer
        z = keras.layers.Conv1D(filters = self.conv_filters, kernel_size = self.kernel_size, 
                                padding = 'same', strides = 2, 
                                activation = self.activation)(z)
    
        # flatten and apply dense layer
        cnn_dense = keras.layers.Flatten()(z)
    
        # define categorical inputs
        cat_input = keras.layers.Input(shape = (input_cats.shape[1]))
    
        # simple dense layer to process categorical inputs
        cat_dense = keras.layers.Dense(self.cat_dense_units, 
                                       activation = self.activation)(cat_input)
    
        # combine embeddings from both rnn and categorical components
        concat = keras.layers.Concatenate()([cat_dense, cnn_dense])
    
        # add dropout for regularisation
        dense = keras.layers.Dropout(self.dense_dropout)(concat)
    
        # dense layer to process combined embeddings
        dense = keras.layers.Dense(self.dense_units, 
                                   activation = self.activation)(concat)
    
        # define dense output (no activation) for regression
        outputs = keras.layers.Dense(self.n_outputs)(dense)
    
        # build overall model with inputs and outputs
        model = keras.models.Model(inputs=[seq_input, cat_input], outputs=[outputs])
    
        # compile model with desired settings
        model.compile(loss = self.loss, 
                      optimizer = keras.optimizers.Adam(learning_rate = self.lr), 
                      metrics = [tf.keras.metrics.RootMeanSquaredError()])
        
        self.model = model
        return model
    
    def train(self, Xtrain_seq, Xtrain_meta, ytrain,
              Xval_seq, Xval_meta, yval):
        
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience = 15, 
                                              restore_best_weights = True,
                                              monitor = 'val_loss')

        # create learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, 
                                                   patience = 5, verbose = 0, 
                                                   min_delta = 0.001, mode = 'min')

        # use only our early stopper callback
        trg_callbacks = [early_stopper, lr_scheduler]

        # train model, obtain results, and record total training time
        start_time = time()
        cnn_history = self.model.fit([Xtrain_seq, Xtrain_meta], ytrain,
                                     epochs = self.epochs, batch_size = self.batch_size,
                                     validation_data = ([Xval_seq, Xval_meta], yval),
                                     callbacks = trg_callbacks)
        cnn_trg_time = time() - start_time
        self.cnn_trg_time = cnn_trg_time
        
        train_preds = self.model.predict([Xtrain_seq, Xtrain_meta])
        train_rmse = mean_squared_error(ytrain, train_preds, squared = False)
        train_mae = mean_absolute_error(ytrain, train_preds)
        train_r2 = r2_score(ytrain, train_preds)
        
        self.train_rmse = train_rmse
        self.train_mae = train_mae
        self.train_r2 = train_r2
        
        return cnn_history
    
    def simple_prediction(self, X):
        preds = self.model.predict(X)
        return preds
        
    def prediction(self, Xval_seq, Xval_meta, yval,
                   Xtest_seq, Xtest_meta, ytest,
                   WINDOW_SIZE):
        
        # Initialize the plotter
        Plots = Results()
        val_preds = self.model.predict([Xval_seq, Xval_meta])
        
        # time how long it takes to make predictions on test set
        start_pred_time = time()
        test_preds = self.model.predict([Xtest_seq, Xtest_meta])
        cnn_test_pred_time = time() - start_pred_time
        cnn_avg_pred_time = cnn_test_pred_time / Xtest_seq.shape[0]

        print(f"CLSTM Test Inference Time: {cnn_test_pred_time}")
        print(f"CLSTM Individual Pred time: {cnn_avg_pred_time}")

        # add our regression metric results to final dataframe
        model_results = Plots.get_model_results(yval, val_preds, 
                                                ytest, test_preds,
                                                "CNN Regressor")
        model_results['Training Time'] = self.cnn_trg_time
        model_results['Avg Pred Time'] = cnn_avg_pred_time
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