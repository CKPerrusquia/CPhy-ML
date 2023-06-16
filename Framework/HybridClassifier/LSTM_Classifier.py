from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class LSTM_Network(object):
    
    def __init__(self, lstm_layers = [32, 32],
                 dense_units = 32, rnn_dropout = 0.4,
                 dense_activation = 'relu',
                 n_classes = 4, lr = 1e-4, epochs = 75,
                 batch_size = 256):
        
        self.model = []
        self.lstm_trg_time = []
        self.lstm_layers = lstm_layers
        self.dense_units = dense_units
        self.rnn_dropout = rnn_dropout
        self.dense_activation = dense_activation
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    def LSTM_model(self, Xtrain):
        
        inputs = keras.layers.Input(shape = (Xtrain.shape[1],
                                             Xtrain.shape[2]))
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                z = keras.layers.LSTM(self.lstm_layers[i], dropout = self.rnn_dropout,
                                      return_sequences = True)(inputs)
            else:
                z = keras.layers.LSTM(self.lstm_layers[i], dropout = self.rnn_dropout,
                                      return_sequences = True)(z)
        # flatten and apply dense layer
        dense = keras.layers.Flatten()(z)
        dense = keras.layers.Dense(self.dense_units, 
                                   activation = self.dense_activation)(dense)
        
        # define softmax output according to our number of classes
        outputs = keras.layers.Dense(self.n_classes, activation = 'softmax')(dense)
        
        # build overall model with inputs and outputs
        model = keras.models.Model(inputs = [inputs], outputs = [outputs])
        
        # compile model with desired settings
        model.compile(loss="categorical_crossentropy", 
                      optimizer=keras.optimizers.Adam(learning_rate = self.lr),
                      metrics=['accuracy'])
        self.model = model
        return model
    
    def train(self,Xtrain, ytrain, Xval, yval):       
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience=20, 
                                                      restore_best_weights=True,
                                                      monitor='val_loss')

        # use only our early stopper callback
        trg_callbacks = [early_stopper]
        lstm_start_time = time()

        lstm_history = self.model.fit(Xtrain, ytrain,
                                      epochs = self.epochs, batch_size = self.batch_size,
                                      validation_data=(Xval, yval),
                                      callbacks=trg_callbacks)
        
        lstm_trg_time = time() - lstm_start_time
        self.lstm_trg_time = lstm_trg_time
        print(f"LSTM training time: {lstm_trg_time}")
        return lstm_history
    
    def prediction(self, Xval, yval, Xtest, ytest, WINDOW_SIZE):
        # Initialize the Plotter
        Plots = Results()
        
        # Predict under the Xval dataset
        val_preds = self.model.predict(Xval)
        val_pred_labels = val_preds.argmax(axis=1)
        
        # time how long it takes to make predictions on test set
        start_pred_time = time()
        
        # Predict under the Xtest dataset
        test_preds = self.model.predict(Xtest)
        test_pred_labels = test_preds.argmax(axis=1)
        lstm_test_inference_time = time() - start_pred_time
        lstm_avg_pred_time = lstm_test_inference_time / Xtest.shape[0]

        print(f"Test Inference Time: {lstm_test_inference_time}")
        print(f"Individual Pred time: {lstm_avg_pred_time}")
        
        # get model classification results in final format for saving
        model_results = Plots.get_model_results(yval, val_pred_labels, ytest, 
                                                test_pred_labels, "LSTM")
        model_results['Training Time'] = self.lstm_trg_time
        model_results['Avg Pred Time'] = lstm_avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        print(f"{'-'*20} Validation Data {'-'*20}")
        print(classification_report(yval, val_pred_labels, digits=4))
        print(f"\n{'-'*20} Test Data {'-'*20}")
        print(classification_report(ytest, test_pred_labels, digits=4))
        
        # Confusion matrices for the validation and testing datasets
        Plots.plot_confusion_matrix(yval, val_pred_labels, 
                      title="Validation Data Confusion Matrix")
        
        Plots.plot_confusion_matrix(ytest, test_pred_labels, 
                      title="Test Data Confusion Matrix")
        return model_results
