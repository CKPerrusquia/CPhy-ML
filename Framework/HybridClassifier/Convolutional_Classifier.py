from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
from Framework.HybridClassifier.Attention_Layer import AttentionLayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np
from sklearn.metrics import classification_report

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class Conv_Network(object):
    
    def __init__(self, activation ='relu', dense_units = 64,
                 dense_dropout = 0.4, filters = [32, 64],
                 kernel_size = 12, optimizer = 'adam', 
                 attention = False, n_classes = 4, lr = 1e-3, 
                 batch_size = 128, epochs = 75):
        
        self.model = []
        self.conv_trg_time = []
        self.activation = activation
        self.dense_units = dense_units
        self.dense_dropout = dense_dropout
        self.filters = filters
        self.kernel_size = kernel_size
        self.optimizer = optimizer
        self.attention = attention
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
    
    def Conv_model(self,Xtrain):
        
        """ One-Dimensional CNN Autoencoder Architecture, with dropout """
        
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape = (Xtrain.shape[1], Xtrain.shape[2])))
        model.add(keras.layers.Conv1D(filters = self.filters[0], kernel_size = self.kernel_size, 
                                padding = 'same', strides = 2, activation = self.activation))
        #model.add(keras.layers.Dropout(dense_dropout))
        model.add(keras.layers.Conv1D(filters = self.filters[1], kernel_size = self.kernel_size, 
                                padding = 'same', strides = 2, activation = self.activation))

        # flatten and apply dense layer
        if self.attention:
            Attention = AttentionLayer()
            model.add(Attention)
        else:
            model.add(keras.layers.Flatten())
            
        model.add(keras.layers.Dropout(self.dense_dropout))
        model.add(keras.layers.Dense(self.dense_units, activation = self.activation))
              
        # define softmax output according to our number of classes
        model.add(keras.layers.Dense(self.n_classes, activation = 'softmax'))
    
        # compile model with desired settings
        model.compile(loss = "categorical_crossentropy", 
                      optimizer = keras.optimizers.Adam(learning_rate = self.lr), 
                      metrics = ['accuracy'])
        
        self.model = model
        return model
    
    def train(self, Xtrain, ytrain, Xval, yval):
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience = 15, 
                                                      restore_best_weights = True,
                                                      monitor = 'val_accuracy')
        
        # create learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.75, 
                                                         patience = 10, verbose = 0, 
                                                         min_delta = 0.001, mode = 'min')

        # use only our early stopper callback
        trg_callbacks = [early_stopper, lr_scheduler]
        
        conv_start_time = time()

        conv_history = self.model.fit(Xtrain, ytrain,
                                          epochs = self.epochs, batch_size = self.batch_size,
                                          validation_data = (Xval, yval),
                                          callbacks = trg_callbacks)

        conv_trg_time = time() - conv_start_time
        self.conv_trg_time = conv_trg_time
        
        print(f"Conv training time: {conv_trg_time}")
        
        return conv_history
    
    def prediction(self, Xval, yval, Xtest, ytest, WINDOW_SIZE):
        # Initialize the plotter
        Plots = Results()
        
        # Predict under the Xval dataset
        val_preds = self.model.predict(Xval)
        val_pred_labels = val_preds.argmax(axis=1)
        
        # time how long it takes to make predictions on test set
        start_pred_time = time()
        
        # Predict under the Xtest dataset
        test_preds = self.model.predict(Xtest)
        test_pred_labels = test_preds.argmax(axis=1)
        
        # Inference and average time
        conv_test_inference_time = time() - start_pred_time
        conv_avg_pred_time = conv_test_inference_time / Xtest.shape[0]

        print(f"Test Inference Time: {conv_test_inference_time}")
        print(f"Individual Pred time: {conv_avg_pred_time}")
        
        # get model classification results in final format for saving
        model_results = Plots.get_model_results(yval, val_pred_labels, ytest, 
                                                test_pred_labels, "Conv")
        model_results['Training Time'] = self.conv_trg_time
        model_results['Avg Pred Time'] = conv_avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        print(f"{'-'*20} Validation Data {'-'*20}")
        print(classification_report(yval, val_pred_labels, digits=4))
        print(f"\n{'-'*20} Test Data {'-'*20}")
        print(classification_report(ytest, test_pred_labels, digits=4))
        
        # Plot confusion matrices for the validation and testing datasets
        Plots.plot_confusion_matrix(yval, val_pred_labels, 
                      title="Validation Data Confusion Matrix")
        
        Plots.plot_confusion_matrix(ytest, test_pred_labels, 
                      title="Test Data Confusion Matrix")
        
        return model_results
        