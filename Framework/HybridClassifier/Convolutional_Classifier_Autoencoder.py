from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np
from sklearn.metrics import accuracy_score

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class Conv_AE_Classifier(object):
    
    def __init__(self, layer_filters = [32, 32], 
                       activation = 'relu', dense_units = 20, 
                       dropout_rate = 0.4, kernel_size = 16, 
                       optimizer = 'adam', lr = 5e-4,
                       clf_model_weight = 0.85,
                       n_classes = 4,
                       epochs = 150, batch_size = 128):
        
        self.model = []
        self.layer_filters = layer_filters
        self.activation = activation
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.optimizer = optimizer
        self.lr = lr
        self.clf_model_weight = clf_model_weight
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        
    def Conv_AE_Classifier(self, Xtrain):
        """ Composite One-Dimensional CNN Autoencoder and Trajectory 
            Classifier Architecture, with dropout and multi-output optimisation.
        """
        Xtrain = np.transpose(Xtrain, (0, 2, 1)).copy()
        # initial encoder layers
        inputs = keras.layers.Input(shape = Xtrain.shape[1:], name='Input')
        z = keras.layers.Conv1D(filters = self.layer_filters[0], kernel_size = self.kernel_size, 
                        padding = 'same', strides = 2, name = 'encoder_1',
                        activation = self.activation)(inputs)
        z = keras.layers.Dropout(self.dropout_rate, name = 'encoder_2')(z)
        z = keras.layers.Conv1D(filters = self.layer_filters[1], kernel_size = self.kernel_size, 
                        padding = 'same', strides = 2, name = 'encoder_3',
                        activation = self.activation)(z)
    
        # final dense network for classification for n output classes
        clf_out = keras.layers.Flatten(name = 'clf_1')(z)
        clf_out = keras.layers.Dropout(self.dropout_rate, name = 'clf_2')(clf_out)
        clf_out = keras.layers.Dense(self.dense_units, name = 'clf_3',
                                 activation = self.activation)(clf_out)
        clf_out = keras.layers.Dense(self.n_classes, name = 'clf_out',
                                 activation = 'softmax')(clf_out)
    
        # define decoder components
        x = keras.layers.Conv1DTranspose(filters = self.layer_filters[1], kernel_size = self.kernel_size, 
                                padding = 'same', strides = 2, name = 'decoder_1',
                                activation = self.activation)(z)
        x = keras.layers.Conv1DTranspose(filters = self.layer_filters[0], kernel_size = self.kernel_size, 
                                padding = 'same', strides = 2, name = 'decoder_2',
                                activation = self.activation)(x)
    
        # final reconstruction layer for decoder
        x = keras.layers.Conv1DTranspose(filters = Xtrain.shape[2], name = 'ae_out', 
                                         kernel_size = self.kernel_size, padding = 'same', 
                                         activation = None)(x)
    
        # define composite model for both autoencoder and trajectory classification
        composite_model = keras.Model(inputs=[inputs], outputs=[clf_out, x])
    
        # compile with selected losses and weights
        composite_model.compile(optimizer=keras.optimizers.Adam(learning_rate = self.lr), 
                                loss=['categorical_crossentropy', 'mse'], 
                                loss_weights=[self.clf_model_weight, 1 - self.clf_model_weight],
                                metrics=[['accuracy'], ['mse']])
        
        self.model = composite_model
    
        return composite_model   
    
    def train(self, Xtrain, ytrain, Xval, yval):
        
        Xtrain = np.transpose(Xtrain, (0, 2, 1)).copy()
        Xval = np.transpose(Xval, (0, 2, 1)).copy()
        
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience = 20, restore_best_weights = True,
                                                      monitor = 'val_clf_out_accuracy')

        # create learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor = 'val_clf_out_loss', factor = 0.01, 
                                                         patience = 5, verbose = 0, 
                                                         min_delta = 0.0001, mode = 'min')

        # list of callbacks to use
        trg_callbacks = [early_stopper, lr_scheduler]
        
        conv_ae_history = self.model.fit(Xtrain, [ytrain, Xtrain], 
                                  epochs = self.epochs, batch_size = self.batch_size, verbose = 1,
                                  validation_data=([Xval], 
                                                   [yval, Xval]),
                                  callbacks = trg_callbacks)
        
        return conv_ae_history
        
    def prediction(self, X, y):
        # Transform the data in a suitable format
        X = np.transpose(X, (0, 2, 1)).copy()
        
        # Initialize the plotter
        Plots = Results()
        
        # Predict under the X dataset
        preds, recons = self.model.predict(X)
        
        # Reconstruction results for saving
        recon_mses, recon_stds = Plots.get_avg_recon_results(recons, X)
        
        pred_labels = preds.argmax(axis=1)
        
        # Plot confusion matrix
        Plots.plot_confusion_matrix(y, pred_labels, figsize=(6,5),
                          title="Data Confusion Matrix")        
        
        acc = accuracy_score(pred_labels, y)
        print(f'Accuracy: {acc}')

        return preds, recons, recon_mses, recon_stds
    
    def predictionUnseen(self, X):
        # Transform the data in a suitable format
        X = np.transpose(X, (0, 2, 1)).copy()
        
        # Initialize the plotter
        Plots = Results()
        
        # Predict under the X dataset
        preds, recons = self.model.predict(X)
        
        # Reconstruction results for saving
        recon_mses, recon_stds = Plots.get_avg_recon_results(recons, X)

        print(f"Unseen (unknown intents) reconstruction average MSE: {recon_mses}")
        
        pred_labels = preds.argmax(axis=1)
        np.unique(pred_labels, return_counts=True)

        return preds, recons, recon_mses, recon_stds
        
        
    