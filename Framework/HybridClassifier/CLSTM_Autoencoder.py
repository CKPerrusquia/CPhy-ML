from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
from Framework.HybridClassifier.Attention_Layer import AttentionLayer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
                             classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class CLSTM_AE(object):
    
    def __init__(self, conv_filters = [20], 
                 lstm_units = [20, 20], activation = 'relu', 
                 bidirectional = False, dense_units = 20, 
                 dense_dropout = 0.3, kernel_size = 8, 
                 rnn_dropout = 0.3, optimizer = 'adam', 
                 lr = 1e-3, clf_model_weight = 0.90,
                 attention = False, codings_size = 16, 
                 n_classes = 4, epochs = 75, batch_size = 128):
        
        self.model = []
        self.clstm_ae_trg_time = []
        self.trg_recon_mse = []
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.activation = activation
        self.bidirectional = bidirectional
        self.dense_units = dense_units
        self.dense_dropout = dense_dropout
        self.kernel_size = kernel_size
        self.rnn_dropout = rnn_dropout
        self.optimizer = optimizer
        self.lr = lr
        self.clf_model_weight = clf_model_weight
        self.attention = attention
        self.codings_size = codings_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        
    def CLSTM_AE(self, Xtrain):
        
        """ Composite One-Dimensional CNN Autoencoder and Trajectory 
        Classifier Architecture, with dropout and multi-output optimisation.
        """
        # initial encoder layers
        inputs = keras.layers.Input(shape = Xtrain.shape[1:], name = 'Input')
        z = keras.layers.Conv1D(filters = self.conv_filters[0], kernel_size = self.kernel_size, 
                        padding = 'same', strides = 2, name = 'encoder_1',
                        activation = self.activation)(inputs)
        z = keras.layers.Dropout(self.dense_dropout, name = 'encoder_2')(z)
    
        if self.bidirectional:
            z = keras.layers.Bidirectional(
                    keras.layers.LSTM(self.lstm_units[0], return_sequences = True, 
                                      name = 'encoder_3'))(z)
            z = keras.layers.Bidirectional(
                    keras.layers.LSTM(self.lstm_units[1], return_sequences = True,
                                  name = 'encoder_4', dropout = self.rnn_dropout))(z)
        else:
            z = keras.layers.LSTM(self.lstm_units[0], return_sequences = True,
                              name = 'encoder_3')(z)
            z = keras.layers.LSTM(self.lstm_units[1], return_sequences = True,
                              name = 'encoder_4', dropout = self.rnn_dropout)(z)
    
        # define lower-dimensional latent space for encoder
        codings = keras.layers.LSTM(self.codings_size, return_sequences = False,
                                    name = 'latent_codings')(z)
        encoder = keras.models.Model(inputs = [inputs], outputs = [codings])
    
        # final dense layers for classification (apply attention if selected)
        if self.attention:
            Attention = AttentionLayer(name = 'clf_1')
            clf_out = Attention(z)
        else:
            clf_out = keras.layers.Flatten(name = 'clf_1')(z)
        
        clf_out = keras.layers.Dropout(self.dense_dropout, name = 'clf_2')(clf_out)
        clf_out = keras.layers.Dense(self.dense_units, name = 'clf_3',
                                     activation = self.activation)(clf_out)
        clf_out = keras.layers.Dense(self.n_classes, name = 'clf_out',
                                     activation = 'softmax')(clf_out)
    
    
        # define decoder component (simplify to only using RNNs)
        x = keras.layers.RepeatVector(Xtrain.shape[1], name = 'decoder_1',
                                      input_shape = [self.codings_size])(codings)
        x = keras.layers.LSTM(self.lstm_units[1], name = 'decoder_2',
                              return_sequences = True)(x)
        x = keras.layers.LSTM(self.lstm_units[0], name = 'decoder_3',
                              return_sequences = True)(x)
        decoder_output = keras.layers.TimeDistributed(
                            keras.layers.Dense(Xtrain.shape[2], 
                                    name = 'dense_out', activation = None), 
                            name = 'decoder_output')(x)
    
        # define composite model for both autoencoder and trajectory classification
        composite_model = keras.Model(inputs = [inputs], outputs = [clf_out, decoder_output])
    
        # compile with selected losses and weights
        composite_model.compile(optimizer=keras.optimizers.Adam(learning_rate = self.lr), 
                                loss = ['categorical_crossentropy', 'mse'], 
                                loss_weights = [self.clf_model_weight, 1 - self.clf_model_weight],
                                metrics=[['accuracy'], ['mse']])
        self.model = composite_model
    
        return composite_model 
    
    def train(self, Xtrain, ytrain, Xval, yval, ylabel, WINDOW_SIZE):
                
        # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience = 25, 
                                              restore_best_weights = True,
                                              monitor = 'val_clf_out_accuracy')

        # create learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor = 'val_clf_out_loss', 
                                                 factor = 0.75, patience = 14, 
                                                 verbose = 0, min_delta = 0.001, 
                                                 mode = 'min')

        # list of callbacks to use
        trg_callbacks = [early_stopper, lr_scheduler]
        
        clstm_ae_start_time = time()
        clstm_ae_history = self.model.fit(Xtrain, [ytrain, Xtrain], 
                                  epochs = self.epochs, batch_size = self.batch_size, verbose = 1,
                                  validation_data=([Xval], 
                                                   [yval, Xval]),
                                  callbacks = trg_callbacks)
        
        # compute total training time
        clstm_ae_trg_time = time() - clstm_ae_start_time
        self.clstm_ae_trg_time = clstm_ae_trg_time
        
        trg_preds, trg_recons = self.model.predict(Xtrain)
        trg_pred_labels = trg_preds.argmax(axis=1)
                
        print(f"ConvLSTM AE Clf (window size: {WINDOW_SIZE}) Training Time: {clstm_ae_trg_time}")
        clstm_ae_df = pd.DataFrame(clstm_ae_history.history)

        fig, ax = plt.subplots(2,2, figsize=(14,10))
        ax = ax.flatten()

        clstm_ae_df[['clf_out_accuracy', 'val_clf_out_accuracy']].plot(ax=ax[0], 
                             color=['tab:blue', 'tab:orange'])
        ax[0].grid(0.5)
        ax[0].set_title("Classification Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend(['Training', 'Validation'])
        ax[0].set_ylim([0.85, 0.99])

        clstm_ae_df[['clf_out_loss', 'val_clf_out_loss']].plot(ax=ax[1],
                                color=['tab:blue', 'tab:orange'])
        ax[1].grid(0.5)
        ax[1].set_title("Classification Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss (Categorical Cross-Entropy)")
        ax[1].legend(['Training', 'Validation'])

        clstm_ae_df[['decoder_output_loss', 
                    'val_decoder_output_loss']].plot(ax=ax[2], 
                color=['tab:blue', 'tab:orange'], label=['Training', 'Validation'])
        ax[2].grid(0.5)
        ax[2].set_title("Autoencoder Loss")
        ax[2].set_xlabel("Epochs")
        ax[2].set_ylabel("Loss (MSE)")
        ax[2].legend(['Training', 'Validation'])

        clstm_ae_df['val_decoder_output_mse'].plot(ax=ax[3], 
                color=['tab:orange'], label='Validation')
        ax[3].grid(0.5)
        ax[3].set_title("Autoencoder Validation Loss")
        ax[3].set_xlabel("Epochs")
        ax[3].set_ylabel("Loss (MSE)")
        plt.tight_layout()
        plt.show()
        
        Plots = Results()
        print(f"{'-'*20} Training Data {'-'*20}")
        print(classification_report(ylabel, trg_pred_labels, digits=4))
        
        trg_recon_mse = Plots.get_avg_mse(trg_recons , Xtrain)
        self.trg_recon_mse = trg_recon_mse
        
        print(f"Training set reconstruction average MSE: {trg_recon_mse}")
        
        return clstm_ae_history
        
    def prediction(self, Xval, yval, Xtest, ytest, WINDOW_SIZE):
        # Initialize the Plotter
        Plots = Results()
        
        # Predict under the Xval and Xtest datasets
        val_preds, val_recons = self.model.predict(Xval)
        start_pred_time = time()
        test_preds, test_recons = self.model.predict(Xtest)
        
        # Compute the inference time of the predictions and the average time
        test_inference_time = time()-start_pred_time
        avg_pred_time = test_inference_time / Xtest.shape[0]
        
        print(f"Test Inference Time: {test_inference_time}")
        print(f"Individual Pred Time: {avg_pred_time}")
        
        # Show the reconstruction error results under the validation and testing datasets
        val_recon_mses  = Plots.get_avg_mse(val_recons, Xval)
        test_recon_mses = Plots.get_avg_mse(test_recons, Xtest)
        
        val_pred_labels = val_preds.argmax(axis=1)
        test_pred_labels = test_preds.argmax(axis=1)
        
        print(f"{'-'*20} Validation Data {'-'*20}")
        print(classification_report(yval, val_pred_labels, digits=4))
        print(f"\n{'-'*20} Test Data {'-'*20}")
        print(classification_report(ytest, test_pred_labels, digits=4))
        
        # Plot confusion matrices for validation and testing datasets
        Plots.plot_confusion_matrix(yval, val_pred_labels, figsize=(6,5),
                          title="Validation Data Confusion Matrix") 
        
        Plots.plot_confusion_matrix(ytest, test_pred_labels, figsize=(6,5),
                          title="Test Data Confusion Matrix")    
        
        acc_val = accuracy_score(val_pred_labels, yval)
        acc_test = accuracy_score(test_pred_labels, ytest)
        
        print(f'Validation Accuracy: {acc_val}')
        print(f'Test Accuracy: {acc_test}')
        
        # Save the results in an adequate format
        model_results = Plots.get_model_results(yval, val_pred_labels, ytest,
                                               test_pred_labels, "Hybrid AE Classifier")
        model_results['Training Time'] = self.clstm_ae_trg_time
        model_results['Avg Pred Time'] = avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        val_recon_mse = Plots.get_avg_mse(val_recons , Xval)
        print(f"Validation set reconstruction average MSE: {val_recon_mse}")

        test_recon_mse = Plots.get_avg_mse(test_recons , Xtest)
        print(f"Test set reconstruction average MSE: {test_recon_mse}")
        
        trg_recon_mse = self.trg_recon_mse
       
        return model_results, val_preds, val_recons, test_preds, test_recons, trg_recon_mse, val_recon_mse, test_recon_mse
    
    def predictionUnseen(self, X):
        """ Prediction under unseen trajectories """
        
        Plots = Results()
        preds, recons = self.model.predict(X)
        
        recon_mse = Plots.get_avg_mse(recons, X)

        print(f"Unseen (unknown intents) reconstruction average MSE: {recon_mse}")
        print(f"Unseen MSE factor relative to training: {recon_mse/self.trg_recon_mse}")     
        return preds, recons, recon_mse
    
    def save(self, model_savepath):
        """ Save the final model"""
        self.model.save(model_savepath)
        print("Model Saved!")
        return