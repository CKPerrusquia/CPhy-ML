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
from Framework.TrajectoryRegression.Linear_Regressor import Linear_Regression
from Framework.TrajectoryRegression.LSTM_Regressor import MultiInputLSTM
from Framework.TrajectoryRegression.CLSTM_Regressor import CLSTM_Regressor
from Framework.TrajectoryRegression.CNN_Regressor import CNN_Regressor

from numpy.random import seed
seed(14)
from tensorflow.random import set_seed
set_seed(14)


class MixtureOfExperts:
    
    def __init__(self, expert_base, model_params, expert_names, epochs = 150,
                 batch_size = 128, n_experts = 4):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.expert_base = expert_base
        self.n_experts = n_experts
        self.expert_names = expert_names
        self.model_params = model_params
        self.moe_training_time = []
        self.weighted_train_preds = []
        
        # initialise our expert models
        self.experts = [self.expert_base(**model_params)
                       for _ in range(n_experts)]
        

    def train(self, X_seqs, X_meta, y, trg_expert_labels,
              X_val_seqs, X_val_meta, y_val, val_expert_labels):
        
        """ Selectively train each model on subsets of total training 
            features using passed expert labels. History is saved
            for each expert. Each expert has its own preprocessor
            for the associated input data.
        """
        histories = []
        # create an early stopper callback
        
        early_stopper = keras.callbacks.EarlyStopping(patience = 15,
                                                      restore_best_weights = True,
                                                      monitor = 'val_loss')

        # create learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                         factor = 0.5,
                                                         patience = 5, verbose = 0,
                                                         min_delta = 0.001, mode = 'min')
        
        # list of callbacks to use
        trg_callbacks = [early_stopper, lr_scheduler]

        for i in range(len(self.experts)):
            
            print(f"Starting training for {self.expert_names[i]}...")
            
            # get indices correpsonding to current expert intent class
            trg_idx = trg_expert_labels == i
            val_idx = val_expert_labels == i
            
            # get selective splits of training data & labels
            trg_seqs = X_seqs[trg_idx]
            trg_meta_feats = X_meta[trg_idx]
            trg_labels = y[trg_idx]
            
            # also gather validation data for current expert
            val_seqs = X_val_seqs[val_idx]
            val_meta_feats = X_val_meta[val_idx]
            val_labels = y_val[val_idx]
                        
            # fit current expert with specified data subset(s)
            expert_history = self.experts[i].fit([trg_seqs, trg_meta_feats], trg_labels,
                                                 validation_data = ([val_seqs, val_meta_feats], val_labels),
                                                 epochs = self.epochs, batch_size = self.batch_size,
                                                 shuffle = True, callbacks = trg_callbacks)
            
            histories.append(expert_history.history)
            
            print(f"Finished training for {self.expert_names[i]}!\n")
                
        # save training histories for later extraction
        self.trg_histories = histories
        
    
    def predict_Train(self, X_seqs, X_meta, train_intent_weights):
        
        train_total_preds = []
        for expert in self.experts:
            train_preds = expert.predict([X_seqs, X_meta])
            train_total_preds.append(train_preds)
            
        train_total_preds = np.array(train_total_preds)
        
        print(f"Shape of val_total_preds: {train_total_preds.shape}")
        
        # format weights to match our preds for broadcasting
        
        train_weights = np.transpose(train_intent_weights, (1, 0))
        train_weights = np.expand_dims(train_weights, axis=-1)
        
        # compute final weighted average of preds
        weighted_train_preds = (train_weights * train_total_preds).sum(axis=0)
        return weighted_train_preds
        
               
    def predict(self, Xval_seqs, Xval_meta, val_intent_weights,
                Xtest_seqs, Xtest_meta, test_intent_weights):
        
        """ Predict using mixture of experts, with weighting for
            each expert determined by intent classifier softmax 
            output predictions. 
        """      
        # get predictions from all expert models
        val_total_preds = []
        test_total_preds = []
        
        for expert in self.experts:
            preds_val = expert.predict([Xval_seqs, Xval_meta])
            val_total_preds.append(preds_val)
        
        start_pred_time = time()
        for expert in self.experts:
            preds_test = expert.predict([Xtest_seqs, Xtest_meta])
            test_total_preds.append(preds_test)
            
        test_inference_time = time() - start_pred_time
            
        val_total_preds = np.array(val_total_preds)
        test_total_preds = np.array(test_total_preds)
        
        print(f"Shape of val_total_preds: {val_total_preds.shape}")
        print(f"Shape of test_total_preds: {test_total_preds.shape}")
        
        # format weights to match our preds for broadcasting
        val_weights = np.transpose(val_intent_weights, (1, 0))
        val_weights = np.expand_dims(val_weights, axis=-1)
        
        test_weights = np.transpose(test_intent_weights, (1, 0))
        test_weights = np.expand_dims(test_weights, axis=-1)
        
        # compute final weighted average of preds
        weighted_val_preds = (val_weights * val_total_preds).sum(axis=0)

        weighted_test_preds = (test_weights * test_total_preds).sum(axis=0)
                   
        return weighted_val_preds, weighted_test_preds
  
    def save_model(self, destination_dir):
        """ Save model to desired directory """
        # save each model to directory
        #for model in self.experts:
        pass
        
    def load_model(self, source_dir, model_names):
        """ Load pre-trained model from chosen source dir """
        # load each model from chosen dir
        #for i, model_name in enumerate(model_names):
        pass