from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np
from sklearn.metrics import classification_report

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class Transformer_Network(object):
    
    def __init__(self, head_size = 128, n_heads = 4, ff_dim = 4, 
                 n_transformer_blocks = 4, dense_layers = [64],
                 ff_dropout = 0.25, dense_dropout = 0.4, dropout = 0,
                 learn_rate=1e-3, n_classes=4, epochs = 80, 
                batch_size = 256):
        
        self.model = []
        self.trans_trg_time = []
        self.head_size = head_size
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_transformer_blocks = n_transformer_blocks
        self.dense_layers = dense_layers
        self.ff_dropout = ff_dropout
        self.dropout = dropout
        self.dense_dropout = dense_dropout
        self.learn_rate = learn_rate
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        
    def transformer_encoder(self, inputs):
        """ Transformer attention encoder, with layer normalisation """
        x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = keras.layers.MultiHeadAttention(
            key_dim = self.head_size, num_heads = self.n_heads, 
            dropout = self.dropout)(x, x)
        x = keras.layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = keras.layers.Conv1D(filters = self.ff_dim, kernel_size = 1, activation = "relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Conv1D(filters = inputs.shape[-1], kernel_size = 1)(x)
        return x + res

    def transformer_dnn_model(self, Xtrain):
        """ Transformer DNN model for classification """
        inputs = keras.Input(shape = Xtrain.shape[1:])
        x = inputs
        for _ in range(self.n_transformer_blocks):
            x = self.transformer_encoder(x)

        x = keras.layers.GlobalAveragePooling1D(data_format = "channels_first")(x)
        for dim in self.dense_layers:
            x = keras.layers.Dense(dim, activation = "relu")(x)
            x = keras.layers.Dropout(self.dense_dropout)(x)
        outputs = keras.layers.Dense(self.n_classes, activation = "softmax")(x)
    
        model = keras.Model(inputs, outputs)
        model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=["accuracy"])
        self.model = model
        
        return model
    
    def train(self, Xtrain, ytrain, Xval, yval):
        
                # create an early stopper callback
        early_stopper = keras.callbacks.EarlyStopping(patience = 15, 
                                                      restore_best_weights = True,
                                                      monitor = 'val_accuracy')
      
        # use only our early stopper callback
        trg_callbacks = [early_stopper]
        
        trans_start_time = time()

        trans_history = self.model.fit(Xtrain, ytrain,
                                          epochs = self.epochs, batch_size = self.batch_size,
                                          validation_data = (Xval, yval),
                                          callbacks = trg_callbacks)

        trans_trg_time = time() - trans_start_time
        self.trans_trg_time = trans_trg_time
        
        print(f"Transformer training time: {trans_trg_time}")
        
        return trans_history
    
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
        
        # Inference and average time of the predictions
        trans_test_inference_time = time() - start_pred_time
        trans_avg_pred_time = trans_test_inference_time / Xtest.shape[0]

        print(f"Test Inference Time: {trans_test_inference_time}")
        print(f"Individual Pred time: {trans_avg_pred_time}")
        
        # get model classification results in final format for saving
        model_results = Plots.get_model_results(yval, val_pred_labels, ytest, 
                                                test_pred_labels, "Transformer")
        model_results['Training Time'] = self.trans_trg_time
        model_results['Avg Pred Time'] = trans_avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        print(f"{'-'*20} Validation Data {'-'*20}")
        print(classification_report(yval, val_pred_labels, digits=4))
        print(f"\n{'-'*20} Test Data {'-'*20}")
        print(classification_report(ytest, test_pred_labels, digits=4))
        
        # Plot confusion matrices under the validation and testing datasets
        Plots.plot_confusion_matrix(yval, val_pred_labels, 
                      title="Validation Data Confusion Matrix")
        
        Plots.plot_confusion_matrix(ytest, test_pred_labels, 
                      title="Test Data Confusion Matrix")
        
        return model_results