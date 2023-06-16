import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import time
import numpy as np

import keras.backend as K
from tensorflow.random import set_seed
set_seed(12)

class AttentionLayer(keras.layers.Layer):
    """ Custom attention layer """
    def __init__(self,**kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # alignment scores to pass through tanh activation
        e = K.tanh(K.dot(x,self.W)+self.b)
        # squeeze our output tensor (remove dim 1)
        e = K.squeeze(e, axis=-1)   
        # compute attention weights and expand dims again
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        # compute attention context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context