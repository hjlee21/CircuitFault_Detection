import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense

logit = tf.random.uniform(shape=(8,5), minval=-10, maxval=10)
dense = Dense(units=8, activation='softmax')
Y = dense(logit)
print(tf.reduce_sum(Y, axis=1))


import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model #subclass를 활용해보자

class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.dense1 = Dense(units=8, activation='relu')
        self.dense2 = Dense(units=5, activation='relu')
        self.dense3 = Dense(units=3, activation='softmax')
    
    def call(self, x):
        
        print("=========================================== x")
        print(f"X_Shape:{x.shape}")
        print(f"X      :{x.numpy()}")
        
        x= self.dense1(x)
        print("=========================================== after layer 1")
        print(f"X_Shape:{x.shape}")
        print(f"X_D1   :{x.numpy()}")
        
        x= self.dense2(x)
        print("=========================================== after layer 2")
        print(f"X_Shape:{x.shape}")
        print(f"X_D2   :{x.numpy()}")
        
        x= self.dense3(x)
        print("=========================================== after layer 3")
        print(f"X_Shape:{x.shape}")
        print(f"X_D3   :{x.numpy()}")
        print(f"Sum of Vectors  : \n{tf.reduce_sum(x, axis=1)}")
        
        return x
    
model = TestModel()
X = tf.random.uniform(shape=(8, 3), minval=-10, maxval=10)
Y = model(X)
