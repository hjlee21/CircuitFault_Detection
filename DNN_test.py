import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.list_physical_devices(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
            
class DataLoader:
    def __init__(self, train_dir, val_dir, test_dir, input_cols):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.input_cols = input_cols
        
    def load_data_from_dir(self, dir_path):
        data_list = []
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dir_path, file_name)
                data = pd.read_csv(file_path)
                data_list.append(data)
        return pd.concat(data_list, ignore_index=False)
    
    def load_data(self):
        train_data = self.load_data_from_dir(self.train_dir)
        val_data = self.load_data_from_dir(self.val_dir)
        test_data = self.load_data_from_dir(self.test_dir)

        x_train = train_data[self.input_cols].values
        y_train = train_data.iloc[:,-1].values

        x_val = val_data[self.input_cols].values
        y_val = val_data.iloc[:, -1].values

        x_test = test_data[self.input_cols].values
        y_test = test_data.iloc[:, -1].values

        return x_train, y_train, x_val, y_val, x_test, y_test

class DNN_Model:
    def __init__(self, input_dim, num_classes, learning_rate = 0.001):
        self.model = self.build_model(input_dim, num_classes)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def build_model(self, input_dim, num_classes):
        model = Sequential([
            Dense(128, input_dim = input_dim, activation = 'tanh'),
            Dense(64, activation = 'tanh'),
            Dense(32, activation = 'tanh'),
            Dense(num_classes, activation = 'softmax')
        ])
        return model
    
    def train(self, x_train, y_train, x_val, y_val, epochs = 600, batch_size=128):
        history = self.model.fit(x_train, y_train,
                                 epochs = epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val))
        return history
    
    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, accuracy
    
    def save_model(self, model_path):
        self.model.save(model_path)
    
    def plot_history(self, history):
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label = 'Train Accuracy')
        plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label = 'Train Loss')
        plt.plot(history.history['val_loss'], label = 'Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.show()

train_dir = './Plan_DataAcquisition/training_dataset'
val_dir = './Plan_DataAcquisition/validation_dataset'
test_dir = './Plan_DataAcquisition/test_dataset'

input_cols = ['V2', 'V8', 'V10']

#=========================================== Load data
data_loader = DataLoader(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, input_cols=input_cols)
x_train, y_train, x_val, y_val, x_test, y_test = data_loader.load_data()

#=========================================== Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

#=========================================== Define and Compile model
input_dim = x_train.shape[1]
print(f'input_dim = {input_dim}')

num_classes = len(np.unique(y_train))
print(f'num_classes = {num_classes}')

dnn_model = DNN_Model(input_dim=input_dim, num_classes=num_classes)

#=========================================== Train and Evaluate model
history = dnn_model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
loss, accuracy = dnn_model.evaluate(x_test=x_test, y_test=y_test)
print(f'Test accuracy = {accuracy * 100:.2f} %')

#=========================================== Train and Evaluate model
serial_no = 2
model_path = f"./DNN/dnn{serial_no}"
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
dnn_model.save_model(model_path=model_path)
print(f'Model saved to {model_path}')

dnn_model.plot_history(history=history)
plt.savefig(f"./DNN/training_result/dnn{serial_no}.png")