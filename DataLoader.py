import os
import pandas as pd
import numpy as np

class Data_Loader:
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
    

if __name__ == "__main__":
    train_dir = './Plan_DataAcquisition/training_dataset'
    val_dir = './Plan_DataAcquisition/validation_dataset'
    test_dir = './Plan_DataAcquisition/test_dataset'

    input_cols = ['V2', 'V8', 'V10']

    data_loader = Data_Loader(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, input_cols=input_cols)
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.load_data()
