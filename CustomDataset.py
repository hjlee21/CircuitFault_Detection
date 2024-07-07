import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_folder, input_cols, sequence_length):
        
        self.data, self.labels = self.load_csv_files(data_folder, input_cols, sequence_length)
        print(f'data.shape : {self.data.shape}      labels.shape : {self.labels.shape}') 
        #(total data length - sequence length, sequence_length, 1)
        #(total data length - sequence length, 1)

        # print(f'data: {self.data}')
        # print(f'label: {self.labels}')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def load_csv_files(self, data_folder, input_cols, sequence_length):
        data = []
        labels = []
       
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_folder, filename)
                df = pd.read_csv(file_path)
                label = self.get_label_from_filename(filename)
                df_cols = df[input_cols].values
                for i in range(len(df) - sequence_length):
                    data.append(df_cols[i:i+sequence_length])
                    labels.append([label])
                # print(f'df len {len(df)} | total data len {len(data)} | seq {sequence_length}') # Test line
        return np.array(data), np.array(labels)
    
    def get_label_from_filename(self, filename):
        if 'NORM' in filename:   return 0  # 고장 라벨 0
        elif 'FSEN' in filename: return 1  
        elif 'FAMP' in filename: return 2  
        elif 'FLPF' in filename: return 3
        else: raise ValueError(f"Unknown label for file: {filename}")

def create_data_loader(data_folder, input_cols, sequence_length=10, batch_size=2, shuffle=True):
    dataset = CustomDataset(data_folder, input_cols, sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, collate_fn=collate_fn)
    return data_loader
     
class ValidationData:
    def __init__(self, data_folder, input_cols, sequence_length) -> None:
        self.data_folder = data_folder
        self.input_cols = input_cols
        self.sequence_length = sequence_length
        
        self.se_data = {}
        
        for filename in os.listdir(data_folder):
            data = []
            labels = []
            if filename.endswith('.csv'):
                file_path = os.path.join(data_folder, filename)
                df = pd.read_csv(file_path)
                label = self.get_label_from_filename(filename)
                df_cols = df[input_cols].values
                for i in range(len(df) - sequence_length):
                    data.append(df_cols[i:i+sequence_length])
                    labels.append([label])
                    
                self.se_data[file_path] = {'data': torch.tensor(np.array(data), dtype=torch.float32), 
                                           'labels': torch.tensor(np.array(labels), dtype=torch.long)}
            
    def get_label_from_filename(self, filename):
        if 'NORM' in filename:   return 0  # 고장 라벨 0
        elif 'FSEN' in filename: return 1  
        elif 'FAMP' in filename: return 2  
        elif 'FLPF' in filename: return 3
        else: raise ValueError(f"Unknown label for file: {filename}")
        
    def keys(self):
        return list(self.se_data.keys())
    
    def get_se(self, se_file_name):
        return self.se_data[se_file_name]['data'], self.se_data[se_file_name]['labels'] 