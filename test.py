import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader

def create_data_loader(data_folder, input_cols, sequence_length=10, batch_size=2, shuffle=True):
    dataset = CustomDataset(data_folder, input_cols, sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, collate_fn=collate_fn)
    return data_loader

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first = True)
        # batch_first = True -> (batch_size, sequence_length, input_dim)
        # batch_first = False -> (sequence_length, batch_size, input_dim)

        # setup output layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) #Initialize hidden h0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) #Initialize cell c0
        # h0.size: torch.Size([3, 2, 128]),  c0.size: torch.Size([3, 2, 128])
        
        # LSTM에 입력데이터 전달하고, 출력 및 새로운 은닉상태 받기
        out, _ = self.lstm(x, (h0, c0))
        # out -> torch.Size([2, 10, 128])
        # _[0] -> torch.Size([3, 2, 128])
        # _[1] -> torch.Size([3, 2, 128])

        # out[:, -1, :] -> [2, 128]
        out = self.fc(out[:, -1, :]) #sequence length의 최종끝단
        # out -> [2, 4]
        return out


model.train()
