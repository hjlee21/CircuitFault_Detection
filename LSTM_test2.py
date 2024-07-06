import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from CustomDataset import CustomDataset
from torch.utils.data import Dataset, DataLoader

def create_data_loader(data_folder, input_cols, sequence_length=10, batch_size=10, shuffle=True):
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
        check_gpu_usage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

class ModelTrainer:
    def __init__(self, model, train_db, val_db=None):
        self.model = model
        self.train_db = train_db
        self.val_db = val_db
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def train_model(self, num_epochs = 50, learning_rate = 0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_db:
                # inputs.size() -> torch.Size([2, 10, 3])
                # labels.size() -> torch.Size([2, 1])
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                # outputs.size() -> torch.Size([2, 4])
                #labels.squeeze() -> torch.Size([2])
                labels = labels.squeeze(dim=0) if labels.size(0) ==1 else labels.squeeze()
                loss = criterion(outputs, labels)
                # print(loss)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_db)
            epoch_acc = correct/total
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc)

            if self.val_db:
                val_loss, val_acc = self.evaluate_model()
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

            print(f'Epoch [{epoch+1} / {num_epochs}],   Loss: {epoch_loss:.4f},    Accuracy: {epoch_acc:.4f}')

            if self.val_db:
                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        print('Training Complete')
    
    def evaluate_model(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in self.val_db:
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                outputs = self.model(inputs)
                # labels = labels.squeeze()
                labels = labels.squeeze(dim=0) if labels.size(0) == 1 else labels.squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_db)
        val_acc = correct / total
        self.model.train()
        return val_loss, val_acc
    
    def plot_history(self, save_path=None):
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(self.history['accuracy'], label = 'Train Accuracy')
        if 'val_accuracy' in self.history:
            plt.plot(self.history['val_accuracy'], label = 'Validation Accuracy')


        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.subplot(1,2,2)
        plt.plot(self.history['loss'], label = 'Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label = 'Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        if save_path:
            plt.savefig(save_path)

        plt.show()
    
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

def check_gpu_usage():
    if torch.cuda.is_available():
        pass
    else:
        print("GPU is not available.")

# ===========================================================================================
# Run
# ===========================================================================================
input_cols = ['V2', 'V8', 'V10']
print("Loading 'train_db'--------------------------------------------------------------------")
train_db = create_data_loader('./Data/data_training', input_cols, sequence_length=10, batch_size=10)
print("Loading 'test_db'---------------------------------------------------------------------")
test_db = create_data_loader('./Data/data_test', input_cols, sequence_length=10, batch_size=10)
# print("Loading 'validation_db'---------------------------------------------------------------")
# validation_db = create_data_loader('./Data/data_validation', input_cols, sequence_length=10, batch_size=10)

# 데이터 로더에서 하나의 배치를 가져와서 테스트해봄
# data_iter = iter(train_db)
# sample_data, sample_label = next(data_iter)
# sample_data  : torch.size([2, 10, 3])
# sample_label : torch.size([2, 1])

num_classes = 4
input_dim = 3  # Number of features in the input sequence
hidden_dim = 128
output_dim = num_classes
num_layers = 1
model = LSTM_Model(input_dim, hidden_dim, output_dim, num_layers)

check_gpu_usage()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = ModelTrainer(model=model, train_db=train_db, val_db=test_db)
trainer.train_model(num_epochs=100, learning_rate=0.001)

serial_no = 1
model_path = f'./LSTM/lstm{serial_no}'
if not os.path.exists(model_path):
    os.makedirs(model_path)
trainer.save_model(model_path=f'{model_path}/model.pth')

plot_path = f'./LSTM/training_result/LSTM{serial_no}.png'
if not os.path.exists(os.path.dirname(plot_path)):
    os.makedirs(os.path.dirname(plot_path))
trainer.plot_history(save_path=plot_path)
