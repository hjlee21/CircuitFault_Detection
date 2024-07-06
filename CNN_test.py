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

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimeSeriesCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, hidden_dim)  # Adjust input size according to the final feature map size
        # setup output layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        check_gpu_usage()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc(x)
        
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


        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_db:
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                inputs = inputs.permute(0, 2, 1)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                labels = labels.squeeze(dim=0) if labels.size(0) ==1 else labels.squeeze()
                loss = criterion(outputs, labels)
                
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
                inputs = inputs.permute(0, 2, 1)
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
batch_size = 10
sequence_length=sequence_length = 10

train_db = create_data_loader('./Data/data_training', input_cols, sequence_length=sequence_length, batch_size=batch_size)
test_db = create_data_loader('./Data/data_test', input_cols, sequence_length=sequence_length, batch_size=batch_size)
#validation_db = create_data_loader('./Data/data_validation', input_cols, sequence_length=sequence_length, batch_size=batch_size)

# 데이터 로더에서 하나의 배치를 가져와서 테스트해봄
# data_iter = iter(train_db)
# sample_data, sample_label = next(data_iter)
# sample_data  : torch.size([2, 10, 3])
# sample_label : torch.size([2, 1])

num_classes = 4  # 라벨 수
hidden_dim = 64
output_dim = num_classes
model = TimeSeriesCNN(len(input_cols), hidden_dim, output_dim)

# check_gpu_usage()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

trainer = ModelTrainer(model=model, train_db=train_db, val_db=test_db)
trainer.train_model(num_epochs=50, learning_rate=0.001)

serial_no = 1
model_path = f'./CNN/CNN{serial_no}'
if not os.path.exists(model_path):
    os.makedirs(model_path)
trainer.save_model(model_path=f'{model_path}/model.pth')

plot_path = f'./CNN/training_result/CNN{serial_no}.png'
if not os.path.exists(os.path.dirname(plot_path)):
    os.makedirs(os.path.dirname(plot_path))
trainer.plot_history(save_path=plot_path)