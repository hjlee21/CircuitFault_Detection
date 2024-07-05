import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_folder, input_cols, sequence_length, minmaxscaler=None):
        self.call_generate_minmax = False if minmaxscaler != None else True
        
        if minmaxscaler == None:
            print('Gen MinMax')
            self.minmax = MinMaxScaler()
        else:
            print('Use MinMax')
            self.minmax = minmaxscaler
            
        self.data, self.labels = self.load_csv_files(data_folder, input_cols, sequence_length)
        print(self.data.shape)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def load_csv_files(self, data_folder, input_cols, sequence_length):
        data = []
        labels = []
        if self.call_generate_minmax:
            for filename in os.listdir(data_folder):
                if filename.endswith('.csv'):        
                    file_path = os.path.join(data_folder, filename)
                    df = pd.read_csv(file_path)
                    self.minmax.partial_fit(df[input_cols].values)
            print('Done MinMax Scaler')
        
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_folder, filename)
                df = pd.read_csv(file_path)
                label = self.get_label_from_filename(filename)
                df_minmax = self.minmax.transform(df[input_cols].values)
                for i in range(len(df) - sequence_length):
                    data.append(df_minmax[i:i+sequence_length])
                    labels.append([label])
                print(f'df len {len(df)} | total data len {len(data)} | seq {sequence_length}') # Test line
        return np.array(data), np.array(labels)
    
    def get_label_from_filename(self, filename):
        if 'FAMP' in filename:   return 0  # 고장 라벨 0
        elif 'FLPF' in filename: return 1  
        elif 'FSEN' in filename: return 2
        elif 'NF' in filename:   return 3
        else: raise ValueError(f"Unknown label for file: {filename}")
        
def create_data_loader(data_folder, input_cols, sequence_length=10, batch_size=2, shuffle=True, MinMaxScaler=None):
    dataset = CustomDataset(data_folder, input_cols, sequence_length, MinMaxScaler)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, collate_fn=collate_fn)
    return data_loader

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, 64)  # Adjust input size according to the final feature map size
        self.fc2 = nn.Linear(64, num_classes)
    
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
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_losses = []
    val_losses = []
    val_accuracies = []
    with open('training_validation_results.txt', 'w') as f:
        for epoch in range(num_epochs):
            model.train()  # 모델을 훈련 모드로 설정
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.permute(0, 2, 1), labels.squeeze()  # (batch, input_dim, sequence), 라벨 차원 축소
                optimizer.zero_grad()  # 옵티마이저 초기화
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass 및 최적화
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
            
            # 검증 성능 평가
            model.eval()  # 모델을 평가 모드로 설정
            val_running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.permute(0, 2, 1), val_labels.squeeze()  # (batch, input_dim, sequence), 라벨 차원 축소
                    
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
                    
                    val_running_loss += val_loss.item()
                    
                    _, predicted = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()
            
            val_epoch_loss = val_running_loss / len(val_loader)
            val_accuracy = correct / total
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_accuracy)
            print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}')
            
            # 각 epoch의 결과를 파일에 저장
            f.write(f'Epoch,{epoch}/{num_epochs - 1},Training Loss,{epoch_loss:.4f},Validation Loss,{val_epoch_loss:.4f},Validation Accuracy,{val_accuracy:.4f}\n')
        
    # 시각화
    epochs = range(num_epochs)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('CNN_training_validation_results.svg')  # 그래프 이미지 저장
    plt.show()
    
    print('Training complete')
    return model

# ===========================================================================================
# Run
# ===========================================================================================
input_cols = ['V2', 'V8', 'V10']

train_db = create_data_loader('./Plan_DataAcquisition/training_dataset', input_cols, 10)
test_db = create_data_loader('./Plan_DataAcquisition/test_dataset', input_cols, 10, MinMaxScaler=train_db.dataset.minmax)
validation_db = create_data_loader('./Plan_DataAcquisition/validation_dataset', input_cols, 10, MinMaxScaler=train_db.dataset.minmax)

# 데이터 로더에서 하나의 배치를 가져옴
data_iter = iter(train_db)
sample_data, sample_label = next(data_iter)

print(f'Sample Data:\n{sample_data}\n{sample_data.size()}')
print(f'Sample Label: {sample_label}\t{sample_label.size()}')

num_classes = 4  # 라벨 수

model = TimeSeriesCNN(len(input_cols), num_classes)
output = model(sample_data.permute(0, 2, 1))
print(f'Output shape: {output.shape}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
trained_model = train_model(model, train_db, validation_db, criterion, optimizer, num_epochs)