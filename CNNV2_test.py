import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from CustomDataset import CustomDataset, ValidationData
from torch.utils.data import Dataset, DataLoader

def create_data_loader(data_folder, input_cols, sequence_length=10, batch_size=10, shuffle=True):
    dataset = CustomDataset(data_folder, input_cols, sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, collate_fn=collate_fn)
    return data_loader

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, serial_no, sequence_length, dropout_prob=0.5):
        super(TimeSeriesCNN, self).__init__()
        self.serial_no = serial_no
        hidden_dim = {10:64, 20:160, 30:224, 40:320, 50:384}[sequence_length]
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout layer 추가
        self.fc1 = nn.Linear(hidden_dim, 64)  # Adjust input size according to the final feature map size
        # setup output layer
        self.fc = nn.Linear(64, output_dim)
        
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
        x = self.dropout(x)  # Dropout 적용
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc(x)
        
        return out

class ModelTrainer:
    def __init__(self, model, train_db, val_db=None, save_path=None, serial_no=1, 
                 batch_size=10, sequence_length=10):
        self.model = model
        self.train_db = train_db
        self.val_db = val_db
        self.save_path = save_path
        self.serial_no = serial_no
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def train_model(self, num_epochs = 50, learning_rate = 0.001):
        # Model Save
        model_path = f'{self.save_path}/CNNV2{self.serial_no}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Result PNG
        plot_path = f'{self.save_path}/training_result/CNNV2{self.serial_no}.png'
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
        
        
        self.log_training_info(self.serial_no, self.batch_size, self.sequence_length, num_epochs, learning_rate)
        self.log_training_history(new=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            start_time = time.time()  # 에포크 시작 시간 기록
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
                val_loss, val_acc = self.evaluate_model(epoch)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

            Epoch = f'[{epoch+1} / {num_epochs}]'
            epoch_time = time.time() - start_time  # 에포크 소요 시간 계산
            
            print(f'Epoch {Epoch},   Loss: {epoch_loss:.4f},    Accuracy: {epoch_acc:.4f},    Time: {epoch_time:.2f} seconds')

            if self.val_db:
                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
                self.log_training_history(Epoch=Epoch, epoch_loss=epoch_loss, epoch_acc=epoch_acc, val_loss=val_loss, val_acc=val_acc)
            else:
                self.log_training_history(Epoch=Epoch, epoch_loss=epoch_loss, epoch_acc=epoch_acc)

        # Model Save
        trainer.save_model(model_path=f'{model_path}/model.pth')
        # Result PNG
        trainer.plot_history(save_path=plot_path)

        print('Training Complete')
    
    def log_training_history(self, new=False, Epoch=0, epoch_loss=0, epoch_acc=0, val_loss=0, val_acc=0):
        if new:
            with open(f'{self.save_path}/training_result/CNNV2{self.serial_no}_training_history.txt', 'w') as f:
                if self.val_db:
                    f.write(f'Epoch,epoch_loss,epoch_acc,val_loss,val_acc\n')
                else:
                    f.write(f'Epoch,epoch_loss,epoch_acc\n')
        else:
            with open(f'{self.save_path}/training_result/CNNV2{self.serial_no}_training_history.txt', 'a') as f:
                if self.val_db:
                    f.write(f'{Epoch},{epoch_loss},{epoch_acc},{val_loss},{val_acc}\n')
                else:
                    f.write(f'{Epoch},{epoch_loss},{epoch_acc}\n')

    def log_training_info(self, serial_no, batch_size, sequence_length, num_epochs, learning_rate):
        with open(f'{self.save_path}/training_result/CNNV2{self.serial_no}_training_info.txt', 'w') as f:
            f.write(f'serial_no:{serial_no}\n')
            f.write(f'batch_size:{batch_size}\n')
            f.write(f'sequence_length:{sequence_length}\n')
            f.write(f'num_epochs:{num_epochs}\n')
            f.write(f'learning_rate:{learning_rate}\n')
    
    def log_validation_history(self, Epoch=0, se_file_name='', acc_result=[]):
        with open(f'{self.save_path}/training_result/CNNV2{self.serial_no}_validation_result.txt', 'a') as f:
            f.write(f'Epoch,{Epoch},se_file_name,{se_file_name},{acc_result}\n')
    
    def evaluate_model(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        total_data_points = 0
        with torch.no_grad():
            for se_file_name in self.val_db.keys():
                data_, label_ = test_db.get_se(se_file_name)
                inputs, labels = data_.to(self.model.device), label_.to(self.model.device)
                inputs = inputs.permute(0, 2, 1)
                outputs = self.model(inputs)
                # labels = labels.squeeze()
                labels = labels.squeeze(dim=0) if labels.size(0) == 1 else labels.squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_data_points += len(labels)
                self.log_validation_history(epoch, se_file_name, predicted.tolist())

        #val_loss /= len(self.val_db)
        val_loss /= total_data_points
        val_acc = correct / total
        return val_loss, val_acc
    
    def plot_history(self, save_path):
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

        # plt.show()
    
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
serial_no = 1
input_cols = ['V2', 'V8', 'V10']

for serial_no in range(1, 2):
    batch_size = {1:10, 2:10, 3:10, 4:10, 5:10, 6:20, 7:20, 8:20, 9:20, 10:20}[serial_no]
    sequence_length = {1: 10, 2: 20, 3:30, 4:40, 5:50, 6: 10, 7: 20, 8:30, 9:40, 10:50}[serial_no]

    print("Loading 'train_db'--------------------------------------------------------------------")
    train_db = create_data_loader('./Data_2/data_training', input_cols, sequence_length=sequence_length, batch_size=batch_size)
    print("Loading 'test_db'---------------------------------------------------------------------")
    test_db = ValidationData('./Data_2/data_test', input_cols, sequence_length=sequence_length)

    output_dim = 4
    hidden_dim = 64
    model = TimeSeriesCNN(len(input_cols), hidden_dim, output_dim, serial_no, sequence_length)
    
    # data_, label_ = test_db.get_se(test_db.keys()[0])
    # inputs, labels = data_.to(model.device), label_.to(model.device)
    # inputs = inputs.permute(0, 2, 1)
    # out = model(inputs)

    trainer = ModelTrainer(model=model, train_db=train_db, val_db=test_db, save_path=f'./CNNV2', 
                        serial_no=serial_no, batch_size=batch_size, sequence_length=sequence_length)
    trainer.train_model(num_epochs=100, learning_rate=0.001)
    
    del model
    torch.cuda.empty_cache()
    
