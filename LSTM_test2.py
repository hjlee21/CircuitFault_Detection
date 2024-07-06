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
        self.serial_no = serial_no

        # setup LSTM layer
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first = True)
        # batch_first = True -> (batch_size, sequence_length, input_dim)
        # batch_first = False -> (sequence_length, batch_size, input_dim)

        # setup output layer
        
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        
        
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
        model_path = f'{self.save_path}/LSTM{self.serial_no}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # Result PNG
        plot_path = f'{self.save_path}/training_result/LSTM{self.serial_no}.png'
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
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
                
            Epoch = f'[{epoch+1} / {num_epochs}]'
            
            print(f'Epoch {Epoch},   Loss: {epoch_loss:.4f},    Accuracy: {epoch_acc:.4f}')

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
            with open(f'{self.save_path}/training_result/LSTM{self.serial_no}_training_history.txt', 'w') as f:
                if self.val_db:
                    f.write(f'Epoch,epoch_loss,epoch_acc,val_loss,val_acc\n')
                else:
                    f.write(f'Epoch,epoch_loss,epoch_acc\n')
        else:
            with open(f'{self.save_path}/training_result/LSTM{self.serial_no}_training_history.txt', 'a') as f:
                if self.val_db:
                    f.write(f'{Epoch},{epoch_loss},{epoch_acc},{val_loss},{val_acc}\n')
                else:
                    f.write(f'{Epoch},{epoch_loss},{epoch_acc}\n')

    def log_training_info(self, serial_no, batch_size, sequence_length, num_epochs, learning_rate):
        with open(f'{self.save_path}/training_result/LSTM{self.serial_no}_training_info.txt', 'w') as f:
            f.write(f'serial_no:{serial_no}\n')
            f.write(f'batch_size:{batch_size}\n')
            f.write(f'sequence_length:{sequence_length}\n')
            f.write(f'num_epochs:{num_epochs}\n')
            f.write(f'learning_rate:{learning_rate}\n')
            
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
input_cols = ['V2', 'V8', 'V10']

output_dim = 4
input_dim = 3  # Number of features in the input sequence
hidden_dim = 128

for serial_no in range(13, 14):
    batch_size = {1: 20, 2:20, 3:20, 
                  4: 40, 5: 40, 6: 40, 
                  7: 60, 8: 60, 9: 60,
                  10:20, 11:20, 12:20,
                  13:50}[serial_no]
    sequence_length = {1: 10, 2: 20, 3:30, 
                       4: 10, 5: 20, 6:30, 
                       7: 10, 8: 20, 9:30, 
                       10: 10, 11: 20, 12:30,
                       13: 5}[serial_no]
    num_layers = {1: 1, 2:1, 3:1, 
                  4:1, 5:1, 6:1, 
                  7: 1, 8:1, 9:1, 
                  10:2, 11:2, 12:2,
                  13: 1}[serial_no]
    
    print("--------------------------------------------------------------------Loading 'train_db'")
    train_db = create_data_loader('./Data/data_training', input_cols, sequence_length=sequence_length, batch_size=batch_size)
    print("---------------------------------------------------------------------Loading 'test_db'")
    test_db = create_data_loader('./Data/data_test', input_cols, sequence_length=sequence_length, batch_size=batch_size)

    model = LSTM_Model(input_dim, hidden_dim, output_dim, num_layers)

    trainer = ModelTrainer(model=model, train_db=train_db, val_db=test_db, save_path=f'./LSTM', 
                        serial_no=serial_no, batch_size=batch_size, sequence_length=sequence_length)
    trainer.train_model(num_epochs=50, learning_rate=0.001)
    
    del model
    torch.cuda.empty_cache()