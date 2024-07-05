import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from DataLoader import Data_Loader
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTM_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # print(f'1. out.shape : {out.shape}')
        out = out[:, -1, :]
        # print(f'2. out.shape : {out.shape}')
        out = self.fc(out)
        return out

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # print(f'inputs  = {inputs}')
                # print(f'labels  = {labels}')
                print(f'labels.shape = {labels.shape}')

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # print(f'outputs = {outputs}')

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss.append(running_loss/ len(self.train_loader.dataset))
            train_acc.append(correct/total)

            self.model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted==labels).sum().item()
            
            val_loss.append(running_loss/len(self.val_loader.dataset))
            val_acc.append(correct/total)

            print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}')
        return train_loss, val_loss, train_acc, val_acc
    
    def evaluate(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

        test_loss = running_loss/len(test_loader.dataset)
        test_acc = correct/total
        return test_loss, test_acc

def plot_history(train_loss, val_loss, train_acc, val_acc, serial_no, plot_save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label = 'Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_path, f"lstm{serial_no}.png"))
    plt.show()

if __name__=="__main__":
    train_dir = './Plan_DataAcquisition/training_dataset'
    val_dir = './Plan_DataAcquisition/validation_dataset'
    test_dir = './Plan_DataAcquisition/test_dataset'

    input_cols = ['V2', 'V8', 'V10']
    #=========================================== Load data
    data_loader = Data_Loader(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, input_cols=input_cols)
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.load_data()
    #=========================================== Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    #=========================================== Convert data to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)
    #=========================================== Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    #=========================================== Define Model Parameter
    # x.train.shape: torch.Size([16903, 3])
    input_dim = x_train.shape[-1] 
    # input_dim: 3

    hidden_dim = 100
    num_layers = 3
    num_classes = len(np.unique(y_train))
    #=========================================== Create Model
    model = LSTM_Model(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #=========================================== Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #=========================================== Define Trainer
    num_epochs = 10
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    #=========================================== Train Model
    train_loss, val_loss, train_acc, val_acc = trainer.train()
    #=========================================== Evaluate Model
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f'Test accuracy = {test_acc * 100:.2f} %')
    #=========================================== Save Model and Plot
    serial_no = 1
    model_path = f'./LSTM/lstm{serial_no}/lstm{serial_no}.pth'
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    torch.save(model.state_dict(), model_path)
    print(f'Model is saved in {model_path}')

    plot_save_path = f'./LSTM/training_result'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    plot_history(train_loss, val_loss, train_acc, val_acc, serial_no, plot_save_path)
