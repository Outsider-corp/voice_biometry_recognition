import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import pandas as pd
import numpy as np

import os
import random

df = pd.read_csv("dirt_data.csv")

try:
    df = df.drop(labels=["unix", "date", "symbol", "Volume BTC", "Volume USD"], axis=1)
except:
    pass

train = []
val = []

back_data = 100 # количество свечь на вход
size_item = 4 # 4 элемента в свече

def calculate_changes(array):
    changes = []
    prev_stage = [0] * 4  # Начинаем с нулевого изменения

    for i in range(0, len(array), size_item):
        stage = array[i:i+size_item]
        if i != 0:
            percentage_change = [((stage[j] - prev_stage[j]) / prev_stage[j]) * 100 if prev_stage[j] != 0 else 0 for j in range(4)]
            changes.extend(percentage_change)
        else:
            changes.extend([0.0,0.0,0.0,0.0])
        prev_stage = stage

    return changes

def write(arr, data):
    data = data[::-1]
    data = [item for sublist in data for item in sublist]
    changes = calculate_changes(data)
    arr.append(changes)

temp_data = []
for index, row in df.iterrows():
    temp_data.append(row.values.tolist())
    if index >= 29000 and index != 0 and index % back_data == 0:
        write(val, temp_data)
        temp_data = []
    elif index < 29000 and index != 0 and index % back_data == 0:
        write(train, temp_data)
        temp_data = []

# Определение архитектуры LSTM модели
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm11 = nn.LSTM(input_size, hidden_size)
        self.lstm12 = nn.LSTM(hidden_size, hidden_size)
        self.lstm13 = nn.LSTM(hidden_size, hidden_size)
        self.linear11 = nn.Linear(hidden_size, output_size)

        self.lstm21 = nn.LSTM(input_size, hidden_size)
        self.lstm22 = nn.LSTM(hidden_size, hidden_size)
        self.lstm23 = nn.LSTM(hidden_size, hidden_size)
        self.linear21 = nn.Linear(hidden_size, output_size)

        self.linear = nn.Linear(output_size, output_size)

    def forward(self, input):
        output1, (h_n1, c_n1) = self.lstm11(input)
        output2, (h_n2, c_n2) = self.lstm21(input)
       
        output12 = output1 + output2
        h_n12 = h_n1 + h_n2
        c_n12 = c_n1 + c_n2

        output12, (h_n_12, c_n_12) = self.lstm12(output12, (h_n12, c_n12))
        output13, _ = self.lstm13(output12, (h_n_12, c_n_12))
        output_linear_out_1 = self.linear11(output13)

        output22, (h_n_22, c_n_22) = self.lstm22(output12, (h_n12, c_n12))
        output23, _ = self.lstm23(output22, (h_n_22, c_n_22))
        output_linear_out_2 = self.linear21(output23)

        output = output_linear_out_1 + output_linear_out_2
        
        last_output = output[:size_item]
        output = self.linear(last_output)  
        return output
    

# Инициализация модели
input_size = 1
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# Определение функции потерь и оптимизатора
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    idx = np.random.randint(0, len(train))
    input_sequence = torch.from_numpy(np.array(train[idx][0:back_data-size_item], dtype=np.float32)).unsqueeze(1)  # Создание тензора из массива numpy
    target_sequence = torch.from_numpy(np.array(train[idx][back_data-size_item:back_data], dtype=np.float32)).unsqueeze(1) # Создание тензора из массива numpy
    output = model(input_sequence)
    
    # Вычисляем ошибку с учетом весов
    try:
        loss = (torch.mean((output - target_sequence) ** 2 * 1_000_000)).squeeze()
    except:
        pass
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        # print(f'Input: {input_sequence}')
        print(f'Target: {target_sequence.tolist()}')
        # output_values = [round(float(val)) for val in output.squeeze().detach().numpy()]
        print(f'Output: {output.tolist()}')
        print()
torch.save(model.state_dict(), 'model_weights.pth')
