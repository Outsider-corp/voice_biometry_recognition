import os
import random
from typing import List, Dict

import librosa
from librosa import feature
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

data_path = os.environ['DATASET_PATH']


class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, input_size: int = 40, hidden_size: int = 256, embedding_dim: int = 128):
        super(SpeakerEmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConstrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = (output1 - output2).pow(2).sum(1)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 0.5) +
                                      (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 0.5))
        return loss_contrastive


def fit_model(model: nn.Module, input_data: torch.Tensor = None, labels: torch.Tensor = None,
              loader=None, epoches: int = 10,
              lr=0.001, batch_size: int = 32, criterion=None):
    criterion = nn.TripletMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not loader:
        dataset = TensorDataset(input_data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epoches):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data.float()).squeeze()
            loss = criterion()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model


def train_model(model: nn.Module, dataloaders: Dict, criterion, optimizer: optim.Optimizer,
                num_epochs: int = 25, device: str = 'cuda') -> nn.Module:
    """
    Обучает модель и выводит информацию о процессе обучения.
   :param model: torch.nn.Module -  Модель для обучения.
   :param dataloaders: dict - Словарь содержащий 'train' и 'val' DataLoader.
   :param criterion: torch.nn.modules.loss - Функция потерь.
   :param optimizer: torch.optim.Optimizer - Оптимизатор.
   :param num_epochs: int - Количество эпох обучения.
   :param device: str - Устройство для обучения ('cuda' или 'cpu').
    :return: nn.Module - Обученная модель.
    """
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Каждая эпоха имеет фазу обучения и валидации
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Установка модели в режим обучения
            else:
                model.eval()  # Установка модели в режим оценки

            running_loss = 0.0
            running_corrects = 0

            # Итерация по данным.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Обнуление градиентов параметров
                optimizer.zero_grad()

                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Обратное распространение и оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Копирование модели, если она показала лучшую точность
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Лучшая точность валидации: {best_acc:.4f}')

    # Загрузка лучших весов модели
    model.load_state_dict(best_model_wts)
    return model


def get_audio_for_id(folder: str, id: str) -> List[str]:
    """
    Получение всех аудиозаписей для конкретного id
    :param folder: str - папка со всеми id (папками)
    :param id: str - название id
    :return: List[str] - список файлов с аудиозаписями
    """
    id_path = os.path.join(folder, id)
    audiofiles_list = []
    for root, dir, files in os.walk(id_path):
        audiofiles_list.extend(os.path.join(root, file) for file in files)
    return audiofiles_list


def average_mfcc(mfcc_list: List) -> np.ndarray:
    mfcc_stack = np.hstack(mfcc_list)
    mean_mfcc = np.mean(mfcc_stack, axis=1)
    return mean_mfcc


def get_voice_mfccs(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
    audio, sample_rate = librosa.load(audio_path, sr=None)
    mfcc = feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc


# Получение данных
id_list = os.listdir(data_path)

# Получение голосовых признаков
voice_params = []
for person_id in id_list:
    files = get_audio_for_id(data_path, person_id)
    person_params = []
    for file in files:
        voice_params.append((person_id, get_voice_mfccs(file, n_mfcc=15)))

speak_rec = SpeakerEmbeddingModel(15)

random.shuffle(voice_params)

# Создание тензоров
inputs =[params[1] for params in voice_params]
labels =[params[0] for params in voice_params]


mfccs_train, mfccs_val, labels_train, labels_val = train_test_split(
    inputs, labels, test_size=0.2, random_state=42
)
# dataset = {'train': }
speak_rec = train_model(speak_rec, inputs, labels, epoches=10)
torch.save(speak_rec.state_dict(), f'speak_rec_15_256_128_10epo.pth')
