import os
import time
from typing import Dict, Union
import json

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from MD.parts.loss_acc import TripletLoss, triplet_accuracy, ConstrastiveLoss, \
    cosine_similarity_pair, pair_accuracy, batch_cosine_similarity


def pre_train_model(model: nn.Module, dataloaders,
                    lr=0.001, epoches: int = 25,
                    device: str = 'cuda',
                    stat: Union[Dict, None] = None) -> nn.Module:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    criterion = nn.CrossEntropyLoss()
    stat = stat or {}
    for epoch in range(epoches):
        running_loss = 0.0
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloaders.dataset)
        stat[f'{epoch}'] = {f'epoch_loss': epoch_loss}
        print(f'Epoch {epoch + 1}/{epoches}, Loss: {epoch_loss:.4f}')

    return model, stat


def train_model_triplet(model: nn.Module, dataloaders: Dict,
                        lr=0.001, epoches: int = 25,
                        device: str = 'cuda', save_name: str = None,
                        margin: float = 1, start_epo: int = 0,
                        stat: Union[Dict, None] = None) -> nn.Module:
    """
    Обучает модель и выводит информацию о процессе обучения.
   :param model: torch.nn.Module -  Модель для обучения.
   :param dataloaders: dict - Словарь содержащий 'train' и 'val' DataLoader.
   :param criterion: torch.nn.modules.loss - Функция потерь.
   :param lr: float
   :param epoches: int - Количество эпох обучения.
   :param device: str - Устройство для обучения ('cuda' или 'cpu').
    :return: nn.Module - Обученная модель.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TripletLoss(margin)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    stat = stat or {}
    start_time = time.time()
    for epoch in range(start_epo, epoches):
        print(f'Epoch {epoch + 1}/{epoches} | time_elapsed: {time.time() - start_time}')
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
            for inputs1, inputs2, inputs3 in dataloaders[phase]:
                inputs1 = inputs1.to(device).float()
                inputs2 = inputs2.to(device).float()
                inputs3 = inputs3.to(device).float()

                # Обнуление градиентов параметров
                optimizer.zero_grad()

                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1 = model(inputs1)
                    outputs2 = model(inputs2)
                    outputs3 = model(inputs3)
                    loss = criterion(outputs1, outputs2, outputs3)

                    # Обратное распространение и оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                acc = triplet_accuracy(outputs1, outputs2, outputs3, margin)
                # Статистика
                running_loss += loss.item() * inputs1.size(0)
                running_corrects += acc.item() * inputs1.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            stat[f'{epoch}'] = {f'epoch_loss_{phase}': epoch_loss,
                                f'epoch_acc_{phase}': epoch_acc}
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Копирование модели, если она показала лучшую точность
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if save_name:
                torch.save(model.state_dict(),
                           os.path.join('models', f'{save_name}_epo{epoch}.pth'))
                json.dump(stat, open(os.path.join('models', f'{save_name}.json'), 'w'))
        print()
    print(f'Лучшая точность валидации: {best_acc:.4f}')
    stat['full_time'] = time.time() - start_time
    if save_name:
        json.dump(stat, open(os.path.join('models', f'{save_name}.json'), 'w'))
    # Загрузка лучших весов модели
    model.load_state_dict(best_model_wts)
    return model, stat


def train_model_pair(model: nn.Module, dataloaders: Dict, lr=0.001,
                     epoches: int = 25, device: str = 'cuda', save_name: str = None,
                     margin: float = 1, start_epo: int = 0,
                     stat: Union[Dict, None] = None) -> nn.Module:
    """
    Обучает модель и выводит информацию о процессе обучения.
   :param model: torch.nn.Module -  Модель для обучения.
   :param dataloaders: dict - Словарь содержащий 'train' и 'val' DataLoader.
   :param criterion: torch.nn.modules.loss - Функция потерь.
   :param lr: float
   :param epoches: int - Количество эпох обучения.
   :param device: str - Устройство для обучения ('cuda' или 'cpu').
    :return: nn.Module - Обученная модель.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ConstrastiveLoss(margin)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    stat = stat or {}
    start_time = time.time()
    for epoch in range(start_epo, epoches):
        print(f'Epoch {epoch + 1}/{epoches} | time_elapsed: {time.time() - start_time}')
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
            for inputs1, inputs2, label in dataloaders[phase]:
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                label = label.to(device)

                # Обнуление градиентов параметров
                optimizer.zero_grad()

                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1 = model(inputs1)
                    outputs2 = model(inputs2)

                    cos = batch_cosine_similarity(outputs1, outputs2)
                    loss = criterion(cos, label)
                    # Обратное распространение и оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                acc = pair_accuracy(cos, label, margin)
                # Статистика
                running_loss += loss.item() * inputs1.size(0)
                running_corrects += acc.item() * inputs1.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            stat[f'{epoch}'] = {f'epoch_loss_{phase}': epoch_loss,
                                f'epoch_acc_{phase}': epoch_acc}
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # Копирование модели, если она показала лучшую точность
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if save_name:
                torch.save(model.state_dict(),
                           os.path.join('models', f'{save_name}_epo{epoch}.pth'))
                json.dump(stat, open(os.path.join('models', f'{save_name}.json'), 'w'))
        print()
    print(f'Лучшая точность валидации: {best_acc:.4f}')
    stat['full_time'] = time.time() - start_time
    if save_name:
        json.dump(stat, open(os.path.join('models', f'{save_name}.json'), 'w'))
    # Загрузка лучших весов модели
    model.load_state_dict(best_model_wts)

    return model, stat
