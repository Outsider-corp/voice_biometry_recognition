import itertools
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class PreTrainDataset(Dataset):
    def __init__(self, data_dict):
        self.speakers = list(data_dict.keys())
        self.features = []  # список всех MFCC признаков
        self.labels = []  # список меток классов

        for key in data_dict:
            self.features.extend(data_dict[key])
            self.labels.extend([int(key[-3:]) - 1] * len(data_dict[key]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.features[index],
                            dtype=torch.float32).squeeze().t(), self.labels[index]

    @staticmethod
    def collate_fn(batch):
        mfccs, labels = zip(*batch)
        # Преобразуем каждый элемент mfccs в тензор, добавляем размерность канала и корректируем размерности
        # Нужно убедиться, что размерности соответствуют (batch_size, channels, height, width)
        mfccs_padded = pad_sequence(mfccs, batch_first=True, padding_value=0).transpose(1, 2).unsqueeze(1)
        labels = torch.tensor(labels, dtype=torch.long)
        return mfccs_padded, labels


class VoicePairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mfcc1, mfcc2, label = self.pairs[idx]
        return torch.tensor(mfcc1, dtype=torch.float32).t(), torch.tensor(mfcc2,
                                                                          dtype=torch.float32).t(), torch.tensor(
            label, dtype=torch.int64)

    @staticmethod
    def collate_fn(batch):
        mfcc1s, mfcc2s, labels = zip(*batch)
        mfcc1s_padded = pad_sequence(mfcc1s,
                                     batch_first=True,
                                     padding_value=0).transpose(1, 2).unsqueeze(1)
        mfcc2s_padded = pad_sequence(mfcc2s,
                                     batch_first=True,
                                     padding_value=0).transpose(1, 2).unsqueeze(1)

        labels = torch.stack(labels).long()  # Убедитесь, что labels имеют правильный тип

        return mfcc1s_padded, mfcc2s_padded, labels


class VoiceTripletsDataset(Dataset):
    def __init__(self, pairs):
        self.triplets = pairs

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        mfcc1, mfcc2, mfcc3 = self.triplets[idx]
        return mfcc1.squeeze(0).t(), mfcc2.squeeze(0).t(), mfcc3.squeeze(0).t()

    @staticmethod
    def collate_fn(batch) -> Tuple:
        mfcc1s, mfcc2s, mfcc3s = zip(*batch)
        mfcc1s_padded = pad_sequence(mfcc1s, batch_first=True, padding_value=0).transpose(1, 2).unsqueeze(1)
        mfcc2s_padded = pad_sequence(mfcc2s, batch_first=True, padding_value=0).transpose(1, 2).unsqueeze(1)
        mfcc3s_padded = pad_sequence(mfcc3s, batch_first=True, padding_value=0).transpose(1, 2).unsqueeze(1)
        return mfcc1s_padded, mfcc2s_padded, mfcc3s_padded

class VoiceTripletsTDNNDataset(Dataset):
    def __init__(self, pairs):
        self.triplets = pairs

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        mfcc1, mfcc2, mfcc3 = self.triplets[idx]
        # Переворачиваем размерности сразу здесь, так что последовательность идёт первой, а MFCC второй
        return mfcc1.squeeze(0).t(), mfcc2.squeeze(0).t(), mfcc3.squeeze(0).t()

    @staticmethod
    def collate_fn(batch) -> Tuple:
        mfcc1s, mfcc2s, mfcc3s = zip(*batch)
        # Используем pad_sequence для создания паддинга, и устанавливаем batch_first=True
        # Не нужно переставлять размерности так, как было предложено
        mfcc1s_padded = pad_sequence(mfcc1s, batch_first=True, padding_value=0).permute(0, 2, 1)
        mfcc2s_padded = pad_sequence(mfcc2s, batch_first=True, padding_value=0).permute(0, 2, 1)
        mfcc3s_padded = pad_sequence(mfcc3s, batch_first=True, padding_value=0).permute(0, 2, 1)
        # Нужно изменить форму тензоров, чтобы коэффициенты MFCC были в каналах
        return mfcc1s_padded, mfcc2s_padded, mfcc3s_padded

def create_pair_dataset(voice_params, model_params):
    data_pairs = create_pairs(voice_params)
    random.shuffle(data_pairs)
    # Разделение на обучающую и валидационную выборки
    val_size = int(0.2 * len(data_pairs))
    data_train = data_pairs[val_size:]
    data_val = data_pairs[:val_size]

    dataset_train = VoicePairsDataset(data_train)
    dataset_val = VoicePairsDataset(data_val)
    dataloader_train = DataLoader(dataset_train, batch_size=model_params['batch_size'],
                                  shuffle=True,
                                  collate_fn=VoicePairsDataset.collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=model_params['batch_size'], shuffle=True,
                                collate_fn=VoicePairsDataset.collate_fn)
    dataloaders = {'train': dataloader_train,
                   'val': dataloader_val}
    return dataloaders


def create_triplets_dataset(voice_params, model_params, count_x_data: int = 2):
    dataset_class = VoiceTripletsTDNNDataset

    data_pairs = create_triplets(voice_params, count_x_data)
    random.shuffle(data_pairs)
    # Разделение на обучающую и валидационную выборки
    val_size = int(0.2 * len(data_pairs))
    data_train = data_pairs[val_size:]
    data_val = data_pairs[:val_size]

    dataset_train = dataset_class(data_train)
    dataset_val = dataset_class(data_val)
    dataloader_train = DataLoader(dataset_train, batch_size=model_params['batch_size'],
                                  shuffle=True,
                                  collate_fn=dataset_class.collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=model_params['batch_size'],
                                shuffle=True,
                                collate_fn=dataset_class.collate_fn)
    dataloaders = {'train': dataloader_train,
                   'val': dataloader_val}

    return dataloaders


def create_pairs(data):
    positive_pairs = []
    negative_pairs = []

    # Создание позитивных пар
    for user_id, mfccs in data.items():
        if len(mfccs) > 1:
            for mfcc1, mfcc2 in itertools.combinations(mfccs, 2):
                positive_pairs.append((mfcc1, mfcc2, 1))

    # Создание негативных пар
    users_pairs = itertools.combinations(data.keys(), 2)
    for user1, user2 in users_pairs:
        if data[user1] and data[user2]:
            mfcc1 = random.choice(data[user1])
            mfcc2 = random.choice(data[user2])
            negative_pairs.append((mfcc1, mfcc2, 0))

    return positive_pairs + negative_pairs


def create_balanced_pairs(data, max_pairs_per_user=10):
    positive_pairs = []
    negative_pairs = []

    # Создание позитивных пар
    for user_id, mfccs in data.items():
        if len(mfccs) > 1:
            for mfcc1, mfcc2 in itertools.combinations(mfccs, 2):
                positive_pairs.append((mfcc1, mfcc2, 1))
                if len(positive_pairs) >= max_pairs_per_user:
                    break

    # Создание негативных пар
    all_users = list(data.keys())
    for _ in range(len(positive_pairs)):  # Балансировка количества негативных пар
        user1, user2 = np.random.choice(all_users, 2, replace=False)
        if data[user1] and data[user2]:
            mfcc1 = random.choice(data[user1])
            mfcc2 = random.choice(data[user2])
            negative_pairs.append((mfcc1, mfcc2, 0))

    return positive_pairs + negative_pairs


def create_triplets(data, count_x_data: int = 50):
    triplets = []

    # Список всех пользователей для легкого доступа
    user_ids = list(data.keys())

    pair_for_user = {user_id: list(itertools.combinations(mfccs, 2)) for user_id, mfccs in
                     data.items() if len(mfccs) > 1}
    total_pairs = sum([len(x) for x in pair_for_user.values()])
    user_weights = [len(pair_for_user[uid]) for uid in user_ids if uid in pair_for_user]

    desired_triplets_count = total_pairs * count_x_data

    # Создание троек
    while len(triplets) < desired_triplets_count:
        chosen_user_id = random.choices([uid for uid in user_ids if uid in pair_for_user],
                                        weights=user_weights,
                                        k=1)[0]
        mfccs = data[chosen_user_id]

        # Создаем все возможные пары анкер-позитив и выбираем одну случайную пару
        anchor_positive_pairs = list(itertools.combinations(mfccs, 2))
        anchor, positive = random.choice(anchor_positive_pairs)

        other_users = [uid for uid in user_ids if uid != chosen_user_id and data[uid]]
        if other_users:
            negative_user_id = random.choice(other_users)
            negative = random.choice(data[negative_user_id])
            triplets.append((anchor, positive, negative))

    return triplets
