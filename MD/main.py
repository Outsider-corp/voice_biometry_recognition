import itertools
import os
import random
from typing import List, Dict, Tuple, Optional, Union

import librosa
from librosa import feature
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
import noisereduce as nr

import clearing_voice.main_project


class VoiceEmbeddingModel(nn.Module):
    def __init__(self, input_size: int = 40):
        super(VoiceEmbeddingModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.conv1 = nn.Conv1d(128, 128, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16, 256)
        self.fc2 = nn.Linear(256, 128)  # Предположим, размерность эмбеддинга 128

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Add the residual (skip) connection
        return F.relu(out)


class ResCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(ResCNN, self).__init__()

        # Первоначальный свёрточный слой
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual блоки
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 256)
        self.res_block4 = ResidualBlock(256, 512)

        # Последующие свёрточные слои
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)

        # Временной пуллинг
        self.pool = nn.AvgPool2d((1, 1))

        # Полносвязный слой
        self.fc = nn.Linear(512, 512)

        # Нормализация длины
        self.ln = nn.BatchNorm1d(512)

    def forward(self, x):
        # Начальный свёрточный слой
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        # Пропуск через рес-блоки
        x = self.res_block1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.res_block2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.res_block3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.res_block4(x)

        # Временной пуллинг
        x = self.pool(x)

        # Полносвязный слой и нормализация длины
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)

        return x


class VoicePairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mfcc1, mfcc2, label = self.pairs[idx]
        return torch.Tensor(mfcc1).flatten(), torch.Tensor(mfcc2).flatten(), torch.Tensor([label])

    @staticmethod
    def collate_fn(batch) -> Tuple:
        mfcc1s, mfcc2s, labels = zip(*batch)
        mfcc1s_padded = []
        mfcc2s_padded = []

        # Выравниваем каждую пару по максимальной длине в паре
        for mfcc1, mfcc2 in zip(mfcc1s, mfcc2s):
            max_length = max(mfcc1.shape[0], mfcc2.shape[0])
            mfcc1s_padded.append(
                F.pad(mfcc1, (0, 0, 0, max_length - mfcc1.shape[0]), "constant", 0))
            mfcc2s_padded.append(
                F.pad(mfcc2, (0, 0, 0, max_length - mfcc2.shape[0]), "constant", 0))

        mfcc1s_padded = torch.stack(mfcc1s_padded)
        mfcc2s_padded = torch.stack(mfcc2s_padded)
        labels = torch.cat(labels)

        return mfcc1s_padded, mfcc2s_padded, labels


class ConstrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ConstrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 0.5) +
                                      (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 0.5))
        loss_contrastive *= 1000
        return loss_contrastive


def train_model(model: nn.Module, dataloaders: Dict, criterion, lr=0.001,
                epoches: int = 25, device: str = 'cuda') -> nn.Module:
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
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(epoches):
        print(f'Epoch {epoch + 1}/{epoches}')
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
            for inputs1, inputs2, labels in dataloaders[phase]:
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)

                # Обнуление градиентов параметров
                optimizer.zero_grad()

                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1 = model(inputs1)
                    outputs2 = model(inputs2)
                    loss = criterion(outputs1, outputs2, labels)

                    # Обратное распространение и оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs1.size(0)

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


def preprocess_audio(file_path: str, target_sr: int = 16000, segment_length: float = 0,
                     clear: bool = False, clear_output: str = None) -> Union[
    List[np.ndarray], np.ndarray]:
    """
    Функция для предобработки аудиозаписей.
    :param file_path: str - Путь к аудиофайлу.
    :param target_sr: int - Целевая частота дискретизации.
    :param segment_length: int - Длина сегмента в секундах.
    :return: list -  Список сегментов аудиосигнала, каждый из которых является np.array.
    """
    # Загрузка аудио файла
    y, sr = get_voice_audio(file_path, clear=clear, clear_output=clear_output)

    # Уменьшение шума
    y_clean = nr.reduce_noise(y=y, sr=sr)

    # Нормализация амплитуды
    y_norm = librosa.util.normalize(y_clean)

    # Ресемплинг до целевой частоты дискретизации
    if sr != target_sr:
        y_resampled = librosa.resample(y_norm, orig_sr=sr, target_sr=target_sr)
    else:
        y_resampled = y_norm

    # Обрезка тихих участков
    y_trimmed, _ = librosa.effects.trim(y_resampled)

    # Сегментация
    if segment_length:
        segment_samples = segment_length * target_sr
        segments = [y_trimmed[i:i + segment_samples] for i in
                    range(0, len(y_trimmed), segment_samples)
                    if i + segment_samples <= len(y_trimmed)]
        return segments
    else:
        return y_trimmed


# Загрузка аудиофайла
def load_audio(filename):
    audio, sr = librosa.load(filename, sr=None)
    return audio, sr


# Вычисление MFCC
def compute_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


# Вычисление спектрограммы
def get_spectrogram(audio, sr):
    stft = np.abs(librosa.stft(audio))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return spectrogram


# Вычисление хромаграммы
def compute_chromagram(audio, sr):
    chromagram = librosa.feature.chroma_stft(y=audio, sr=sr)
    return chromagram


# Вычисление zero-crossing rate
def compute_zcr(audio):
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    return zcr


# Вычисление тональности (pitch)
def compute_pitch(audio, sr):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = np.max(pitches, axis=0)
    return pitch


def get_voice_audio(audio_path: str, clear: bool = False, clear_output: Optional[str] = None):
    if clear:
        audio, sample_rate = clearing_voice.main_project.clear_audio(audio_path,
                                                                     output_path=clear_output)
    else:
        audio, sample_rate = librosa.load(audio_path, sr=None)
    # mfcc = feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return audio, sample_rate


def get_mfccs(audio, sample_rate: int, n_mfcc: int = 40):
    return feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)


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


if __name__ == '__main__':

    load_dotenv()

    data_path = r'{}'.format(os.environ['DATASET_PATH'])

    # Получение данных
    id_list = os.listdir(data_path)
    batch_size = 32
    # Получение голосовых признаков
    voice_params = {}
    for person_id in id_list:
        files = get_audio_for_id(data_path, person_id)
        person_params = []
        for file in files:
            person_params.append(get_voice_mfccs(file, n_mfcc=15))
        voice_params[person_id] = person_params
        print(f'Person {person_id} saved.')

    data_pairs = create_pairs(voice_params)
    random.shuffle(data_pairs)
    # Разделение на обучающую и валидационную выборки
    val_size = int(0.2 * len(data_pairs))
    data_train = data_pairs[val_size:]
    data_val = data_pairs[:val_size]

    dataset_train = VoicePairsDataset(data_train)
    dataset_val = VoicePairsDataset(data_val)
    dataloaders = {'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
                   'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=True)}

    model = VoiceEmbeddingModel()
    model = train_model(model, dataloaders, ConstrastiveLoss(), epoches=15)
    torch.save(model.state_dict(), f'speak_rec_15_256_128_10epo.pth')

    # dataset_train = TensorDataset(torch.tensor(mfccs_train), torch.tensor(labels_train))
    # dataset_val = TensorDataset(torch.tensor(mfccs_val), torch.tensor(labels_val))
    # dataset = {'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
    #            'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=True)}
    # speak_rec = train_model(speak_rec, inputs, labels, epoches=10)
