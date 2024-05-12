import json
import pickle
import random
from typing import Tuple

import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os

from torchvision.models import ResNet34_Weights

from MD.parts.datasets import create_triplets
from MD.parts.train_funcs import train_model_triplet

os.chdir(r'D:\Py_Projects\neuro')


# Подготовка данных
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, transform=None):
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]
        if self.transform:
            spec = self.transform(spec)
        return spec, label


class VoiceTripletsSpectrDataset(Dataset):
    def __init__(self, pairs):
        self.triplets = pairs
        # Трансформация спектрограмм
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Изменение размера спектрограммы
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        mfcc1, mfcc2, mfcc3 = self.triplets[idx]
        # Применяем трансформацию
        mfcc1 = self.transform(torch.from_numpy(mfcc1).float())
        mfcc2 = self.transform(torch.from_numpy(mfcc2).float())
        mfcc3 = self.transform(torch.from_numpy(mfcc3).float())
        return mfcc1, mfcc2, mfcc3


def create_triplets_dataset(voice_params, model_params, count_x_data: int = 2):
    dataset_class = VoiceTripletsSpectrDataset

    data_pairs = create_triplets(voice_params, count_x_data)
    random.shuffle(data_pairs)
    # Разделение на обучающую и валидационную выборки
    val_size = int(0.2 * len(data_pairs))
    data_train = data_pairs[val_size:]
    data_val = data_pairs[:val_size]

    dataset_train = dataset_class(data_train)
    dataset_val = dataset_class(data_val)
    dataloader_train = DataLoader(dataset_train, batch_size=model_params['batch_size'],
                                  shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=model_params['batch_size'],
                                shuffle=True)
    dataloaders = {'train': dataloader_train,
                   'val': dataloader_val}

    return dataloaders


model_params = {'persons_count': 100,
                'max_voices': 10,
                'mfcc_count': 30,
                'batch_size': 16,
                'target_sr': 16000,
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 64,
                'lr': 0.001,
                'margin_triplet': 0.1,
                'epoches': 100}

with open(f'voice_params/v10_voice_params_100pers_10vox_spectro.pkl', 'rb') as f:
    spec_params = pickle.load(f)

dataloaders = create_triplets_dataset(spec_params, model_params, count_x_data=2)

model_name = 'ResNet34_100p_10vox_003_128'

with open(os.path.join('models', f'params_{model_name}.json'), 'w') as f:
    json.dump(model_params, f)
# Загрузка предобученной модели ResNet34
model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Обучение модели
model.fc = nn.Linear(in_features=512, out_features=128, bias=True)

state_dict = torch.load(f'models/ResNet34_100p_10vox_003_128_epo66.pth')
model.load_state_dict(state_dict, strict=False)
stat = json.load(open(os.path.join('models', f'ResNet34_100p_10vox_003_128.json')))
model, stat = train_model_triplet(model, dataloaders,
                                  epoches=model_params['epoches'],
                                  save_name=model_name,
                                  lr=model_params['lr'],
                                  margin=model_params['margin_triplet'],
                                  start_epo=67, stat=stat)
torch.save(model.state_dict(), os.path.join('models', f'{model_name}___end.pth'))
