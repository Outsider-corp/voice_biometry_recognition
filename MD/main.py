import os
from typing import List

import librosa
from librosa import feature
import torch
import torch.nn as nn
import numpy as np

data_path = r'D:\University\Диссерт\test_data_01'


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


# Получение данных
id_list = os.listdir(data_path)

# Получение голосовых признаков
voice_params = {}

for person_id in id_list:
    files = get_audio_for_id(data_path, person_id)
    person_params = []
    for file in files:
        audio, sample_rate = librosa.load(file, sr=None)

        mfcc = feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        person_params.append(mfcc)
    voice_params[person_id] = person_params