from typing import List
import librosa
import numpy as np
from librosa.feature import mfcc


def preprocess_audio(audio_path: str, target_sr: int = 16000,
                     duration: int = 5):
    """
    Предобработка сигнала
    :param duration: int - длительность выходного сигнала
    :param audio_path: str -путь к аудиофайлу
    :param target_sr: int - целевой битрейт
    :return: ndarray, int - обработанный сигнал и его битрейт
    """
    # Загрузка аудиофайла
    audio_data, sr = librosa.load(audio_path, sr=None, duration=duration,
                                  mono=True)

    # Нормализация сигнала
    audio_data = librosa.util.normalize(audio_data)

    # Изменение битрейта
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)


    return audio_data, target_sr


def extract_mfcc(audio_data, num_mfcc: int = 13) -> List[float]:
    """
    Извлечение мел-кепстральных коэффициентов из аудиозаписи
    :param audio_data - входной сигнал
    :param num_mfcc: int - количество возвращаемых коэффициентов
    :return: list - усреднённые по времени коэффициенты mfcc
    """

    # Извлечение MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=len(audio_data), n_mfcc=num_mfcc)

    # Усреднение по времени
    avg_mfccs = np.mean(mfccs.T, axis=0)

    return mfccs
