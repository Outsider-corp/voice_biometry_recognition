from typing import Dict, List, Union
import torch
import librosa
import soundfile as sf
import os
import numpy as np
import torchaudio
import sounddevice as sd
from librosa import feature

from MD.parts.torchaudio_clear import preprocess_audio_from_file, preprocess_audio_recording, \
    compute_mfcc_features

os.chdir(r'D:\Py_Projects\neuro')


def crop_and_convert_audios(input_folder: str, output_folder: str, duration_s: float = None):
    # Получаем список всех файлов во входной папке
    files = os.listdir(input_folder)

    # Проходимся по каждому файлу
    for file in files:
        # Проверяем, является ли файл аудиофайлом
        if file.endswith(".ogg"):
            crop_and_convert_audio(file, output_folder, duration_s)
    print("Готово!")


def crop_and_convert_audio(file, output_path: str = None, duration_s: float = None):
    data, samplerate = sf.read(file)  # Чтение файла

    # Определяем количество сэмплов для заданной длительности
    max_samples = int(duration_s * samplerate)

    # Если файл длиннее заданной длительности, обрезаем его
    if len(data) > max_samples:
        data = data[:max_samples]
    # Формируем новое имя файла и путь для сохранения
    if output_path is None:
        output_path = f'{file[:-3]}wav'
        print(output_path)
    # Сохранение файла в формате WAV
    sf.write(output_path, data, samplerate)

    print(f'File {file} saved.')
    return data, samplerate


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


def get_mfccs(audio, sample_rate: int, n_mfcc: int = 40, n_fft: int = 512,
              hop_length: int = 512, n_mels: int = 128):
    return feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,
                        hop_length=hop_length, n_mels=n_mels)


def get_spectrogram(audio, n_fft: int = 1024, hop_length: int = 512):
    audio = audio.detach().numpy()
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return spectrogram


def get_features_from_file(file, model_params, mfcc: bool = False, spectrogram: bool = False,
                           save_clear_audio_path: Union[str, None] = None):
    good_audio = preprocess_audio_from_file(file, target_sample_rate=model_params['target_sr'])
    if save_clear_audio_path is not None:
        torchaudio.save(f'{save_clear_audio_path}.wav', good_audio, model_params['target_sr'])
    mfcc_features = None
    spectrogram_features = None
    if mfcc:
        mfcc_features = compute_mfcc_features(good_audio,
                                              sample_rate=model_params['target_sr'],
                                              n_mfcc=model_params['mfcc_count'],
                                              n_fft=model_params['n_fft'],
                                              hop_length=model_params['hop_length'],
                                              n_mels=model_params['n_mels'],
                                              win_length=400)
        mfcc_features = mfcc_features.squeeze().tolist()
    if spectrogram:
        spectrogram_features = get_spectrogram(good_audio,
                                               n_fft=model_params['n_fft'],
                                               hop_length=model_params['hop_length'])
    return mfcc_features, spectrogram_features


def get_features_from_recording(model_params, mfcc: bool = False, spectrogram: bool = False,
                                save_clear_audio_path: Union[str, None] = None):
    good_audio = preprocess_audio_recording(target_sample_rate=model_params['target_sr'])
    if save_clear_audio_path is not None:
        torchaudio.save(f'{save_clear_audio_path}.wav', good_audio, model_params['target_sr'])
    mfcc_features = None
    spectrogram_features = None
    if mfcc:
        mfcc_features = compute_mfcc_features(good_audio,
                                              sample_rate=model_params['target_sr'],
                                              n_mfcc=model_params['mfcc_count'],
                                              n_fft=model_params['n_fft'],
                                              hop_length=model_params['hop_length'],
                                              n_mels=model_params['n_mels'],
                                              win_length=400)
        mfcc_features = mfcc_features.squeeze().tolist()
    if spectrogram:
        spectrogram_features = get_spectrogram(good_audio,
                                               n_fft=model_params['n_fft'],
                                               hop_length=model_params['hop_length'])
    return mfcc_features, spectrogram_features


def get_voice_params(data_path: str, model_params: Dict, print_info: bool = True,
                     get_mfccs: bool = True, get_spectrogram: bool = False):
    id_list = os.listdir(data_path)
    voice_params_mfcc = {}
    voice_params_spectro = {}

    for person_id in id_list:
        files = get_audio_for_id(data_path, person_id)
        person_params_mfcc = []
        person_params_spectro = []
        voice_cnt = 0
        for file in files:
            mfcc, spectro = get_features_from_file(file, model_params,
                                                   mfcc=get_mfccs,
                                                   spectrogram=get_spectrogram)
            if get_mfccs:
                person_params_mfcc.append(mfcc)
            if get_spectrogram:
                person_params_spectro.append(spectro)
            voice_cnt += 1
            if voice_cnt > model_params['max_voices']:
                break
        if get_mfccs:
            voice_params_mfcc[person_id] = person_params_mfcc
        if get_spectrogram:
            voice_params_spectro[person_id] = person_params_spectro
        if print_info:
            print(f'Person {person_id} saved.')

        if person_id == 'id10010':
            break

    return voice_params_mfcc, voice_params_spectro


if __name__ == '__main__':
    input_folder = r"D:\Py_Projects\neuro\output\ogg_voices"
    output_folder = r"D:\Py_Projects\neuro\output\wav_voices"
    duration_s = 10

    # Вызываем функцию для обрезки и конвертации аудио
    crop_and_convert_audio(input_folder, output_folder, duration_s)
