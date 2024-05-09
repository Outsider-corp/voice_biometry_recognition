import json
import os
import pickle
from typing import List

import librosa
from dotenv import load_dotenv
import numpy as np
from librosa import feature

from MD.parts.clear_audio_new import clear_audio, clear_audio_newest


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


def get_spectrogram(audio, sr, n_fft: int = 2048):
    stft = np.abs(librosa.stft(audio, n_fft=n_fft))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return spectrogram


os.chdir(r'D:\Py_Projects\neuro')

load_dotenv()
data_path = r'{}'.format(os.environ['DATASET_PATH'])

# Получение данных
id_list = os.listdir(data_path)

model_params = {"persons_count": 100,
                "max_voices": 10,
                "mfcc_count": 20,
                "batch_size": 16,
                "target_sr": 16000,
                "segment_length": 0.025,
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 40,
                "lr": 1e-3,
                "margin_triplet": 0.3}
params_name = 'v1'
# Получение голосовых признаков
voice_params_mfcc = {}
voice_params_spectro = {}
pickle_file = f'voice_params_{model_params["persons_count"]}pers_{model_params["max_voices"]}vox'
pers_i = 0
for person_id in id_list:
    files = get_audio_for_id(data_path, person_id)
    person_params_mfcc = []
    person_params_spectro = []
    voice_cnt = 0
    for file in files:
        # good_audio, good_audio_segm = clear_audio(file, target_sr=model_params['target_sr'],
        #                                           segment_length=model_params['segment_length'])
        good_audio = clear_audio_newest(file, target_sr=model_params['target_sr'],
                                        n_fft=model_params['n_fft'],
                                        hop_length=model_params['hop_length'])
        good_audio_segm = None
        if good_audio_segm is not None:
            person_params_mfcc.extend(
                [get_mfccs(audio, sample_rate=model_params['target_sr'],
                           n_mfcc=model_params['mfcc_count'],
                           n_fft=model_params['n_fft']) for
                 audio in good_audio_segm])
            # person_params_mfcc.extend(
            #     [get_spectrogram(audio, target_sr) for audio in good_audio])
        else:
            person_params_mfcc.append(
                get_mfccs(good_audio, sample_rate=model_params['target_sr'],
                          n_mfcc=model_params['mfcc_count'], n_fft=model_params['n_fft'],
                          hop_length=model_params['hop_length'], n_mels=model_params['n_mels']))
        person_params_spectro.append(get_spectrogram(good_audio, model_params['target_sr'],
                                                     n_fft=model_params['n_fft']))
        voice_cnt += 1
        if voice_cnt > model_params['max_voices']:
            break
    voice_params_mfcc[person_id] = person_params_mfcc
    voice_params_spectro[person_id] = person_params_spectro
    print(f'Person {person_id} saved.')
    pers_i += 1
    if pers_i >= model_params['persons_count']:
        break
with open(os.path.join('voice_params',
                       f'{params_name}_{pickle_file}_mfcc.pkl'),
          'wb') as f:
    pickle.dump(voice_params_mfcc, f)
with open(os.path.join('voice_params', f'{params_name}_{pickle_file}_spectro.pkl'), 'wb') as f:
    pickle.dump(voice_params_spectro, f)

json.dump(model_params,
          open(os.path.join('voice_params', f'{params_name}_{pickle_file}.json'), 'w'))
