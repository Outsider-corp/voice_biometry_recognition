import os
import pickle
import json
from typing import Dict, List, Literal

import pandas as pd
import torch
from torch import nn
from torchvision import models

from MD.get_person_voice import get_voice_params, get_features_from_file, crop_and_convert_audio, \
    get_features_from_recording
from MD.parts.loss_acc import cosine_similarity
from speechnas.network.dtdnnss_base import DtdnnssBase

os.chdir(r'D:\Py_Projects\neuro')


def get_model(model, weights_file: str):
    state_dict = torch.load(weights_file)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def predict_model(model, input_data: List[List], device: str = 'cuda',
                  with_extract: bool = True):
    if with_extract:
        return model.extract_embedding(input_data)
    else:
        model.to(device)
        input_data = torch.Tensor(input_data).to(device).unsqueeze(0)
        output = model(input_data)
        return output.cpu().view(-1).tolist()


def get_embeddings(model, mfccs: Dict, device: str = 'cuda',
                   with_extract: bool = True):
    embs = {pers: [] for pers in mfccs}
    for person in mfccs:
        # embs[person] = predict_model(model, mfccs[person])
        for mfcc in mfccs[person]:
            embs[person].append(predict_model(model, mfcc, device, with_extract))
    return embs


def emb_from_file(model, file: str, model_params,
                  features_type: Literal['mfcc', 'spectro'],
                  device: str = 'cuda', with_extract: bool = True):
    if features_type == 'mfcc':
        features, _ = get_features_from_file(file, model_params, mfcc=True)
    elif features_type == 'spectro':
        _, features = get_features_from_file(file, model_params, spectrogram=True)
    else:
        return None
    emb = predict_model(model, features, device, with_extract)
    return emb


def emb_from_rec(model, model_params,
                 features_type: Literal['mfcc', 'spectro'],
                 save_file: str = None, device: str = 'cuda',
                 with_extract: bool = True):
    if features_type == 'mfcc':
        features, _ = get_features_from_recording(model_params, mfcc=True,
                                                  save_clear_audio_path=save_file)
    elif features_type == 'spectro':
        _, features = get_features_from_recording(model_params, spectrogram=True,
                                                  save_clear_audio_path=save_file)
    else:
        return None
    emb = predict_model(model, features, device, with_extract)
    return emb


def get_DTDNNSS_model(weights_file: str = None, device: str = 'cuda', num_class: int = 7232):
    if weights_file is None:
        weights_file = r'models\7. DTDNNSS_01\DTDNNSS_20mfcc_001.pth'

    model = DtdnnssBase(num_class=num_class, feature_dim=128).float()
    model = get_model(model, weights_file).to(device)
    return model


def get_ResNet_model(weights_file: str = None, device: str = 'cuda'):
    if weights_file is None:
        weights_file = r'models\11. ResNet34_5\ResNet34_100p_10vox_003_128_epo88.pth'

    model = models.resnet34()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Обучение модели
    model.fc = nn.Linear(in_features=512, out_features=128, bias=True)
    model = get_model(model, weights_file).to(device)
    return model


def create_db(model, model_params, voices_path: str, output_path: str,
              features_type: Literal['mfcc', 'spectro']):
    if features_type == 'mfcc':
        mfccs, _ = get_voice_params(voices_path, model_params,
                                    get_mfccs=True, print_info=False)
        embs = get_embeddings(model, mfccs)
    elif features_type == 'spectro':
        _, spectros = get_voice_params(voices_path, model_params,
                                       get_spectrogram=True, print_info=False)
        embs = get_embeddings(model, spectros)
    else:
        return
    with open(output_path, 'w') as f:
        json.dump(embs, f)
    print(f'End.')


def load_db(db_path: str):
    with open(db_path, 'r') as f:
        embs = json.load(f)
    return embs


def identification_emb(db: Dict, target_emb: List):
    df = pd.DataFrame(columns=['user', 'voice', 'cos'])
    for user in db:
        for i, emb in enumerate(db[user]):
            cos = cosine_similarity(emb, target_emb)
            df.loc[len(df)] = {'user': user, 'voice': i, 'cos': cos}
    df.sort_values(by='cos', ascending=False, inplace=True)
    return df


if __name__ == '__main__':
    model = get_DTDNNSS_model()

    with open('voice_params/v3_voice_params_100pers_5vox.json', 'r') as f:
        model_params = json.load(f)
    # model_params['mfcc_count'] = 30
    # create_db(model, model_params, r'D:\Py_Projects\neuro\output\wav_voices',
    #           'output/db3.json', features_type='mfcc')
    test_data = r'output\test_rec_wav.wav'

    mfccs, _ = get_features_from_file(test_data, model_params, mfcc=True)

    emb = model.extract_embedding(mfccs)
    # test_data = r'D:\University\Диссерт\test_data_01_full\id10001\1zcIwhmdeo4\00001.wav'
    # crop_and_convert_audio(test_data, duration_s=10)
    # emb = emb_from_file(model, f'{test_data[:-3]}wav', model_params)
    # emb = emb_from_rec(model, model_params, f'output/test_rec_heads')
    #
    print(emb)
