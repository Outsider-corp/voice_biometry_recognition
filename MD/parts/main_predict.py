import os
import pickle
import json
from typing import Dict, List

import pandas as pd
import torch

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


def predict_model(model, input_data: List[List], device: str = 'cuda'):
    model.to(device)
    input_data = torch.Tensor(input_data).to(device).unsqueeze(0)
    output = model(input_data).cpu()
    return output.cpu().view(-1).tolist()


def get_embeddings(model, mfccs: Dict, device: str = 'cuda'):
    embs = {pers: [] for pers in mfccs}
    for person in mfccs:
        # embs[person] = predict_model(model, mfccs[person])
        for mfcc in mfccs[person]:
            embs[person].append(predict_model(model, mfcc, device))
    return embs


def emb_from_file(model, file: str, model_params, device: str = 'cuda'):
    mfcc = get_features_from_file(file, model_params, mfcc=True)
    emb = predict_model(model, mfcc, device)
    return emb


def emb_from_rec(model, model_params, save_file: str = None, device: str = 'cuda'):
    mfcc = get_features_from_recording(model_params, mfcc=True, save_clear_audio_path=save_file)
    emb = predict_model(model, mfcc, device)
    return emb


def get_DTDNNSS_model():
    models_path = r'models\correct'
    weights = 'DeepSpeaker_100pers_10vox_20mfcc_tripl_011___end.pth'
    weights_file = os.path.join(models_path, weights)

    model = DtdnnssBase(num_class=7232, feature_dim=128).float()
    model = get_model(model, weights_file)
    return model


def create_db(model, model_params, voices_path: str, output_path: str):
    mfccs, spectr = get_voice_params(voices_path, model_params, print_info=False)

    embs = get_embeddings(model, mfccs)
    with open(f'{output_path}.json', 'w') as f:
        json.dump(embs, f)
    print(f'End.')


def load_db(db_path: str):
    with open(f'{db_path}.json', 'r') as f:
        embs = json.load(f)
    return embs


def identification_emb(db: Dict, target_emb: List):
    df = pd.DataFrame(columns=['user', 'voice', 'cos'])
    for user in db:
        for i, emb in enumerate(db[user]):
            cos = cosine_similarity(emb, target_emb)
            df.loc[len(df)] = {'user': user, 'voice': i, 'cos': cos}
    df.sort_values(by='cos', ascending=True, inplace=True)
    return df


if __name__ == '__main__':
    model = get_DTDNNSS_model()

    with open('voice_params/v3_voice_params_100pers_5vox.json', 'r') as f:
        model_params = json.load(f)
    test_data = r'output\test\test_roma1.ogg'
    # test_data = r'D:\University\Диссерт\test_data_01_full\id10001\1zcIwhmdeo4\00001.wav'
    # crop_and_convert_audio(test_data, duration_s=10)
    # emb = emb_from_file(model, f'{test_data[:-3]}wav', model_params)
    emb = emb_from_rec(model, model_params, f'output/test_rec_heads')

    print(emb)
