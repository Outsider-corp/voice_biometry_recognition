import os
import pickle
import json

import torch
from MD.parts.DeepSpeaker import DeepSpeakerModel
from MD.parts.datasets import create_pair_dataset, create_triplets_dataset
from MD.parts.train_funcs import train_model_pair, train_model_triplet

os.chdir(r'D:\Py_Projects\neuro')

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

# with open(
#         f'voice_params/voice_params_{model_params["persons_count"]}pers'
#         f'_{model_params["max_voices"]}vox_{model_params["mfcc_count"]}mfcc_'
#         f'{str(model_params["segment_length"]).replace(".", ",")}segml.pkl',
#         'rb') as f:
#     voice_params_mfcc = pickle.load(f)

with open(f'voice_params/v1_voice_params_100pers_10vox_mfcc.pkl', 'rb') as f:
    voice_params_mfcc = pickle.load(f)

ver = '011'
model_name = (f'DeepSpeaker_{model_params["persons_count"]}pers'
              f'_{model_params["max_voices"]}vox_{model_params["mfcc_count"]}'
              f'mfcc_tripl_{ver}')
with open(os.path.join('models', f'params_{model_name}.json'), 'w') as f:
    model_params.update({"epoches": 100})
    json.dump(model_params, f)
model = DeepSpeakerModel().float()
stat = None
# model.load_state_dict(torch.load(os.path.join('models', f'DeepSpeaker_100pers_5vox_20mfcc_tripl_002_epo20.pth')))
# stat = json.load(open(os.path.join('models', f'DeepSpeaker_100pers_5vox_20mfcc_tripl_002_epo20.json')))
dataloaders = create_triplets_dataset(voice_params_mfcc, model_params, 2)
model, stat = train_model_triplet(model, dataloaders,
                                  epoches=model_params['epoches'],
                                  save_name=model_name,
                                  lr=model_params['lr'],
                                  margin=model_params['margin_triplet'],
                                  start_epo=0,
                                  stat=stat)
torch.save(model.state_dict(), os.path.join('models', f'{model_name}___end.pth'))
