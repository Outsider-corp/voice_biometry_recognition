import os
import pickle
import json

import torch
from MD.parts.datasets import create_pair_dataset, create_triplets_dataset
from MD.parts.train_funcs import train_model_pair, train_model_triplet
from speechnas.network.dtdnnss_base import DtdnnssBase
from speechnas.network.dtdnnss_searched import DtdnnssBase_v1

os.chdir(r'../')
os.chdir(r'../')

model_params = {'persons_count': 100,
                'max_voices': 5,
                'mfcc_count': 20,
                'batch_size': 16,
                'target_sr': 16000,
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 64,
                'lr': 0.001,
                'margin_triplet': 0.3,
                'epoches': 50}

with open(f'voice_params/v7_voice_params_100pers_5vox_mfcc.pkl', 'rb') as f:
    voice_params_mfcc = pickle.load(f)

ver = '008'
model_name = f'DTDNNSS_100p_5vox_{ver}'
with open(os.path.join('models', f'params_{model_name}.json'), 'w') as f:
    json.dump(model_params, f)
model = DtdnnssBase(num_class=3000, feature_dim=128,
                    in_channels=model_params['mfcc_count']).float()
stat = None

dataloaders = create_triplets_dataset(voice_params_mfcc, model_params, 2)
model, stat = train_model_triplet(model, dataloaders,
                                  epoches=model_params['epoches'],
                                  save_name=model_name,
                                  lr=model_params['lr'],
                                  margin=model_params['margin_triplet'],
                                  start_epo=0,
                                  stat=stat)
torch.save(model.state_dict(), os.path.join('models', f'{model_name}___end.pth'))
