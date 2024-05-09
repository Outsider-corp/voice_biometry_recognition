import json
import os
import pickle

from MD.parts import DeepSpeaker
from MD.parts.datasets import PreTrainDataset
from parts.train_funcs import pre_train_model
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

os.chdir(r'D:\Py_Projects\neuro')




# Создание экземпляра Dataset
with open(f'voice_params/v1_voice_params_100pers_10vox_mfcc.pkl', 'rb') as f:
    voice_params_mfcc = pickle.load(f)
dataset = PreTrainDataset(voice_params_mfcc)

# Создание DataLoader
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=PreTrainDataset.collate_fn)

model = DeepSpeaker.DeepSpeakerModel(include_softmax=True, num_classes=100)
model, stat = pre_train_model(model, dataloaders=train_loader, epoches=200,
                              lr=0.001)
torch.save(model.state_dict(),
           os.path.join('models', f'pre_trained_v1.pth'))
json.dump(stat, open(os.path.join('models', f'pre_trained_v1.json'), 'w'))
