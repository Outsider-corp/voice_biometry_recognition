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
with open(f'voice_params/v4_voice_params_50pers_30vox_mfcc.pkl', 'rb') as f:
    voice_params_mfcc = pickle.load(f)
dataset = PreTrainDataset(voice_params_mfcc)

# Создание DataLoader
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=PreTrainDataset.collate_fn)

model = DeepSpeaker.DeepSpeakerModel(include_softmax=True, num_classes=100)
model, stat = pre_train_model(model, dataloaders=train_loader, epoches=15,
                              lr=0.001)
torch.save(model.state_dict(),
           os.path.join('models', f'pre_trained_v4a.pth'))
json.dump(stat, open(os.path.join('models', f'pre_trained_v4a.json'), 'w'))
