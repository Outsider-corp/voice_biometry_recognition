import json
import os
import pickle
from dotenv import load_dotenv

from MD.get_person_voice import get_audio_for_id, get_features_from_file
from MD.parts.torchaudio_clear import preprocess_audio_from_file, compute_mfcc_features

os.chdir(r'D:\Py_Projects\neuro')

load_dotenv()
data_path = r'{}'.format(os.environ['DATASET_PATH'])

# Получение данных
id_list = os.listdir(data_path)

model_params = {'persons_count': 50,
                'max_voices': 5,
                'mfcc_count': 20,
                'batch_size': 16,
                'target_sr': 16000,
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 40,
                'lr': 0.001,
                'margin_triplet': 0.3}
params_name = 'v8'
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
        mfcc, spectro = get_features_from_file(file, model_params, mfcc=True, spectrogram=True)
        person_params_mfcc.append(mfcc)
        person_params_spectro.append(spectro)
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
                       f'{params_name}_{pickle_file}_mfcc.pkl'), 'wb') as f:
    pickle.dump(voice_params_mfcc, f)
with open(os.path.join('voice_params', f'{params_name}_{pickle_file}_spectro.pkl'), 'wb') as f:
    pickle.dump(voice_params_spectro, f)

json.dump(model_params,
          open(os.path.join('voice_params', f'{params_name}_{pickle_file}.json'), 'w'))
