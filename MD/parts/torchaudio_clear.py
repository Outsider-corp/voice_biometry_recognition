import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import butter, lfilter
import numpy as np
import sounddevice as sd


def load_audio(waveform, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                            new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
    return waveform, sample_rate


def record_audio(duration, sample_rate=16000, channels=1, save_path: str = None):
    # Записываем аудио с микрофона
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels,
                       dtype='float32')
    print(f'{duration} seconds speech...')
    sd.wait()  # Ждем окончания записи

    # Преобразуем numpy массив в тензор PyTorch
    waveform = torch.tensor(recording).transpose(0, 1)

    if save_path is not None:
        torchaudio.save(save_path, waveform, sample_rate)
    return waveform, sample_rate


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


def filter_tensor(tensor, lowcut, highcut, fs, order=5):
    data_np = tensor.detach().numpy()
    filtered_np = butter_bandpass_filter(data_np, lowcut, highcut, fs, order=order)
    normalize_np = normalize_audio(filtered_np)
    filtered_tensor = torch.from_numpy(normalize_np).float()
    return filtered_tensor


def compute_mfcc_features(waveform, sample_rate, win_length, hop_length, n_mels, n_fft, n_mfcc):
    # Параметры MFCC: количество коэффициентов, длина окна, размер шага и число мел-фильтров
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,  # Количество MFCC коэффициентов
        melkwargs={
            'n_fft': n_fft,
            'n_mels': n_mels,  # Обычно для MFCC используют больше мел-фильтров
            'win_length': win_length,
            'hop_length': hop_length
        }
    )
    mfcc_features = mfcc_transform(waveform)
    norm_mfcc = normalize_features(mfcc_features)
    return norm_mfcc


def normalize_features(features):
    mean = features.mean()
    std = features.std()
    normalized_features = (features - mean) / std
    return normalized_features


def apply_vad(waveform, sample_rate):
    vad_transform = torchaudio.transforms.Vad(sample_rate=sample_rate)
    voiced_frames = vad_transform(waveform)
    return voiced_frames


def preprocess_audio_from_file(audio_file, target_sample_rate, lowcut=300, highcut=3400, order=5):
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform, sample_rate = load_audio(waveform, sample_rate, target_sample_rate)

    waveform_filtered = filter_tensor(waveform, lowcut, highcut, sample_rate, order)

    voiced_waveform = apply_vad(waveform_filtered, sample_rate)
    return voiced_waveform


def preprocess_audio_recording(target_sample_rate, lowcut=300, highcut=3400,
                               order=5, save_path: str = None):
    waveform, sample_rate = record_audio(7, save_path=save_path)
    waveform, sample_rate = load_audio(waveform, sample_rate, target_sample_rate)

    waveform_filtered = filter_tensor(waveform, lowcut, highcut, sample_rate, order)

    voiced_waveform = apply_vad(waveform_filtered, sample_rate)
    return voiced_waveform
