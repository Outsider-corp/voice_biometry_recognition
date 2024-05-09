import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(audio_file, target_sample_rate):
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != target_sample_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                            new_freq=target_sample_rate)
        waveform = resample_transform(waveform)
    return waveform, target_sample_rate


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


def preprocess_audio(audio_file, target_sample_rate):
    waveform, sample_rate = load_audio(audio_file, target_sample_rate)
    voiced_waveform = apply_vad(waveform, sample_rate)
    return voiced_waveform
