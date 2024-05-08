from typing import Union, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


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


def remove_silence(audio, sr, threshold=20, frame_length=2048):
    # Use librosa's trim function with top_db as threshold
    audio, _ = librosa.effects.trim(audio, top_db=threshold, frame_length=frame_length,
                                    hop_length=512)
    return audio


def resample_audio(audio, sr, target_sr=16000):
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def segment_audio(audio, sr, segment_length=5) -> List[List[int]]:
    buffer = segment_length * sr
    segments = [audio[i:i + buffer] for i in range(0, len(audio), buffer)][:-1]
    return segments


def clear_audio(file_path: str, target_sr: int = 16000, segment_length: int = 5,
                save_full_audio_path: Union[str, None] = None) -> Tuple[
    np.ndarray, Union[List[List[int]], None]]:
    audio, sr = librosa.load(file_path, sr=None)
    # Предварительная обработка
    audio = butter_bandpass_filter(audio, 300, 3400, sr)
    audio = normalize_audio(audio)
    audio = remove_silence(audio, sr)
    audio, sr = resample_audio(audio, sr, target_sr=target_sr)
    if save_full_audio_path is not None:
        sf.write(f'{save_full_audio_path}.wav', audio, sr)
    segments = segment_audio(audio, sr, segment_length=segment_length) if segment_length else None
    return audio, segments


if __name__ == '__main__':
    # Загрузите ваш аудиофайл
    file_path = r'D:\University\Диссерт\test_data_01\id10001\1zcIwhmdeo4\00001.wav'
    clear_audio(file_path, 16000, segment_length=5)
