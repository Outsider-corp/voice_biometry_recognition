import os
from typing import List, Dict, Union

import librosa
from librosa.feature import mfcc
import sounddevice as sd
import numpy as np
from math import floor
from scipy.signal.windows import tukey, gaussian, kaiser
from scipy.signal import ellip, lfilter
from scipy.io.wavfile import write
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def compare_audio(filename1: str, filename2: str):
    mfcc_count = 24
    audio1, fs1 = clear_audio(filename1)
    audio2, fs2 = clear_audio(filename2)

    # librosa.display.waveshow(audio1, sr=fs1)
    # plt.show()

    mfccs1 = mfcc(y=audio1, sr=fs1, n_mfcc=mfcc_count)
    mfccs2 = mfcc(y=audio2, sr=fs2, n_mfcc=mfcc_count)

    hist1 = create_histogram(mfccs1)
    hist2 = create_histogram(mfccs2)

    distances = [euclidean_distance(arr1, arr2) for arr1, arr2 in zip(hist1, hist2)]

    # Усредняем расстояния
    res = np.max(distances)

    if res < 1:
        print(f'Говорит один человек!\nevc_dist = {res}')
    else:
        print(f'Это разные люди...\nevc_dist = {res}')

    # plt.figure(figsize=(10, 6))
    # # librosa.display.specshow(mfccs, x_axis='time')
    # plt.plot(mfccs1)
    # # plt.colorbar()
    # plt.title('MFCC коэффициенты')
    # plt.xlabel('Время (сек)')
    # plt.ylabel('MFCC коэффициенты')
    # plt.show()


def main():
    mfcc_count = 24
    voices = {
        r'voices/rom1.ogg': 'rom',
        r'voices/rom2.ogg': 'rom',
        r'voices/rom3.ogg': 'rom',
        r'voices/tg2.ogg': 'rom',
        r'voices/tg3.ogg': 'rom',
        r'voices/rad1.ogg': 'rad',
        r'voices/rad2.ogg': 'rad',
        r'voices/rad3.ogg': 'rad',
        r'voices/rad4.ogg': 'rad',
    }

    y_tags = {'rom': 1, 'rad': 2}
    data = [np.concatenate(process_audio(voice, mfcc_count), axis=0) for voice in voices]

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        [y_tags[val] for val in voices.values()],
                                                        test_size=0.2,
                                                        random_state=42)

    # X_train_flattened = np.array(X_train).reshape(X_train.shape[0], -1)
    # X_test_flattened = np.array(X_test).reshape(X_test.shape[0], -1)

    # for x in data:
    #     if sum(np.isnan(x)):
    #         print(1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    new_audio = r'voices/rad1.ogg'

    test_audio = np.concatenate(process_audio(new_audio, mfcc_count))
    test_audio = scaler.fit_transform([test_audio])

    predicted_class = model.predict(test_audio)
    print(f'Predicted: {predicted_class}')


def main2():
    audio_file = r'D:\University\Диссерт\test_data_01\id10001\1zcIwhmdeo4\00001.wav'
    clear_audio(audio_file, r'00001.wav')

    # audio, fs = get_audio(audio_file)
    # audio_supp = outlier_suppression(audio, fs)
    #
    # audio_clear = remove_low_freq_noice(audio_supp, fs)
    #
    # signal, noise = get_power_arrays(audio_clear, fs)
    # signal, noise = filter_signals(signal, noise)
    # phrases = get_phrases(signal, noise, threshold=0.1, length_threshold=16)
    # phrases_new = extend_voice(phrases, audio_clear, fs)
    #
    # voice = create_output_array(audio_clear, phrases_new)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].plot(audio)
    # ax[0][1].plot(audio_supp)
    # ax[1][0].plot(audio_clear)
    # plt.show()
    # old_name = os.path.basename(audio_file)
    # new_name = f'output/{old_name.split(".")[0]}_3.wav'
    # write(new_name, fs, (voice * 32767).astype(np.int16))


def record_audio(output_path: str, duration: Union[float, int], sample_rate: int = 44100):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    write(output_path, sample_rate, audio_data.flatten())


def process_audio(file_path: str, mfcc_count: int = 24):
    audio, fs = clear_audio(file_path)
    mfccs = mfcc(y=audio, sr=fs, n_mfcc=mfcc_count)
    return create_histogram(mfccs)


def resize_array(arr, new_shape):
    new_arr = np.zeros(new_shape)
    new_arr[:arr.shape[0], :arr.shape[1]] = arr
    return new_arr


def euclidean_distance(x1, x2):
    distance = np.linalg.norm(x1 - x2)
    return distance


def create_histogram(mfccs: np.ndarray):
    histograms = []
    for i in range(10):
        histograms.append(np.histogram(mfccs[:, i], bins=21, density=True, range=(-10, 10))[0])
    for i in range(10, mfccs.shape[0]):
        histograms.append(np.histogram(mfccs[:, i], bins=21, density=True, range=(-10, 10))[0])
    return histograms


def clear_audio(input_path: str, output_path: str = None):
    buffer_duration = 0.01
    audio, fs = get_audio(input_path)
    audio_supp = outlier_suppression(audio, fs)
    audio_clear = remove_low_freq_noice(audio_supp, fs)

    signal, noise = get_power_arrays(audio_clear, fs, buffer_duration=buffer_duration)
    signal, noise = filter_signals(signal, noise)
    phrases = get_phrases(signal, noise, threshold=0.09)
    merge_small_phrases(phrases,
                        length_threshold=16,
                        threshold=0.09)
    phrases2 = update_phrases(phrases, threshold_power=0.1)
    phrases_new = extend_voice(phrases2, audio_clear, fs, buffer_duration=buffer_duration)

    voice = create_output_array(audio_clear, phrases_new)
    voice_compressed = compress_audio(voice, phrases_new)
    if output_path:
        write(output_path, fs, (voice_compressed * 32767).astype(np.int16))

    return voice_compressed, fs


def get_audio(filename: str, sec_start: float = 0, sec_stop: float = None):
    audio, fs = librosa.load(filename, sr=None, mono=True)

    frame_start = max(sec_start * fs, 1)
    frame_stop = min(sec_stop * fs, len(audio)) if sec_stop else len(audio)

    sec_start = (frame_start / fs) - fs
    sec_stop = frame_stop / fs

    # обрезка
    audio = audio[frame_start: frame_stop]
    # audio = np.mean(audio, 2)
    audio = audio - np.mean(audio)

    return audio, fs


def outlier_suppression(x, fs):
    buffer_duration = 0.1  # in seconds
    accumulation_time = 3  # in seconds
    overlap = 50  # in percent
    outlier_threshold = 7

    buffer_size = round(((1 + 2 * (overlap / 100)) * buffer_duration) * fs)
    sample_size = round(buffer_duration * fs)

    y = x.copy()
    cumm_power = 0
    amount_iteration = floor(len(y) / sample_size)

    for i in range(1, amount_iteration + 1):

        start_index = (i - 1) * sample_size
        start_index = max(round(start_index - (overlap / 100) * sample_size), 1)
        end_index = min(start_index + buffer_size, len(x))

        buffer = y[start_index: end_index]
        power = np.linalg.norm(buffer)

        if (accumulation_time / buffer_duration < i) and (
                outlier_threshold * cumm_power / i) < power:
            buffer = buffer * (1 - tukey(end_index - start_index + 1, alpha=0.8))
            y[start_index: end_index] = buffer

            cumm_power += cumm_power / i
            continue

        cumm_power += power

    return y


def remove_low_freq_noice(x: np.ndarray, fs: float, cutoff_freq: int = 30,
                          stopband_attenuation: int = 20,
                          filter_n: int = 3, rp: int = 1):
    b, a = ellip(N=filter_n, rp=rp, rs=stopband_attenuation,
                 Wn=cutoff_freq / fs * 2, btype='high',
                 analog=False, output='ba')

    new_signal = lfilter(b, a, x)

    return new_signal


def get_power_arrays(x: np.ndarray, fs: float, buffer_duration: float, overlap: float = 50):
    # buffer_duration = 10 * 10 ** (-3) # in seconds
    # overlap = 50 # in percent

    sample_size = round(buffer_duration * fs)
    buffer_size = round(((1 + 2 * (overlap / 100)) * buffer_duration) * fs)

    signal_power = np.zeros((floor(len(x) / sample_size), 1))
    noise_power = np.zeros((len(signal_power), 1))

    for i in range(len(signal_power)):
        start_index = i * sample_size
        start_index = max(round(start_index - (overlap / 100) * sample_size), 1)
        end_index = min(start_index + buffer_size, len(x))

        buffer = x[start_index: end_index]
        signal_power[i] = np.linalg.norm(buffer)
        noise_power[i] = np.linalg.norm(abs(np.diff(np.sign(buffer))))

    return signal_power, noise_power


def filter_signals(signal_power: np.ndarray, noise_power: np.ndarray):
    def base_value(array: np.ndarray):
        el, centers = np.histogram(array, bins=100)
        index_max = np.argmax(el)
        val = centers[index_max]
        return val

    # Фильтрация сигнала с использованием гауссовского окна (8)
    gw = gaussian(8, 2)
    signal_power = lfilter(gw, 1, signal_power)
    signal_power = signal_power - base_value(signal_power)
    signal_power = np.maximum(signal_power, 0)
    signal_power = signal_power / max(signal_power)

    # Фильтрация шума с использованием гауссовского окна (16)
    gw = gaussian(16, 2)
    noise_power = lfilter(gw, 1, noise_power)
    noise_power = noise_power - base_value(noise_power)
    noise_power = np.maximum(noise_power, 0)
    noise_power = np.max(noise_power) - noise_power
    noise_power = noise_power / np.max(noise_power)
    noise_power = 1 - noise_power

    return signal_power, noise_power


def get_phrases(signal_power, noise_power: np.ndarray, threshold: float):
    """Выделение фраз"""

    # Поиск конца фраз
    phrase_detection = np.zeros(np.size(signal_power))
    max_magn = 0
    for i in range(len(signal_power)):
        max_magn = max(max_magn, signal_power[i])
        if signal_power[i] < 2 * threshold * max_magn:
            phrase_detection[i] = 1
            max_magn = 0

    # Поиск начала фраз
    max_magn = 0
    for i in range(len(signal_power) - 1, -1, -1):
        max_magn = max(max_magn, signal_power[i])
        if signal_power[i] < 2 * threshold * max_magn:
            phrase_detection[i] = 1
            max_magn = 0

    # Обработка массива с выделенными фразами
    phrase_detection = -phrase_detection
    phrase_detection[0] = 0
    phrase_detection[-1] = 0
    edge_phrase = np.diff(phrase_detection)

    # Создание массива фраз
    phrases = []
    phrase_counter = -1
    for i in range(len(edge_phrase)):
        if edge_phrase[i] > 0:
            phrase_counter += 1
            phrases.append({'start': i})
        elif edge_phrase[i] < 0:
            if not phrases:
                continue
            start = phrases[phrase_counter]['start']
            phrases[phrase_counter]['end'] = i
            phrases[phrase_counter]['power'] = np.mean(signal_power[start: i + 1])
            phrases[phrase_counter]['noise_power'] = np.mean(noise_power[start: i + 11])

    phrases = [phrase for phrase in phrases if 'start' in phrase and 'end' in phrase]

    return phrases


def merge_small_phrases(phrases: List[Dict], length_threshold: int, threshold: float):
    i = 1
    while i < len(phrases) - 1:
        phrase_length = phrases[i]['end'] - phrases[i]['start']
        if length_threshold > phrase_length:
            lookup_power = phrases[i + 1]['power']
            behind_power = phrases[i - 1]['power']

            if abs(lookup_power - phrases[i]['power']) < abs(
                    behind_power - phrases[i]['power']):
                next_i = i + 1
            else:
                next_i = i - 1

            power_ratio = phrases[next_i]['power'] / phrases[i]['power']
            power_ratio = min(power_ratio, 1 / power_ratio)

            if power_ratio < threshold:
                i += 1
                continue

            min_index = min(phrases[i]['start'], phrases[next_i]['start'])
            max_index = max(phrases[i]['end'], phrases[next_i]['end'])

            phrase_next_length = phrases[next_i]['end'] - phrases[next_i]['start']

            phrases[next_i]['power'] = ((phrases[i]['power'] * phrase_length) + (
                    phrases[next_i]['power'] * phrase_next_length)) / (
                                               phrase_length + phrase_next_length)

            phrases[next_i]['noise_power'] = ((phrases[i]['noise_power'] * phrase_length) + (
                    phrases[next_i]['noise_power'] * phrase_next_length)) / (
                                                     phrase_length + phrase_next_length)

            phrases[next_i]['start'] = min_index
            phrases[next_i]['end'] = max_index

            phrases.pop(i)
        else:
            i += 1


def update_phrases(phrases: List[dict], threshold_power: float):
    i = 0

    while i < len(phrases):
        if threshold_power < phrases[i]['power']:
            phrases[i]['voice'] = 1
        elif ((phrases[i]['power'] > 0.2 * threshold_power) and
              (phrases[i]['power'] > 2 * phrases[i]['noise_power'])):
            phrases[i]['voice'] = 1
        else:
            phrases[i]['voice'] = 0

        if i != 0:
            if phrases[i - 1]['voice'] == phrases[i]['voice']:
                phrases[i - 1]['end'] = phrases[i]['end']
                phrases.pop(i)
                continue

        i += 1

    return phrases


def extend_voice(phrases, x: np.ndarray, fs: float, buffer_duration: float,
                 extend_time: float = 0.05):
    # Вычисление реальных индексов выборок
    sample_size = round(buffer_duration * fs)
    for i in range(len(phrases)):
        start_index = (phrases[i]['start'] - 1) * sample_size + 1
        end_index = phrases[i]['end'] * sample_size
        phrases[i]['start'] = start_index
        phrases[i]['end'] = end_index

    # FIXME Подумать над целесообразностью этого
    phrases[0]['start'] = 1
    phrases[-1]['end'] = len(x)

    # Расширение голоса во времени
    spread_voice = round(extend_time * fs)
    i = 1

    while i < len(phrases) - 1:
        if phrases[i]['voice']:
            phrases[i]['start'] -= spread_voice
            phrases[i]['end'] += spread_voice
        else:
            phrases[i]['start'] = phrases[i - 1]['end'] + 1
            phrases[i]['end'] = phrases[i + 1]['start'] - 1 - spread_voice

        # phrases[i]['start'] = max(phrases[i]['start'], 1)
        # phrases[i]['end'] = min(phrases[i]['end'], len(x))

        if phrases[i]['start'] < phrases[i]['end']:
            i += 1
        else:
            phrases[i - 1]['end'] = phrases[i]['end']
            phrases.pop(i)

    # Обновление границ первой и последней фраз
    phrases[0]['start'] = 1
    phrases[0]['end'] = phrases[1]['start'] - 1
    phrases[-1]['start'] = phrases[-2]['end'] + 1
    phrases[-1]['end'] = len(x) - 1

    return phrases


def create_output_array(x: np.ndarray, phrases: List[dict]):
    """Создание выходного массива выборок"""
    y = np.zeros_like(x)
    # phrase_space = np.zeros_like(x)

    betta = 3

    for i in range(len(phrases)):
        phrase_length = slice(phrases[i]['start'], phrases[i]['end'] + 1)
        if phrases[i]['voice']:
            y[phrase_length] = x[phrase_length] * kaiser(
                len(range(phrases[i]['start'], phrases[i]['end'] + 1)), betta)
            # phrase_space[phrase_length] = -1
        else:
            y[phrase_length] = np.zeros_like(x[phrase_length])

        # if phrases[i]['end'] > 0:
        #     phrase_space[phrases[i]['end']] = 0

    return y


def compress_audio(audio: np.ndarray, phrases: List[Dict]):
    # Удаление данных со значением 0
    for i in range(len(phrases) - 1, -1, -1):
        phrase_length = slice(phrases[i]['start'], phrases[i]['end'] + 1)
        if not phrases[i]['voice']:
            audio = np.delete(audio, phrase_length)
    return audio


if __name__ == '__main__':
    main2()
