import main_project
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    filename1 = r'voices/__audio1.wav'
    filename2 = r'voices/__audio2.wav'
    duration1 = input(
        "Введите длительность первого сообщения (Если указан 0 или '' - пропускается):")

    if duration1 and float(duration1):
        print('Запись начата...')
        main_project.record_audio(filename1, float(duration1))
        print('Запись закончена!')

    duration2 = input(
        "Введите длительность второго сообщения (Если указан 0 или '' - пропускается):")

    if duration2 and float(duration2):
        print('Запись начата...')
        main_project.record_audio(filename2, float(duration2))
        print('Запись закончена!')

    main_project.compare_audio(filename1, filename2)

if __name__ == '__main__':
    main()
