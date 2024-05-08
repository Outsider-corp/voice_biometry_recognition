from rnnoise_model.rnnoise_wrapper import RNNoise


dll_path = r'D:\Py_Projects\neuro\rnnoise_model\rnnoise.dll'
dll_path2 = r'D:\Py_Projects\neuro\rnnoise_model\librnnoise_default.so.0.4.1'

denoiser = RNNoise(f_name_lib=dll_path)

audio = denoiser.read_wav(r'D:\Py_Projects\neuro\test_denoise\0001.wav')
denoised_audio = denoiser.filter(audio)
denoiser.write_wav(r'D:\Py_Projects\neuro\test_denoise\0001_denoised_RNN.wav', denoised_audio)



# import ctypes
#
# dll = ctypes.CDLL(dll_path)
# dll2 = dll.rnnoise_create(None)
# # Получение списка экспортированных функций
# function_list = [func for func in dir(dll) if callable(getattr(dll, func))]
#
# # Вывод списка функций
# for func in function_list:
#     print(func)
