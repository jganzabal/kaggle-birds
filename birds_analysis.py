import torch
import numpy as np
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
from birds_utils import Dataset

def get_class_dataset(val_files, selected_cl, duration, sr, min_std, params, N=160):
    classes_files = {}
    for f in val_files:
        cl = f.split('/')[-2]
        if cl not in classes_files:
            classes_files[cl] = []
        classes_files[cl].append(f)

    files_to_sample = classes_files[selected_cl]*N
    classes = list(classes_files.keys())

    validation_set = Dataset(files_to_sample, list(classes_files.keys()), chunk_seconds=duration, sr=sr, min_std=min_std)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    return validation_generator, classes_files

def sample_audio_clip(clip, chunk_samples, min_std):
    def sample_audio(clip):
        fr = int(np.random.rand(1)*(len(clip)-chunk_samples))
        to = fr + chunk_samples
        x = clip[fr:to]
        return x, fr, to
    X, fr, to = sample_audio(clip)
    std = X.std()
    while std < min_std:
        X, fr, to = sample_audio(clip)
        std = X.std()
    return X

def analyze_audio(model, file, classes, sr, duration, min_std):
    cl = file.split('/')[-2]
    cl_idx = np.where(classes == cl)[0][0]
    audio = np.load(file)
    audio_sample = sample_audio_clip(audio, sr*duration, min_std)
    spec, y_pred = model(torch.tensor(audio_sample.reshape(1, 1, -1)).float().cuda())
    y_pred_softmax = torch.softmax(y_pred, dim = 1)
    print('p:', y_pred_softmax[0][cl_idx].item())
    plt.imshow(np.flipud(spec.reshape(*spec.shape[1:]).cpu().detach().numpy()), cmap='gray')
    plt.title(cl)
    plt.show()
    display(ipd.Audio(audio_sample, rate=sr))
    return audio_sample

def plot_librosa_stft(audio_sample, window_size, top_db=80.0):
    plt.imshow(np.flipud(librosa.amplitude_to_db(np.abs(librosa.stft(audio_sample, n_fft=window_size)), top_db=top_db)), cmap='gray')