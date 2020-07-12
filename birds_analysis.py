import torch
import numpy as np
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd
from birds_utils import Dataset

def multiclass_metrics(y_pred, y_test):
    if len(y_test.shape)>1:
        y_test = y_test.argmax(axis=1)
    y_pred_softmax = torch.softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    TP = correct_pred.sum().item()
    total = len(correct_pred)
    return TP, total

def get_F1_micro(TP, FP, FN):
    return TP.sum()/(TP.sum() + 0.5*(FP.sum() + FN.sum())).item()
    
def multilabel_metrics(y_pred, y_test, p_thres=0.5):
    # f1_score(y.numpy(), y_pred.detach().numpy()>0, average='micro', zero_division='warn')
    thres = -np.log(1/p_thres - 1)
    total = len(y_pred)
    positives = 1*(y_pred > thres)
    TP = (positives * y_test).sum(axis=0)
    FP = (positives * (1 - y_test)).sum(axis=0)
    FN = ((1 - positives) * y_test).sum(axis=0)
    micro_F1 = get_F1_micro(TP, FP, FN)
    return TP, FP, FN, micro_F1, total

def get_class_dataset(val_files, selected_cl, duration, sr, min_std, params, N=160, multilabel=False):
    classes_files = {}
    for f in val_files:
        cl = f.split('/')[-2]
        if cl not in classes_files:
            classes_files[cl] = []
        classes_files[cl].append(f)

    files_to_sample = classes_files[selected_cl]*N
    classes = list(classes_files.keys())

    validation_set = Dataset(files_to_sample, list(classes_files.keys()), chunk_seconds=duration, sr=sr, min_std=min_std, multilabel=multilabel)
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
    return X, fr, to

def analyze_audio(model, file, classes, sr, duration, min_std, device):
    cl = file.split('/')[-2]
    cl_idx = np.where(classes == cl)[0][0]
    audio = np.load(file)
    audio_sample, fr, to = sample_audio_clip(audio, sr*duration, min_std)
    spec, y_pred = model(torch.tensor(audio_sample.reshape(1, 1, -1)).float().to(device))
    y_pred_softmax = torch.sigmoid(y_pred)
    print('p:', y_pred_softmax[0][cl_idx].item())
    plt.imshow(np.flipud(spec.reshape(*spec.shape[1:]).cpu().detach().numpy()), cmap='gray')
    plt.title(cl)
    plt.show()
    t = np.linspace(0, len(audio)-1, len(audio))/sr
    plt.plot(t, audio)
    plt.show()
    display(ipd.Audio(audio_sample, rate=sr))
    return audio_sample, fr/sr, to/sr

def plot_librosa_stft(audio_sample, window_size, top_db=80.0):
    plt.imshow(np.flipud(librosa.amplitude_to_db(np.abs(librosa.stft(audio_sample, n_fft=window_size)), top_db=top_db)), cmap='gray')