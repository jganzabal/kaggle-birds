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

def get_std_thres(clip, sr, seconds_to_analyze = 1):
    samples_to_analyse = int(sr*seconds_to_analyze)
    croped_audio = samples_to_analyse*(len(clip)//samples_to_analyse)
    stds = clip[:croped_audio].reshape(-1, samples_to_analyse).std(axis=1)
    std_thres = np.mean(stds)
    return std_thres, np.min(stds), np.max(stds)

def analyze_audio(model, audio, classes, sr, duration, min_std, device, fr=-1, add_noise=False):
    
    std_thres, std_min, std_max = get_std_thres(audio, sr)
    
    
    if min_std is None:
        min_std = std_thres
    if fr>=0:
        fr = int(fr*sr)
        to = int(fr + duration*sr)
        audio_sample = audio[fr:to]
    else:
        audio_sample, fr, to = sample_audio_clip(audio, sr*duration, min_std)
        
    sample_std = audio_sample.std()
    
    if add_noise:
        audio_sample = audio_sample + np.random.normal(0, 1-std_min, len(audio_sample))
    
    
    spec, y_pred = model(torch.tensor(audio_sample.reshape(1, 1, -1)).float().to(device))
    
    y_pred_softmax = torch.sigmoid(y_pred)
    
    sorted_idxs = y_pred_softmax[0].argsort().detach().numpy()[::-1]
    for sid in sorted_idxs:
        print(classes[sid], int(y_pred_softmax[0][sid].item()*100)/100, end=', ')
        
    f, ax = plt.subplots(1, 3, figsize=(30,4))
    ax[0].imshow(np.flipud(spec.reshape(*spec.shape[1:]).cpu().detach().numpy()), cmap='gray')
    t = np.linspace(0, len(audio)-1, len(audio))/sr
    ax[1].plot(t, audio)
    min_val = min(audio)
    max_val = max(audio)
    ax[1].vlines(fr/sr, min_val, max_val)
    ax[1].vlines(to/sr, min_val, max_val)
    
    ax[2].plot(t[:len(audio_sample)], audio_sample, alpha=0.5)
    ax[2].hlines(std_thres, 0, t[len(audio_sample)])
    plt.show()
    display(ipd.Audio(audio_sample, rate=sr))
    print(f'from: {fr/sr}, to: {to/sr}, std_thres: {std_thres}, std_min: {std_min}, clip_std:{sample_std}')
    return audio_sample, fr/sr, to/sr

def plot_librosa_stft(audio_sample, window_size, top_db=80.0):
    plt.imshow(np.flipud(librosa.amplitude_to_db(np.abs(librosa.stft(audio_sample, n_fft=window_size)), top_db=top_db)), cmap='gray')