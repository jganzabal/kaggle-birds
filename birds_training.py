import torch
import librosa
import numpy as np
import time
from datetime import timedelta
from birds_filters import get_ambient_noise

def get_fourier_weights_for_mel(window_size):
    frec = np.linspace(-window_size//2, 0, window_size//2+1)
    time = np.linspace(0, window_size-1, window_size)
    hanning_window = np.hanning(window_size)

    filters_cos = []
    filters_sin = []
    for f in frec:
        filters_cos.append(np.cos(2*np.pi*f*time/window_size))
        filters_sin.append(np.sin(2*np.pi*f*time/window_size))
    filters_cos = np.array(filters_cos)[::-1]*hanning_window
    filters_sin = np.array(filters_sin)[::-1]*hanning_window
    return filters_cos, filters_sin


class MelSpectrogramNet(torch.nn.Module):
    def __init__(self, window_size = 2048, init_fourier=True, train_fourier=False, init_mel=True, train_mel=False, sr=22050, n_mels=128, hop_size=256, amin=1e-10, top_db=80.0, fmin=0.0, fmax=None, in_db=True):
        super(MelSpectrogramNet, self).__init__()
        self.in_db = in_db
        self.amin = amin
        self.top_db = top_db
        kernel_size = window_size
        stride = hop_size
        filters = kernel_size//2 + 1
        mel_filters = librosa.filters.mel(sr, kernel_size, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.cos = torch.nn.Conv1d(1, filters, kernel_size, stride=stride, bias=False)
        self.sin = torch.nn.Conv1d(1, filters, kernel_size, stride=stride, bias=False)

        if init_fourier:
            cos_weights, sin_weights = get_fourier_weights_for_mel(window_size)
            self.cos.weight.data = torch.from_numpy(cos_weights.reshape(cos_weights.shape[0], 1, cos_weights.shape[1])).float()
            self.sin.weight.data = torch.from_numpy(sin_weights.reshape(sin_weights.shape[0], 1, sin_weights.shape[1])).float()
            
        list(self.cos.parameters())[0].requires_grad = train_fourier
        list(self.sin.parameters())[0].requires_grad = train_fourier
            

        self.mel_filter = torch.nn.Conv1d(mel_filters.shape[1], mel_filters.shape[0], 1, bias=False)

        if init_mel:
            self.mel_filter.weight.data[:,:,0] = torch.from_numpy(mel_filters)
        list(self.mel_filter.parameters())[0].requires_grad = train_mel

    def forward(self, x):
        stft = self.cos(x)**2 + self.sin(x)**2
        mel_out = self.mel_filter(stft)
        if self.in_db:
            x_spec = 10.0 * torch.log10(torch.clamp(mel_out, min=self.amin))
            x_spec = torch.clamp(x_spec, min=x_spec.max().item() - self.top_db)
            x_spec = (x_spec + 25)/80
            return x_spec
        else:
            return mel_out
    
def resnet_BW(resnet18):
    resnet18.conv1.in_channels = 1
    resnet18.conv1.weight.data = resnet18.conv1.weight.data.sum(axis=1).reshape(64, 1, 7, 7)
    return resnet18

class BirdsNet(torch.nn.Module):
    def __init__(self, window_size=2048, hop_size=256, n_mels=128, sr=22050, pretrained=True, n_classes=264, resnet_type='resnet18', init_fourier=True, init_mel=True, train_fourier=False, train_mel=False, amin=1e-10, top_db=80.0, mel_db=True, a=-1.2):
        super(BirdsNet, self).__init__()
        if resnet_type=='resnet18':
            linear_inp = 512
        else:
            linear_inp = 2048
        self.mel_spectrogram = MelSpectrogramNet(
            window_size=window_size,
            init_fourier=init_fourier,
            init_mel=init_mel,
            train_fourier=train_fourier,
            train_mel=train_mel,
            sr=sr,
            n_mels=n_mels,
            hop_size=hop_size,
            amin=amin,
            top_db=top_db,
            in_db=mel_db
        )
        self.mel_db = mel_db
        if not mel_db:
            self.a = torch.nn.Parameter(torch.tensor([a]))
        
        self.bn1 = torch.nn.BatchNorm2d(1)
        model_resnet = torch.hub.load('pytorch/vision:v0.6.0', resnet_type, pretrained=pretrained)
        model_resnet_BW = resnet_BW(model_resnet)
        self.resnet = torch.nn.Sequential(*list(model_resnet_BW.children())[:-1])
        self.conv_out = torch.nn.Conv2d(linear_inp, n_classes, 1)
        
    
    def forward(self, x):
        mel_spectrogram = self.mel_spectrogram(x)
        if not self.mel_db:
            mel_spectrogram = torch.pow(mel_spectrogram, torch.sigmoid(self.a))
        mel_spectrogram = mel_spectrogram.reshape(-1, 1, *mel_spectrogram.shape[1:])
        mel_spectrogram_normalized = self.bn1(mel_spectrogram)
        x = self.resnet(mel_spectrogram_normalized)
        x = self.conv_out(x).flatten(start_dim=1)
        return mel_spectrogram_normalized, x
    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, classes, std_stats, chunk_seconds, bytes_per_sample, sr, multilabel=True, add_noise=False, add_ambient_noise=False, max_tries = 10):
        'Initialization'
        self.bytes_per_sample = bytes_per_sample
        self.std_stats = std_stats
        self.add_ambient_noise = add_ambient_noise
        self.list_IDs = list_IDs
        self.classes = classes
        self.n_classes = len(classes)
        self.sr = sr
        self.chunk_samples = chunk_seconds * sr
        self.classes_dict = {cl:i for i, cl in enumerate(self.classes)}
        self.multilabel = multilabel
        self.add_noise = add_noise
        self.chunk_seconds = chunk_seconds
        self.stats = {
            'noise': 0,
            'noise+signal': 0,
            'signal': 0,
            'passband_noise': 0,
            'short_file': 0
        }
        self.max_tries = max_tries

    def sample_audio_clip(self, clip):
        fr = int(np.random.rand(1)*(len(clip)-self.chunk_samples))
        to = fr + self.chunk_samples
        x = clip[fr:to]
        return x, fr, to
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get_normal_noise(self, std=1):
        y = torch.zeros(self.n_classes)
        X = torch.from_numpy(np.random.normal(0, std, (1, self.chunk_samples))).float()
        return X, y
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        if ID not in self.std_stats:
            # Failed to load file stats
            self.stats['noise'] = self.stats['noise'] + 1
            return self.get_normal_noise()
        
        file_stats = self.std_stats[ID]
        std_thres = file_stats['std_mean']
        stds_min = file_stats['std_min']
        audio_size = file_stats['size']
        
        # File to short return random noise
        if audio_size < self.chunk_samples:
            # Audio size too short
            self.stats['noise'] = self.stats['noise'] + 1
            self.stats['short_file'] = self.stats['short_file'] + 1
            return self.get_normal_noise()
        
        # Load data and get label
        X = get_audio_chunk(ID, audio_size, self.bytes_per_sample, duration=self.chunk_seconds, sr=self.sr)
        std = X.std()
        
        if len(X) == 0:
            # Failed to load audio
            self.stats['noise'] = self.stats['noise'] + 1
            return self.get_normal_noise()
        
        n_tries = 0

        while (std < std_thres) and (n_tries<self.max_tries):
            n_tries += 1
            new_X = get_audio_chunk(ID, audio_size, self.bytes_per_sample, duration=self.chunk_seconds, sr=self.sr)
            new_std = new_X.std()
            if new_std>std:
                std = new_std
                X = new_X

        # Add random noise half of the time
        if np.random.randint(2) == 1 and self.add_noise:
            # Half of the time add white noise
            X = X + np.random.normal(0, 1-stds_min, len(X))
            self.stats['noise+signal'] = self.stats['noise+signal'] + 1
        
        # Reshape for 1 channel 1D CNN (channel first)
        X = torch.from_numpy(X.reshape(1, -1)).float()
        
        # only noise 1/n_classes
        if np.random.rand()<1/self.n_classes and self.add_noise:
            self.stats['noise'] = self.stats['noise'] + 1
            y = torch.zeros(self.n_classes)
            X = torch.from_numpy(np.random.normal(0, 1, X.shape)).float()
        elif self.multilabel:
            self.stats['signal'] = self.stats['signal'] + 1
            y = torch.zeros(self.n_classes)
            y[self.classes_dict[ID.split('/')[-2]]] = 1
        else:
            y = torch.tensor(self.classes_dict[ID.split('/')[-2]])
            
        if np.random.randint(2) == 1 and self.add_ambient_noise:
            ambient_noise = get_ambient_noise(X.shape[1], self.sr)
            self.stats['passband_noise'] = self.stats['passband_noise'] + 1
            if not np.isnan(ambient_noise.sum()):
                X = X + torch.from_numpy(ambient_noise.reshape(1, -1)).float()
        return X, y
    
def multilabel_metrics(y_pred, y_test, p_thres=0.5):
    # f1_score(y.numpy(), y_pred.detach().numpy()>0, average='micro', zero_division='warn')
    thres = -np.log(1/p_thres - 1)
    total = len(y_pred)
    positives = 1*(y_pred > thres)
    TP = (positives * y_test).sum(axis=0)
    FP = (positives * (1 - y_test)).sum(axis=0)
    FN = ((1 - positives) * y_test).sum(axis=0)
    T_total = (y_test > thres).sum()
    micro_F1 = get_F1_micro(TP, FP, FN)
    return TP, FP, FN, micro_F1, total, T_total

def get_F1_micro(TP, FP, FN):
    return TP.sum()/(TP.sum() + 0.5*(FP.sum() + FN.sum())).item()
    
def validate(model, dgen_val, criterion, device, metrics_func=multilabel_metrics):
    model.eval()  
    with torch.no_grad():
        running_loss = 0.0
        TPs = 0
        FPs = 0
        FNs = 0
        total_predictions = 0
        T_totals = 0
        batches_per_epoch = len(dgen_val)
        for i, (X, y) in enumerate(dgen_val):
            inputs, labels = X.to(device), y.to(device)
            _, y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            TP, FP, FN, micro_F1, total, T_total = metrics_func(y_pred, labels)
            TPs = TPs + TP.sum()
            FPs = FPs + FP.sum()
            FNs = FNs + FN.sum()
            T_totals = T_totals + T_total
            total_predictions = total_predictions + total
            
            running_loss = running_loss + loss
            
            avg_loss = running_loss/(i+1)
            avg_F1 = get_F1_micro(TPs, FPs, FNs)
            avg_acc = TPs/T_totals
            print(f'\r{i+1}/{batches_per_epoch} - val loss: {avg_loss}, val F1 micro: {avg_F1} val acc: {avg_acc}', end='')
    model.train()
    return avg_loss.detach().item(), avg_F1.detach().item(), avg_acc.detach().item()

def get_audio_chunk(filename, size, bytes_per_sample, duration=5, sr=22050, start=None):
    extention = filename.split('.')[-1]
    if bytes_per_sample == 8:
        dtype = 'float64'
    else:
        dtype = 'float32'
    chunk_samples = duration*sr
    
    # size = os.fstat(f.fileno()).st_size // bytes_per_sample
    if start is None:
        if chunk_samples == size:
            start = 0
        else:
            start = np.random.randint(size - chunk_samples)
    if extention == 'bin':
        f = open(filename, 'rb')
        f.seek(start*bytes_per_sample)
        audio_chunk = np.frombuffer(f.read(chunk_samples*bytes_per_sample), dtype=dtype)
        f.close()
        if len(audio_chunk) != chunk_samples:
            print(filename, start, chunk_samples, len(audio_chunk))
            return np.array([])
    elif extention == 'wav':
        offset = start/sr
        audio_chunk, sr = librosa.load(filename, sr=sr, mono=True, offset=offset, duration=duration)
    
    
    return audio_chunk.copy()

def train_model(model, dataset, validation_generator, criterion, optimizer, name, device, metrics_func=multilabel_metrics, epochs=1, best_metric = np.inf):
    model.train()
    batches_per_epoch = len(dataset)
    losses = []
    F1s = []
    val_losses = []
    val_F1s = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        TPs = 0
        FPs = 0
        FNs = 0
        total_predictions = 0
        T_totals = 0
        model.train()
        start_time = time.time()
        for i, (X, y) in enumerate(dataset):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = X.to(device), y.to(device)
            # (1) Initialise gradients
            optimizer.zero_grad()
            # (2) Forward pass
            _, y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            running_loss = running_loss + loss
            # (3) Backward
            loss.backward()
            # (4) Compute the loss and update the weights
            optimizer.step()
            TP, FP, FN, micro_F1, total, T_total = metrics_func(y_pred, labels)
            TPs = TPs + TP.sum()
            FPs = FPs + FP.sum()
            FNs = FNs + FN.sum()
            total_predictions = total_predictions + total
            T_totals = T_totals + T_total
            
            avg_loss = running_loss/(i+1)
            avg_F1 = get_F1_micro(TPs, FPs, FNs)
            avg_acc = TPs/T_totals
            print(f'\r{epoch+1}/{epochs} - {i+1}/{batches_per_epoch} - loss: {avg_loss}, F1 micro: {avg_F1}, acc: {avg_acc}', end=', ')
        elapesed_time = (time.time() - start_time)
        print(f'elapesed_time: {timedelta(seconds=elapesed_time)}')
        losses.append(avg_loss.item())
        F1s.append(avg_F1.item())
        avg_loss, avg_F1, avg_acc = validate(model, validation_generator, criterion, device, metrics_func=multilabel_metrics)
        val_losses.append(avg_loss)
        val_F1s.append(avg_F1)
        if avg_loss<best_metric:
            best_metric = avg_loss
            print()
            print('Best model saved')
            torch.save(model.state_dict(), f'{name}_{int(best_metric*100000 + 0.5)/100000}.pth')
        else:
            print()
        print('--------------------------------------------------------------------------')
    return epoch+1, losses, F1s, val_losses, val_F1s