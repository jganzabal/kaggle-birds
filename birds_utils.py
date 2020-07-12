from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
from glob import glob
from shutil import copyfile
import os

def get_extentions(TRAIN_FOLDER):
    source_filenames = glob(TRAIN_FOLDER+'**/*', recursive=True)
    extentions = []
    for file in source_filenames:
        spl = file.split('.')
        if len(spl)>1:
            ext = spl[-1]
            if ext not in extentions:
                extentions.append(ext)
    return extentions

def audio_to_npy(TRAIN_FOLDER, TARGET_FOLDER, extentions, classes, target_sr = 22050):
    copied = 0
    existing = 0
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
    
    extentions = get_extentions(TRAIN_FOLDER)
    source_filenames = glob(TRAIN_FOLDER+'**/*', recursive=True)
    for i, file in enumerate(source_filenames):
        ext = file.split('.')[-1]
        splt = file.split('/')
        cl = splt[-2]
        name = splt[-1]
        if ext in extentions and cl in classes:
            dst_folder = TARGET_FOLDER  + cl + '/'
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
                
            dst_file = dst_folder  + name + '.npy'
            if not os.path.exists(dst_file):
                sound = AudioSegment.from_mp3(file)
                sound = sound.set_frame_rate(target_sr)
                clip = sound.get_array_of_samples()
                clip = np.array(clip)
                clip = (clip - clip.mean())/np.abs(clip).std()
                np.save(dst_file, clip)
                copied += 1
            else:
                existing += 1
        print(f'\r{copied}, {existing} / {i}', end='')

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_folder, batch_size=32,  shuffle=True, sample_size=44100):
        'Initialization'
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classes = []
        self.audio_files = []
        self.sample_size = sample_size
        filenames = glob(dataset_folder+'**/*', recursive=True)
        for file in filenames:
            filename = file.split('/')[-1]
            if '.npy' not in filename:
                self.classes.append(filename)
            else:
                self.audio_files.append(file)
        self.audio_files = np.array(self.audio_files)
        self.classes_dict = {cl:i for i, cl in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        self.on_epoch_end()
        print(self.classes_dict)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.audio_files) / self.batch_size)) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(self.audio_files[indexes])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.audio_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
#         X = np.empty((self.batch_size, self.sample_size))
#         y = np.empty((self.batch_size, 1), dtype=int)
        X = []
        y = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             X[i,] = np.load(ID)
            x = np.load(ID)
            X.append(x)
            # Store class
#             y[i] = self.classes_dict[ID.split('/')[-2]]
            y.append(self.classes_dict[ID.split('/')[-2]])
        X = np.array(X)
        return X.reshape(*X.shape, 1), to_categorical(y, num_classes=self.n_classes) # np.array(y).reshape(-1, 1) # tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
def create_train_val_folders(dataset_folder, all_subfolder = 'all/', train_subfolder = 'train/', val_subfolder = 'val/', ratio = 0.2):
    dataset_folder_all = dataset_folder + all_subfolder
    dataset_folder_train = dataset_folder + train_subfolder
    dataset_folder_val = dataset_folder + val_subfolder
    filenames = glob(dataset_folder_all+'**/*', recursive=True)
    print(dataset_folder_train)
    print(dataset_folder_val)
    classes = []
    audio_files = []
    for file in filenames:
        filename = file.split('/')[-1]
        if '.npy' not in filename:
            classes.append(filename)
        else:
            audio_files.append(file)
    if not os.path.exists(dataset_folder_train):
        os.makedirs(dataset_folder_train)
    if not os.path.exists(dataset_folder_val):
        os.makedirs(dataset_folder_val)
    for cl in classes:
        new_folder = dataset_folder_train + cl
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_folder = dataset_folder_val + cl
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            
    audio_files = np.array(audio_files)
    np.random.shuffle(audio_files)
    
    fr = int(ratio*len(audio_files))
    audio_files_train = audio_files[fr:]
    audio_files_val = audio_files[:fr]
    
    for f in audio_files_train:
        if '.npy' in f:
            copyfile(f, f.replace(all_subfolder, train_subfolder))
        
    for f in audio_files_val:
        if '.npy' in f:
            copyfile(f, f.replace(all_subfolder, val_subfolder))

def get_train_val_files(audio_files, ratio = 0.2):
    class_audio_files = np.sort(audio_files)
    cut_index = int(len(class_audio_files)*ratio)
    file_id = class_audio_files[cut_index].split('/')[-1].split('_')[0]

    while class_audio_files[cut_index].split('/')[-1].split('_')[0] == file_id:
        cut_index += 1
    val_files = class_audio_files[:cut_index]
    train_files = class_audio_files[cut_index:]
    return train_files, val_files

def create_train_val_folders_with_diff_files(dataset_folder, all_subfolder = 'all/', train_subfolder = 'train/', val_subfolder = 'val/', ratio = 0.2):
    dataset_folder_all = dataset_folder + all_subfolder
    dataset_folder_train = dataset_folder + train_subfolder
    dataset_folder_val = dataset_folder + val_subfolder
    filenames = glob(dataset_folder_all+'**/*', recursive=True)
    print(dataset_folder_train)
    print(dataset_folder_val)
    classes = []
    audio_files = []
    for file in filenames:
        filename = file.split('/')[-1]
        if '.npy' not in filename:
            classes.append(filename)
        else:
            audio_files.append(file)
    if not os.path.exists(dataset_folder_train):
        os.makedirs(dataset_folder_train)
    if not os.path.exists(dataset_folder_val):
        os.makedirs(dataset_folder_val)
        
    files_dict = {}
    for cl in classes:
        new_folder = dataset_folder_train + cl
        files_dict[cl] = glob(dataset_folder_all+cl+'/**/*', recursive=True)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_folder = dataset_folder_val + cl
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            
    for k, v in files_dict.items():
        train_files, val_files = get_train_val_files(v, ratio = ratio)
        print(len(val_files) / (len(train_files) + len(val_files)))
        for f in train_files:
            if '.npy' in f:
                copyfile(f, f.replace(all_subfolder, train_subfolder))

        for f in val_files:
            if '.npy' in f:
                copyfile(f, f.replace(all_subfolder, val_subfolder))
    return



##########
from pydub import AudioSegment

FOLDER = '/home/usuario/birds/birdsong-recognition/'
TRAIN_FOLDER = FOLDER + 'train_audio/'

def get_train_clip(dataframe, resample=None, chunk_size = None):
    if type(dataframe) in [str, np.str_]:
        filename = dataframe
    else:
        filename = TRAIN_FOLDER+dataframe['ebird_code']+'/'+dataframe['filename']
    sound = AudioSegment.from_mp3(filename)
    orig_sr = sound.frame_rate
    if resample is not None:
        sound = sound.set_frame_rate(resample)
    else:
        resample = orig_sr
    clip = sound.get_array_of_samples()
    duration = sound.duration_seconds
    
    if chunk_size is not None:
        fr = int(np.random.rand(1)*(duration-chunk_size) * resample)
        to = fr + chunk_size * resample
        clip = np.array(clip)[fr:to]
    else:
        clip = np.array(clip)
        
    clip = (clip - clip.mean())/np.abs(clip).std()
    return clip, orig_sr, duration


def sound_slice(x, sr = 22050, chunk_seconds=5, hop_seconds=1, discard_last=True):
    hop_size = int(hop_seconds*sr)
    chunk_size = int(chunk_seconds*sr)
    n_chunks = len(x) // hop_size
    chunks = []
    for i in range(n_chunks):
        chunk = x[i*hop_size: i*hop_size + chunk_size]
        if discard_last and len(chunk) == chunk_size:
            chunks.append(chunk)
        elif not discard_last:
            chunks.append(chunk)
            
    return chunks

def save_chunks(ebird_folder, dataframe_row, target_sr = 22050, chunk_seconds=5, hop_seconds=1, std_thres = 0.1, save = True, discard_last=True):
    under_tres = []
    x, orig_sr, duration = get_train_clip(dataframe_row, target_sr)
    chunks = sound_slice(x, sr = target_sr, chunk_seconds=chunk_seconds, hop_seconds=hop_seconds, discard_last=True)

    for j, chunk in enumerate(chunks):
        chunk_std = chunk.std()

        if chunk_std > std_thres:
            if save:
                file_to_save = ebird_folder + ''.join(dataframe_row['filename'].split('.')[:-1]) + f'_{j+1}_{chunk_seconds}_{hop_seconds}.npy'
                print(f'\r{j} - {file_to_save}', end='')
                if not os.path.exists(file_to_save):
                    np.save(file_to_save, chunk)

        else:
            under_tres.append(chunk)

def save_class_dataset(train, dataset_folder, ebird_code, target_sr = 22050, chunk_seconds=5, hop_seconds=1, std_thres = 0.1, discard_last=True):
    dataframe = train[train['ebird_code']==ebird_code]
    ebird_folder = dataset_folder + ebird_code + '/'
    if not os.path.exists(ebird_folder):
        os.makedirs(ebird_folder)
    for i in range(len(dataframe)): 
        dataframe_row = dataframe.iloc[i]
        save_chunks(ebird_folder, dataframe_row, target_sr = target_sr, chunk_seconds=chunk_seconds, hop_seconds=hop_seconds, std_thres = std_thres, discard_last=discard_last)
        
        
        
#####

from pydub.utils import mediainfo
def get_class_audio_files_npy(TRAIN_FOLDER, classes = ['amegfi', 'amecro', 'aldfly'], min_duration=5, ratio = 0.2, extention='.npy', sr=22050):
    class_audiofiles = {}
    for cl in classes:
        audio_files = []
        durations = []
        cl_folder = TRAIN_FOLDER + cl 
        cl_audio_files = glob(cl_folder+'/**/*', recursive=True)
        duration = 0
        for file in cl_audio_files:
            filename = file.split('/')[-1]
            
            if extention in filename:
                clip = np.load(file)
                duration = len(clip)/sr
                if duration>=min_duration:
                    durations.append(duration)
                    audio_files.append(file)

        audio_files = np.array(audio_files) 
        indexes = np.random.shuffle(list(range(len(audio_files))))
        class_audiofiles[cl] = {}
        class_audiofiles[cl]['files'] = np.array(audio_files)[indexes].reshape(-1)
        class_audiofiles[cl]['durations'] = np.array(durations)[indexes].reshape(-1)
        
    train_files = []
    val_files = []
    train_labels = []
    val_labels = []
    for cl in classes:
        duration = class_audiofiles[cl]['durations'].sum()
        acc_dur = 0

        for i, file in enumerate(class_audiofiles[cl]['files']):
            acc_dur += class_audiofiles[cl]['durations'][i]
            if acc_dur/duration >= ratio:
                train_files = train_files + list(class_audiofiles[cl]['files'][i:])
                val_files = val_files + list(class_audiofiles[cl]['files'][:i])
                train_labels = train_labels + [cl]*(len(class_audiofiles[cl]['files']) - i)
                val_labels = val_labels + [cl]*i
                print(acc_dur/duration)
                break
    return class_audiofiles, train_files, val_files, train_labels, val_labels

def get_class_audio_files(TRAIN_FOLDER, classes = ['amegfi', 'amecro', 'aldfly'], min_duration=5, ratio = 0.2):
    class_audiofiles = {}
    for cl in classes:
        audio_files = []
        durations = []
        cl_folder = TRAIN_FOLDER + cl 
        cl_audio_files = glob(cl_folder+'/**/*', recursive=True)
        duration = 0
        for file in cl_audio_files:
            filename = file.split('/')[-1]
            if '.mp3' in filename:
                info = mediainfo(file)
                duration = float(info['duration'])
                if duration>=min_duration:
                    durations.append(duration)
                    audio_files.append(file)

        audio_files = np.array(audio_files) 
        indexes = np.random.shuffle(list(range(len(audio_files))))
        class_audiofiles[cl] = {}
        class_audiofiles[cl]['files'] = np.array(audio_files)[indexes].reshape(-1)
        class_audiofiles[cl]['durations'] = np.array(durations)[indexes].reshape(-1)
        
    train_files = []
    val_files = []
    train_labels = []
    val_labels = []
    for cl in classes:
        duration = class_audiofiles[cl]['durations'].sum()
        acc_dur = 0

        for i, file in enumerate(class_audiofiles[cl]['files']):
            acc_dur += class_audiofiles[cl]['durations'][i]
            if acc_dur/duration >= ratio:
                train_files = train_files + list(class_audiofiles[cl]['files'][i:])
                val_files = val_files + list(class_audiofiles[cl]['files'][:i])
                train_labels = train_labels + [cl]*(len(class_audiofiles[cl]['files']) - i)
                val_labels = val_labels + [cl]*i
                print(acc_dur/duration)
                break
    return class_audiofiles, train_files, val_files, train_labels, val_labels

class DataGeneratorV2(Sequence):
    'Generates data for Keras'
    def __init__(self, files, labels, batch_size=32,  shuffle=True, sr=22050, chunk_seconds=5, min_std=0.02, channel_first=False, one_hot=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classes = np.unique(labels)
        self.labels = labels
        self.audio_files = np.array(files)
        self.chunk_samples = chunk_seconds * sr
        self.min_std = min_std
        self.channel_first = channel_first
        self.one_hot = one_hot
        

        self.classes_dict = {cl:i for i, cl in enumerate(self.classes)}
        self.n_classes = len(self.classes)
        self.on_epoch_end()
        print(self.classes_dict)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.audio_files) / self.batch_size)) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(self.audio_files[indexes])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.audio_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def sample_audio_clip(self, clip):
        fr = int(np.random.rand(1)*(len(clip)-self.chunk_samples))
        to = fr + self.chunk_samples
        x = clip[fr:to]
        return x, fr, to
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.channel_first:
            X = np.empty((self.batch_size, 1, self.chunk_samples))
        else:
            X = np.empty((self.batch_size, self.chunk_samples, 1))
        
        if self.one_hot:
            y = np.zeros((self.batch_size, self.n_classes), dtype=int)
        else:
            y = np.zeros((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             X[i,] = np.load(ID)
            clip = np.load(ID)
            x, fr, to = self.sample_audio_clip(clip)
            std = x.std()
            while std < self.min_std:
                x, fr, to = self.sample_audio_clip(clip)
                std = x.std()
            if self.channel_first:
                X[i, 0, :] = x
            else:
                X[i, :, 0] = x
            if self.one_hot:
                y[i][self.classes_dict[ID.split('/')[-2]]] = 1
            else:
                y[i] = self.classes_dict[ID.split('/')[-2]]
        return X, y


###################
###################
###################

from tensorflow.keras.layers import Dense, Conv1D, Input, MaxPool1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

def get_fourier_weights(window_size):
    frec = np.linspace(-window_size//2, window_size//2-1, window_size)
    time = np.linspace(0, window_size-1, window_size)
    hanning_window = np.hanning(window_size)

    filters_cos = []
    filters_sin = []
    for i in range(window_size//2):
        filters_cos.append(np.cos(2*np.pi*frec[i]*time/window_size))
        filters_sin.append(np.sin(2*np.pi*frec[i]*time/window_size))
    filters_cos = np.array(filters_cos)[::-1]*hanning_window
    filters_sin = np.array(filters_sin)[::-1]*hanning_window
    return filters_cos, filters_sin

def set_cnn_weights(model, filters_cos, filters_sin, cos_layer='cos', sin_layer='sin'):
    weights_cos = model.get_layer(cos_layer).get_weights()
    weights_sin = model.get_layer(sin_layer).get_weights()
    weights_cos[0] = np.array(filters_cos).T.reshape(*weights_cos[0].shape)
    weights_sin[0] = np.array(filters_sin).T.reshape(*weights_sin[0].shape)
    model.get_layer(cos_layer).set_weights(weights_cos)
    model.get_layer(sin_layer).set_weights(weights_sin)
    
    
def get_keras_fourier_model(window_size = 1024, set_fourier_weights=True, trainable=False, min_power=1e-10):
    db_constant = 10/np.log(10)
    kernel_size = window_size
    stride = kernel_size//4
    filters = kernel_size//2
    inp = Input(shape=(None,1))
    cos_out = Conv1D(filters, kernel_size, stride, padding='same', name = 'cos')(inp)
    sin_out = Conv1D(filters, kernel_size, stride, padding='same', name = 'sin')(inp)
    fourier_out = db_constant*K.log(K.square(cos_out) + K.square(sin_out) + min_power)
    model = Model(inp, fourier_out)
    if set_fourier_weights:
        filters_cos, filters_sin = get_fourier_weights(window_size)
        set_cnn_weights(model, filters_cos, filters_sin)
    if not trainable:
        model.get_layer('cos').trainable = False
        model.get_layer('sin').trainable = False
    return model, (inp, fourier_out)



########################
import torch
from torch import nn
from torch.nn import functional as F



class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, classes, chunk_seconds, sr, min_std, multilabel=False):
        'Initialization'
        self.min_std = min_std
        self.list_IDs = list_IDs
        self.classes = classes
        self.n_classes = len(classes)
        self.chunk_samples = chunk_seconds * sr
        self.classes_dict = {cl:i for i, cl in enumerate(self.classes)}
        self.multilabel = multilabel

    def sample_audio_clip(self, clip):
        fr = int(np.random.rand(1)*(len(clip)-self.chunk_samples))
        to = fr + self.chunk_samples
        x = clip[fr:to]
        return x, fr, to
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        clip = np.load(ID)
        X, fr, to = self.sample_audio_clip(clip)
        std = X.std()
        while std < self.min_std:
            X, fr, to = self.sample_audio_clip(clip)
            std = X.std()
        # Reshape for 1 channel 1D CNN
        X = torch.from_numpy(X.reshape(1, -1)).float()
        if self.multilabel:
            y = torch.zeros((self.n_classes))
            y[self.classes_dict[ID.split('/')[-2]]] = 1
        else:
            y = torch.tensor(self.classes_dict[ID.split('/')[-2]])

        return X, y
    
    
def get_pytorch_model(window_size, resnet='resnet18', pretrained=True, n_classes=10, init_fourier=True, train_fourier=False):
    window_size = 1024
    kernel_size = window_size
    stride = kernel_size//4
    filters = kernel_size//2
    
    model_resnet = torch.hub.load('pytorch/vision:v0.6.0', resnet, pretrained=pretrained)
    if resnet=='resnet18':
        linear_inp = 512
    else:
        linear_inp = 2048
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.cos = nn.Conv1d(1, filters, kernel_size, stride=stride)
            self.sin = nn.Conv1d(1, filters, kernel_size, stride=stride)
            if init_fourier:
                cos_weights, sin_weights = get_fourier_weights(window_size)
                self.cos.weight.data = torch.from_numpy(cos_weights.reshape(cos_weights.shape[0], 1, cos_weights.shape[1])).float()
                self.sin.weight.data = torch.from_numpy(sin_weights.reshape(sin_weights.shape[0], 1, sin_weights.shape[1])).float()
            self.resnet = nn.Sequential(*list(model_resnet.children())[:-1])
            self.fc1 = nn.Linear(linear_inp, n_classes)
        def forward(self, x):
            min_power=1e-10
            x_spec = 10*torch.log10(self.cos(x)**2 + self.sin(x)**2 + min_power)
            x_spec = (x_spec + 60)/120 - 0.5
            x = torch.reshape(x_spec, (len(x_spec), 1, 512, -1))
            x = torch.cat([x, x, x], dim=1)
            x = self.resnet(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            return x_spec, x
    model = Net()
    if not train_fourier:
        list(model.cos.parameters())[0].requires_grad = False
        list(model.sin.parameters())[0].requires_grad = False
    return model

def get_pytorch_model_all_conv(window_size, resnet='resnet18', pretrained=True, n_classes=10, init_fourier=True, train_fourier=False):
    window_size = 1024
    kernel_size = window_size
    stride = kernel_size//4
    filters = kernel_size//2
    
    model_resnet = torch.hub.load('pytorch/vision:v0.6.0', resnet, pretrained=pretrained)
    if resnet=='resnet18':
        linear_inp = 512
    else:
        linear_inp = 2048
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.cos = nn.Conv1d(1, filters, kernel_size, stride=stride)
            self.sin = nn.Conv1d(1, filters, kernel_size, stride=stride)
            if init_fourier:
                cos_weights, sin_weights = get_fourier_weights(window_size)
                self.cos.weight.data = torch.from_numpy(cos_weights.reshape(cos_weights.shape[0], 1, cos_weights.shape[1])).float()
                self.sin.weight.data = torch.from_numpy(sin_weights.reshape(sin_weights.shape[0], 1, sin_weights.shape[1])).float()
            self.resnet = nn.Sequential(*list(model_resnet.children())[:-1])
            self.conv_out = nn.Conv2d(linear_inp, 10, 1)
        def forward(self, x):
            min_power=1e-10
            x_spec = 10*torch.log10(self.cos(x)**2 + self.sin(x)**2 + min_power)
            x_spec = (x_spec + 60)/120 - 0.5
            x = torch.reshape(x_spec, (len(x_spec), 1, 512, -1))
            x = torch.cat([x, x, x], dim=1)
            x = self.resnet(x)
            x = self.conv_out(x).flatten(start_dim=1)
            return x_spec, x
    model = Net()
    if not train_fourier:
        list(model.cos.parameters())[0].requires_grad = False
        list(model.sin.parameters())[0].requires_grad = False
    return model

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    
    return correct_pred.sum(), len(correct_pred)

def validate_model_acc_loss(model, dgen_val, criterion, device):
    #model.eval()  
    with torch.no_grad():
        running_loss = 0.0
        total_ok = 0
        total_predictions = 0
        batches_per_epoch = len(dgen_val)
        for i, (X, y) in enumerate(dgen_val):
#             inputs, labels = torch.from_numpy(X).float().to(device), torch.from_numpy(y).long().to(device)
            inputs, labels = X.to(device), y.to(device)
            _, y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            ok, total = multi_acc(y_pred, labels)
            total_ok = total_ok + ok
            running_loss = running_loss + loss
            total_predictions = total_predictions + total
            print(f'\r{i+1}/{batches_per_epoch} - val loss: {running_loss/(i+1)}, val acc: {total_ok/total_predictions}', end='')
    #model.train()
    return (running_loss/(i+1)).detach().item(), (total_ok/total_predictions).detach().item()

def validate_model_loss_detail(model, dgen_val, criterion, device):
    #model.eval()
    Xs = []
    losses = []
    y_preds = []
    ys = []
    with torch.no_grad():
        running_loss = 0.0
        total_ok = 0
        total_predictions = 0
        batches_per_epoch = len(dgen_val)
        for i, (X, y) in enumerate(dgen_val):
#             inputs, labels = torch.from_numpy(X).float().to(device), torch.from_numpy(y).long().to(device)
            Xs.append(X)
            ys.append(y)
            inputs, labels = X.to(device), y.to(device)
            _, y_pred = model(inputs)
            y_preds.append(y_pred)
            loss = criterion(y_pred, labels)
            losses.append(loss.detach())
            ok, total = multi_acc(y_pred, labels)
            total_ok = total_ok + ok
            running_loss = running_loss + loss.mean()
            total_predictions = total_predictions + total
            print(f'\r{i+1}/{batches_per_epoch} - val loss: {running_loss/(i+1)}, val acc: {total_ok/total_predictions}', end='')
    #model.train()
    return (running_loss/(i+1)).detach().item(), (total_ok/total_predictions).detach().item(), Xs, ys, y_preds, losses