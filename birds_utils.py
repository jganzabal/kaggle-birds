from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np
from glob import glob
from shutil import copyfile
import os

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
            print(x.shape)
            X.append(x)
            # Store class
#             y[i] = self.classes_dict[ID.split('/')[-2]]
            y.append(self.classes_dict[ID.split('/')[-2]])
        X = np.array(X)
        print(X.shape)
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
        print(len(val_files) / len(train_files))
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

def get_train_clip(dataframe, resample=None):
    filename = TRAIN_FOLDER+dataframe['ebird_code']+'/'+dataframe['filename']
    sound = AudioSegment.from_mp3(filename)
    orig_sr = sound.frame_rate
    if resample is not None:
        sound = sound.set_frame_rate(resample)
    clip = sound.get_array_of_samples()
    duration = sound.duration_seconds

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