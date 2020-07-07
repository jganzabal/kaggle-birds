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
            X.append(np.load(ID))
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