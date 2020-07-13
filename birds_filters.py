from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_ambient_noise(n_samples, sr, order=5, bandwith = 500, min_frec = 100, insect='random'):
    grillos_artificial = np.random.normal(0, 10, n_samples)
    noise = 0
    if insect == 'random':
        low_cut = np.random.rand() * (0.99*sr/2 - (bandwith + min_frec)) + min_frec
        high_cut = low_cut + bandwith
        noise = noise + butter_bandpass_filter(grillos_artificial, low_cut, high_cut, sr, order=order)
    if insect == 'grillo' or insect == 'both':
        low_cut = 2540.776699029126 
        high_cut = 3145.3125
        noise = noise + butter_bandpass_filter(grillos_artificial, low_cut, high_cut, sr, order=order)
    if insect == 'chicharra' or insect == 'both':
        low_cut = 7492.718
        high_cut = 8043.94
        noise = noise + butter_bandpass_filter(grillos_artificial, low_cut, high_cut, sr, order=order)
    return noise