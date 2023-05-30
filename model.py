import os
import shutil
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import librosa
import soundfile as sf

# Set to some specific value if you want it to be used for every song, otherwise first loaded song will overwrite this with its value
sample_rate = None 

sample_duration = 4

cut_width = 22 # each second is ~44 width units in those spectograms (for n_fft=2048, hop_length=512)

scaler = MinMaxScaler()
scaler_fitted = False

n_fft = 2048
hop_length = 512

def load_dataset(max_files=1):
    global sample_rate
    """
    Returns python list with [num_tracks] numpy array of shape (track_length, 1), track_length may be different for each song.
    """
    dataset = []
    for i, file in enumerate(os.listdir('uploads')):
        if i >= max_files:
            break
        data, sample_rate = librosa.load(f'./uploads/{file}', sr=sample_rate)
        data = np.expand_dims(data, axis=-1)
        dataset.append(data)
    return dataset

def sample_dataset(dataset):
    assert sample_rate is not None
    sampled_dataset = []
    sample_length = int(sample_rate * sample_duration)
    for track in dataset:
        for i in range(0, track.shape[0], sample_length):
            sampled_dataset.append(track[i:i+sample_length])
        if sampled_dataset[-1].shape[0] != sample_length:
            sampled_dataset.pop()
    return sampled_dataset

def write_audio(filename, audio):
  sf.write(filename, audio, sample_rate, 'PCM_24')

def scaler_transform(spectrograms):
    global scaler_fitted
    num_samples, width, height, channels = spectrograms.shape
    spectrograms = np.reshape(spectrograms, (num_samples, width * height * channels))
    if scaler_fitted: # type: ignore
        spectrograms = scaler.transform(spectrograms)
    else:
        spectrograms = scaler.fit_transform(spectrograms)
        scaler_fitted = True
    spectrograms = np.reshape(spectrograms, (num_samples, width, height, channels)) # type: ignore
    return spectrograms

def scaler_inverse_transform(spectrograms):
    assert scaler_fitted is True
    num_samples, width, height, channels = spectrograms.shape
    spectrograms = np.reshape(spectrograms, (num_samples, width * height * channels))
    spectrograms = scaler.inverse_transform(spectrograms)
    spectrograms = np.reshape(spectrograms, (num_samples, width, height, channels)) # type: ignore
    return spectrograms

def pad_audio(audio, pad):
    pad_for = pad - (audio.shape[0] % pad)
    if pad == pad_for:
        return audio
    return np.expand_dims(np.pad(audio.reshape(-1), (0, pad_for), 'constant', constant_values=(0, 0)), axis=-1)

def get_mel_spectogram(audio):
    assert sample_rate is not None
    padded_audio = pad_audio(audio, pad=n_fft).reshape(-1)
    S = librosa.feature.melspectrogram(y=padded_audio, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
    S = np.expand_dims(S, axis=-1) # Add channel dimension
    return S

def get_cut_range(S):
    cut_y_start = S.shape[1] // 2 - cut_width // 2
    cut_height = 128 # 128 is the max height (at least for n_fft=2048, hop_length=512)
    return slice(0, cut_height), slice(cut_y_start, cut_y_start+cut_width)

# For now cuts are always in the middle of spectogram with the same size
def cut_spectogram(S):
    S_cut = np.copy(S)
    cut_range = get_cut_range(S)
    cut_region = np.array(S_cut[cut_range])
    S_cut[cut_range] = 0
    return S_cut, cut_region

def show_mel_spectogram(S):
    assert sample_rate is not None
    S_dB = librosa.power_to_db(S.reshape(S.shape[:-1]), ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')

def reconstruct_raw_audio_from_mel_spectogram(S):
    assert sample_rate is not None
    return np.expand_dims(librosa.feature.inverse.mel_to_audio(S.reshape(S.shape[:-1]), sr=sample_rate, n_fft=n_fft, hop_length=hop_length), axis=-1)

def reconstruct_final_audio(predicted_S, cut_S, cut_raw_audio):
    cut_range = get_cut_range(cut_S)
    S = np.copy(cut_S)
    S[cut_range] = predicted_S[cut_range]
    # rec_raw_audio = reconstruct_raw_audio_from_mel_spectogram(S)
    # rec = np.copy(cut_raw_audio)
    # rec[cut_range] = rec_raw_audio[cut_range]
    return S

cnn_model = tf.keras.models.load_model('models/CNN-256-128-128-0')

def run():
    ds = load_dataset()
    sample_ds = sample_dataset(ds)
    spectrogram_ds = np.array([get_mel_spectogram(sample) for sample in sample_ds])

    CS = [cut_spectogram(S) for S in spectrogram_ds]
    X = np.array([S_cut for S_cut, _ in CS])
    y = np.copy(spectrogram_ds)

    y = scaler_transform(y)
    X = scaler_transform(X)

    y_pred = cnn_model.predict(X)

    y = scaler_inverse_transform(y)
    X = scaler_inverse_transform(X)
    y_pred = scaler_inverse_transform(y_pred)

    og_audio = sample_ds[0]
    cut_audio = reconstruct_raw_audio_from_mel_spectogram(y_pred[0])

    filename = f'{uuid.uuid1()}.wav'
    write_audio(f'tmp/{filename}', cut_audio)

    return filename

