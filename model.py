import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import librosa
import soundfile as sf
from tensorflow import keras

sample_duration = 4
cut_width = 22
n_fft = 2048
hop_length = 512


def load_song(path):
    sample_rate = None
    data, sample_rate = librosa.load(path, sr=sample_rate)
    data = np.expand_dims(data, axis=-1)
    return data, sample_rate


def sample_song(song, sample_rate):
    sampled_song = []
    sample_length = int(sample_rate * sample_duration)
    for i in range(0, song.shape[0], sample_length):
        sampled_song.append(song[i:i+sample_length])
    if sampled_song[-1].shape[0] != sample_length:
        sampled_song.pop()
    return sampled_song


def write_audio(filename, audio, sample_rate):
  sf.write(filename, audio, sample_rate, 'PCM_24')


def scaler_transform(scaler, spectrograms):
    num_samples, width, height, channels = spectrograms.shape
    spectrograms = np.reshape(spectrograms, (num_samples, width * height * channels))
    spectrograms = scaler.fit_transform(spectrograms)
    spectrograms = np.reshape(spectrograms, (num_samples, width, height, channels)) # type: ignore
    return scaler, spectrograms


def scaler_inverse_transform(scaler, spectrograms):
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


def get_mel_spectogram(audio, sample_rate):
    padded_audio = pad_audio(audio, pad=n_fft).reshape(-1)
    S = librosa.feature.melspectrogram(y=padded_audio, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
    S = np.expand_dims(S, axis=-1) # Add channel dimension
    return S


def show_mel_spectogram(S, sample_rate, name):
    try:
        plt.clf()
    except:
        pass
    S_dB = librosa.power_to_db(S.reshape(S.shape[:-1]), ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(f'Mel-frequency spectrogram {name}')
    plt.savefig(f"tmp/spectrogram-${uuid.uuid1()}.jpg")


def restore_spectrogram(S_cut, S_pred):
    def get_cut_range(S):
        tmp = S.reshape(S.shape[:-1]).T
        start, end = None, None

        for i, s_col in enumerate(tmp[10:-10]):
            if start == None:
                if np.all(s_col < 0.05):
                    start = i + 10
            else:
                if not np.all(s_col < 0.05):
                    end = i + 10
                    break

        return slice(start, end)


    def splice_spectrograms(S_cut, S_pred):
        cut_range = get_cut_range(S_cut)
        S = np.copy(S_cut)
        S[:, cut_range] = S_pred[:, cut_range]
        return S

    tmp_S = splice_spectrograms(S_cut, S_pred)
    cr0 = get_cut_range(S_cut)
    cr = slice(cr0.start - 4, cr0.stop + 4)
    roi = tmp_S[:, cr].copy()
    blurred_roi = cv2.GaussianBlur(roi, (3, 1), 0)
    blurred_roi = cv2.GaussianBlur(blurred_roi, (3, 1), 0)
    blurred_roi = np.expand_dims(blurred_roi, axis=-1)
    blurred_S = tmp_S.copy()
    blurred_S[:, cr] = blurred_roi
    return blurred_S, cr0


def reconstruct_raw_audio_from_mel_spectogram(S, sample_rate):
    return np.expand_dims(librosa.feature.inverse.mel_to_audio(S.reshape(S.shape[:-1]), sr=sample_rate, n_fft=n_fft, hop_length=hop_length), axis=-1)


def get_cnn_model(input_shape):
    input = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    conv2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

    add1 = keras.layers.Add()([conv1, conv2])

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(add1)
    conv3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)

    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(add1)
    conv4 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(add1)
    
    conv6 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(add1)

    add2 = keras.layers.Add()([conv3, conv4, conv5])

    conv7 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(add2)

    add3 = keras.layers.Add()([conv7, add2, conv6])

    output = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(add3)
    output = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(output)

    return keras.Model(inputs=input, outputs=output)


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


random_state = 45100
np.random.seed(random_state)
cnn_model = keras.models.load_model('models/2', custom_objects={'SSIMLoss': SSIMLoss})


def run(path):
    scaler = MinMaxScaler()
    song_og, sample_rate = load_song(path)
    song_samples = sample_song(song_og, sample_rate)
    spectrograms = np.array([get_mel_spectogram(sample, sample_rate) for sample in song_samples])
    scaler, scaled_spectograms = scaler_transform(scaler, spectrograms)
    y_pred = cnn_model.predict(scaled_spectograms)
    y_pred = scaler_inverse_transform(scaler, y_pred)
    restored_S, _ = restore_spectrogram(spectrograms[0], y_pred[0])
    cut_audio = reconstruct_raw_audio_from_mel_spectogram(restored_S, sample_rate)
    filename = f'{uuid.uuid1()}.wav'
    write_audio(f'tmp/{filename}', cut_audio, sample_rate)
    return filename
