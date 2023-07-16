"""
Author: Anton Altmeyer
Date: 25.06.2023

Sensor data fusion of PPG and ACC data for heart rate estimation using neural network
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import skimage

F = 50
T = 1 / F


def load_data(filename):
    """

    :param filename: file name of the data to load
    :return: raw PPG and ACC data and the labels as bpm of ECG
    along with a time vector and the used window duration/size
    """
    try:
        data = scipy.io.loadmat(filename)
        rawPPG = pd.DataFrame.from_dict(data["rawPPG"])
        rawACC = pd.DataFrame.from_dict(data["rawAcc"])
        bpmECG = pd.DataFrame.from_dict(data["bpm_ecg"])

        time = np.arange(0, len(rawPPG) * T, T)
        window = np.arange(0, 8 * F, 1)

        return rawPPG, rawACC, bpmECG, time, window

    except Exception as e:
        print("Error: ", e)

def design_bandpass_filter(order, lowcut, highcut, fs):
    """
    Creating the coefficients (b, a), frequency samples and filter response of a bandpass filter
    :param order: filter order
    :param lowcut: lower cutoff frequency
    :param highcut: higher cutoff frequency
    :param fs: sampling frequency
    :return: coefficients, frequency samples and filter response
    """
    nyq = 0.5 * fs
    low_n = lowcut / nyq
    high_n = highcut / nyq

    # Design the Butterworth bandpass filter
    b, a = scipy.signal.butter(order, [low_n, high_n], btype='band')

    freq, resp = scipy.signal.freqz(b, a, fs=fs)

    return b, a, freq, resp

def visualize_filter(f, r):
    """
    Function visualize the filter response
    :param f: frequency samples
    :param r: filter response
    :return: None
    """
    plt.figure()
    plt.plot(f, np.abs(r))
    plt.xlabel("Frequency in (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Magnitude response")
    plt.grid(True)
    plt.show()

def create_windows(signal, window_size, shift):
    """
    Creating the windows for the data
    :param signal: signal data, here: PPG or ACC
    :param window_size: the window size or duration
    :param shift: the seconds to shift the window along the data
    :return: returns the created windows
    """
    windows = []
    start = 0
    end = window_size
    while end <= len(signal):
        window = signal[start:end]
        windows.append(window)
        start += shift
        end += shift
    return windows


def filter_windows(windows, b, a):
    """
    Function filters the windows using the created bandpass filter coefficients
    :param windows: the window data which should be filtered
    :param b: coefficients of numerator
    :param a: coefficients of denominator
    :return: returns the filtered windows
    """
    w_filtered = []
    for w in windows:
        out = scipy.signal.filtfilt(b, a, w)
        w_filtered.append(out)
    return w_filtered


def create_power_spectrum(signal):
    """
    Creates the spectra of each window in the frequency domain
    :param signal: signal/window which will translated from time to frequency domain
    :return: returns the power spectra
    """
    power_spectrum = []
    for i in range(0, len(signal)):
        out = np.fft.fft(signal[i])
        power_spectrum.append(np.abs(out) ** 2)
    return power_spectrum

def zero_pad(spectrum, length):
    """
    Applies zero padding on the previously created spectra
    :param spectrum: spectrum which is then zero padded
    :param length: number of total frequency bins to pad
    :return: returns the padded spectrum
    """
    pad_signal = []
    for i in range(0, len(spectrum)):
        out = np.pad(spectrum[i], (0, length - len(spectrum[0])), mode="constant")
        pad_signal.append(out)
    return pad_signal


def normalize_power_spectrum(spectrum):
    """
    Normalizing the power spectra
    :param spectrum: spectrum which is then normalized
    :return: returns the normalized spectrum
    """
    norm_spectrum = []
    for i in range(0, len(spectrum)):
        out = (spectrum[i] - np.min(spectrum[i])) / (np.max(spectrum[i]) - np.min(spectrum[i]))
        norm_spectrum.append(out)
    return norm_spectrum


def extract_frequency(spectrum, min_freq, max_freq):
    """
    Extracting/Filtering frequencies of the spectrum
    :param spectrum: spectrum which is filtered
    :param min_freq: lower frequency
    :param max_freq: higher frequency
    :return: returns the spectrum with extracted frequencies
    """
    bin_axis = np.arange(0, 2047, 1)
    bin_res = 0.01215
    idx_min = min_freq / bin_res
    idx_max = max_freq / bin_res
    idx = np.where((bin_axis >= idx_min) & (bin_axis <= idx_max))

    extracted_spectrum = []
    for i in range(0, len(spectrum)):
        extracted_spectrum.append(spectrum[i][idx])
    return extracted_spectrum


def calc_mean_intensity_acc(windows_x, windows_y, windows_z):
    """
    Calculates the mean acc intensity using the absolute Hilbert signal method
    :param windows_x: windows of the x-axis acc data
    :param windows_y: windows of the y-axis acc data
    :param windows_z: windows of the z-axis acc data
    :return: returns the mean acc intensity (envelope signal of the mean acc data)
    """
    windows_x_env = []
    windows_y_env = []
    windows_z_env = []

    for i in range(0, len(windows_x)):
        windows_x_env = np.abs(scipy.signal.hilbert(windows_x[i]))
        windows_y_env = np.abs(scipy.signal.hilbert(windows_y[i]))
        windows_z_env = np.abs(scipy.signal.hilbert(windows_z[i]))

    mean_windows_env = np.mean([windows_x_env, windows_y_env, windows_z_env], axis=0)

    return mean_windows_env


if __name__ == '__main__':

    # create filter coefficients
    b_1, a_1, frequency, response = design_bandpass_filter(4, 0.4, 4, F)
    # visualize_filter(f=frequency, r=response)

    ####################################################################################################################

    power_spectra_ppg = []
    power_spectra_acc = []
    intensity_acc = []
    ground_truth = []

    for i in range(1, 3):
        # load data
        PPG, ACC, GT, t, w = load_data("BAMI-1/BAMI1_{}.mat".format(i))

        # save ground truth data
        ground_truth.append(GT)

        # create the windows of all 3 PPGs
        windows_ppg_1 = create_windows(PPG.iloc[0], len(w), 2 * F)
        windows_ppg_2 = create_windows(PPG.iloc[1], len(w), 2 * F)
        windows_ppg_3 = create_windows(PPG.iloc[2], len(w), 2 * F)

        # filter all windows of all 3 PPGs
        filtered_windows_ppg_1 = filter_windows(windows_ppg_1, b_1, a_1)
        filtered_windows_ppg_2 = filter_windows(windows_ppg_2, b_1, a_1)
        filtered_windows_ppg_3 = filter_windows(windows_ppg_3, b_1, a_1)

        # normalize all windows of all 3 PPGs
        norm_filtered_windows_ppg_1 = scipy.stats.zscore(filtered_windows_ppg_1)
        norm_filtered_windows_ppg_2 = scipy.stats.zscore(filtered_windows_ppg_2)
        norm_filtered_windows_ppg_3 = scipy.stats.zscore(filtered_windows_ppg_3)

        # calculate the mean of all windows of all 3 PPGs
        norm_filtered_windows_ppg = np.mean([norm_filtered_windows_ppg_1, norm_filtered_windows_ppg_2,
                                             norm_filtered_windows_ppg_3], axis=0)

        # resample the mean filtered windows
        ds_norm_filtered_windows_ppg = skimage.transform.resize(norm_filtered_windows_ppg,
                                                                (len(norm_filtered_windows_ppg), 200))

        # zero pad the down sampled windows to length of 2048
        pad_windows_ppg = zero_pad(ds_norm_filtered_windows_ppg, 2048)

        # calculate the power spectrum
        power_spectrum_ppg = create_power_spectrum(pad_windows_ppg)

        # calculate the normalized power spectrum
        norm_power_spectrum_ppg = normalize_power_spectrum(power_spectrum_ppg)

        # extract the power spectrum in range between 0.6 and 3.3 Hz
        extract_power_spectrum_ppg = extract_frequency(norm_power_spectrum_ppg, 0.6, 3.3)

        power_spectra_ppg.append(extract_power_spectrum_ppg)

    ####################################################################################################################

        # create the windows of all 3 ACCs
        windows_acc_x = create_windows(ACC.iloc[0], len(w), 2 * F)
        windows_acc_y = create_windows(ACC.iloc[1], len(w), 2 * F)
        windows_acc_z = create_windows(ACC.iloc[2], len(w), 2 * F)

        # filter all windows of all 3 ACCs
        filtered_windows_acc_x = filter_windows(windows_acc_x, b_1, a_1)
        filtered_windows_acc_y = filter_windows(windows_acc_y, b_1, a_1)
        filtered_windows_acc_z = filter_windows(windows_acc_z, b_1, a_1)

        # calculate the mean of all windows of all 3 ACCs
        filtered_windows_acc = np.mean([filtered_windows_acc_x, filtered_windows_acc_y, filtered_windows_acc_z], axis=0)

        # resample the mean filtered windows
        ds_filtered_windows_acc = skimage.transform.resize(filtered_windows_acc,
                                                           (len(filtered_windows_acc), 200))

        # zero pad the down sampled windows to length of 2048
        pad_windows_acc = zero_pad(ds_filtered_windows_acc, 2048)

        # calculate the power spectrum
        power_spectrum_acc = create_power_spectrum(pad_windows_acc)

        # calculate the normalized power spectrum
        norm_power_spectrum_acc = normalize_power_spectrum(power_spectrum_acc)

        # extract the power spectrum in range between 0.6 and 3.3 Hz
        extract_power_spectrum_acc = extract_frequency(norm_power_spectrum_acc, 0.6, 3.3)

        power_spectra_acc.append(extract_power_spectrum_acc)

    ####################################################################################################################

        mean_intensity_acc = calc_mean_intensity_acc(windows_acc_x, windows_acc_y, windows_acc_z)
        intensity_acc.append(mean_intensity_acc)

    ####################################################################################################################

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 37), strides=(4, 4),
                               activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                               input_shape=(2, 222, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1,
                               activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.Reshape((19, 64)),  # reshape to (None, 19, 64) from 4D to 3D input shape
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="relu"),

        # Concatenation with acceleration intensity
        tf.keras.layers.Dense(1),
        tf.keras.layers.Reshape((1, 1)),
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x[1], axis=1)),

        tf.keras.layers.LSTM(512, return_sequences=True),
        tf.keras.layers.LSTM(222),
        tf.keras.layers.Dense(222, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.Softmax(),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics="accuracy",
    )

    model.summary()

    ####################################################################################################################

    # X_train: train data with features
    # X_test: test data with features
    # y_train: train data with ground truth
    # y_test: test data with ground truth

    X_train = np.array([power_spectra_ppg[0][0], power_spectra_acc[0][0]])
    X_test = np.array([power_spectra_ppg[1][0], power_spectra_acc[1][0]])
    y_train = np.array(ground_truth[0][range(1, 3)])
    y_test = np.array(ground_truth[1][range(1, 3)])

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
