"""
Author: Anton Altmeyer
Date: 25.06.2023

Sensor data fusion of PPG and ACC data for heart rate estimation using neural network
"""

import numpy as np
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
        rawPPG = np.array(data["rawPPG"])
        rawACC = np.array(data["rawAcc"])
        bpmECG = np.array(data["bpm_ecg"])

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

def create_windows(signal, window_size, shift, number_windows):
    """
    Creating the windows for the data
    :param number_windows: number of windows
    :param signal: signal data, here: PPG or ACC
    :param window_size: the window size or duration
    :param shift: the seconds to shift the window along the data
    :return: returns the created windows
    """
    windows = np.empty((number_windows, 400))
    start = 0
    end = window_size
    temp = 0
    while end <= len(signal):
        window = signal[start:end]
        windows[temp] = window
        start += shift
        end += shift
        temp += 1
    return windows


def filter_windows(windows, b, a, num_windows):
    """
    Function filters the windows using the created bandpass filter coefficients
    :param num_windows: number of windows
    :param windows: the window data which should be filtered
    :param b: coefficients of numerator
    :param a: coefficients of denominator
    :return: returns the filtered windows
    """
    w_filtered = np.empty((num_windows, 400))
    for x in range(0, num_windows):
        out = scipy.signal.filtfilt(b, a, windows[x])
        w_filtered[x] = out
    return w_filtered


def create_power_spectrum(signal, number_windows):
    """
    Creates the spectra of each window in the frequency domain
    :param number_windows: number of windows
    :param signal: signal/window which will translated from time to frequency domain
    :return: returns the power spectra
    """
    power_spectrum = np.empty((number_windows, 2048))
    for i in range(0, len(signal)):
        out = np.fft.fft(signal[i])
        power_spectrum[i] = (np.abs(out) ** 2)
    return power_spectrum

def zero_pad(signal, length, number_windows):
    """
    Applies zero padding on the previously created spectra
    :param number_windows: number of windows
    :param signal: spectrum which is then zero padded
    :param length: number of total frequency bins to pad
    :return: returns the padded spectrum
    """
    pad_signal = np.empty((number_windows, length))
    for i in range(0, number_windows):
        out = np.pad(signal[i], (0, length - len(signal[i])), mode="constant")
        pad_signal[i] = out
    return pad_signal


def normalize_power_spectrum(spectrum, number_windows):
    """
    Normalizing the power spectra
    :param number_windows: number of windows
    :param spectrum: spectrum which is then normalized
    :return: returns the normalized spectrum
    """
    norm_spectrum = np.empty((number_windows, 2048))
    out = np.empty((number_windows, 2048))
    for i in range(0, len(spectrum)):
        out[i] = np.linalg.norm(spectrum[i], keepdims=True)
        norm_spectrum[i] = spectrum[i] / out[i]
    return norm_spectrum


def extract_frequency(spectrum, min_freq, max_freq, number_windows):
    """
    Extracting/Filtering frequencies of the spectrum
    :param number_windows: number of windows
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

    extracted_spectrum = np.empty((number_windows, 222))
    for i in range(0, len(spectrum)):
        extracted_spectrum[i] = spectrum[i][idx]
    return extracted_spectrum


def calc_mean_intensity_acc(windows):
    """
    Calculates the mean acc intensity using the absolute Hilbert signal method
    :return: returns the mean acc intensity (envelope signal of the mean acc data)
    """

    mean_window_intensity = np.empty((len(windows), 1))

    for i in range(0, len(windows)):
        mean_window_intensity[i] = np.mean(np.abs(scipy.signal.hilbert(windows[i])))

    return mean_window_intensity

def gaussian_heart_rate(labels, num_samples=222, sigma=3, mu=0):
    out = []

    for i in range(0, len(labels)):
        gaussian_samples = np.empty((len(labels[i]), num_samples))
        for j in range(0, len(labels[i])):
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, num_samples)
            gaussian_samples[j] = labels[i][j] * np.exp(-0.5 * ((x - mu) / sigma)**2)
        out.append(gaussian_samples)

    return out

def custom_loss(y_true, y_pred, sigma=3):
    loss = (-1) * np.log(y_pred * ((np.exp(-((y_true**2)/(2*sigma**2))))/np.max(np.exp(-((y_true**2)/(2*sigma**2))))))

    return loss


if __name__ == '__main__':

    # create filter coefficients
    b_1, a_1, frequency, response = design_bandpass_filter(4, 0.4, 4, F)
    # visualize_filter(f=frequency, r=response)

    ####################################################################################################################

    power_spectra_ppg = []
    power_spectra_acc = []
    intensity_acc = []
    ground_truth = []
    number_windows = []

    for i in range(1, 25):
        # load data
        PPG, ACC, GT, t, w = load_data("BAMI-1/BAMI1_{}.mat".format(i))
        number_windows.append(len(GT))

        # save ground truth data
        # GT = np.expand_dims(GT, axis=2)
        # ground_truth = np.concatenate(GT)
        ground_truth.append(GT)

        # create the windows of all 3 PPGs
        windows_ppg_1 = create_windows(PPG[0], len(w), 2 * F, number_windows=len(GT))
        windows_ppg_2 = create_windows(PPG[1], len(w), 2 * F, number_windows=len(GT))
        windows_ppg_3 = create_windows(PPG[2], len(w), 2 * F, number_windows=len(GT))

        # filter all windows of all 3 PPGs
        filtered_windows_ppg_1 = filter_windows(windows_ppg_1, b_1, a_1, num_windows=len(GT))
        filtered_windows_ppg_2 = filter_windows(windows_ppg_2, b_1, a_1, num_windows=len(GT))
        filtered_windows_ppg_3 = filter_windows(windows_ppg_3, b_1, a_1, num_windows=len(GT))

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
        pad_windows_ppg = zero_pad(ds_norm_filtered_windows_ppg, 2048, number_windows=len(GT))

        # calculate the power spectrum
        power_spectrum_ppg = create_power_spectrum(pad_windows_ppg, number_windows=len(GT))

        # calculate the normalized power spectrum
        norm_power_spectrum_ppg = normalize_power_spectrum(power_spectrum_ppg, number_windows=len(GT))

        # extract the power spectrum in range between 0.6 and 3.3 Hz
        extract_power_spectrum_ppg = extract_frequency(norm_power_spectrum_ppg, 0.6, 3.3, number_windows=len(GT))

        # extract_power_spectrum_ppg = np.expand_dims(extract_power_spectrum_ppg, axis=2)
        power_spectra_ppg.append(extract_power_spectrum_ppg)

    ####################################################################################################################

        # create the windows of all 3 ACCs
        windows_acc_x = create_windows(ACC[0], len(w), 2 * F, number_windows=len(GT))
        windows_acc_y = create_windows(ACC[1], len(w), 2 * F, number_windows=len(GT))
        windows_acc_z = create_windows(ACC[2], len(w), 2 * F, number_windows=len(GT))

        # filter all windows of all 3 ACCs
        filtered_windows_acc_x = filter_windows(windows_acc_x, b_1, a_1, num_windows=len(GT))
        filtered_windows_acc_y = filter_windows(windows_acc_y, b_1, a_1, num_windows=len(GT))
        filtered_windows_acc_z = filter_windows(windows_acc_z, b_1, a_1, num_windows=len(GT))

        # resample the mean filtered windows
        ds_filtered_windows_acc_x = skimage.transform.resize(filtered_windows_acc_x,
                                                           (len(filtered_windows_acc_x), 200))
        ds_filtered_windows_acc_y = skimage.transform.resize(filtered_windows_acc_y,
                                                           (len(filtered_windows_acc_y), 200))
        ds_filtered_windows_acc_z = skimage.transform.resize(filtered_windows_acc_z,
                                                           (len(filtered_windows_acc_z), 200))

        # zero pad the down sampled windows to length of 2048
        pad_windows_acc_x = zero_pad(ds_filtered_windows_acc_x, 2048, number_windows=len(GT))
        pad_windows_acc_y = zero_pad(ds_filtered_windows_acc_y, 2048, number_windows=len(GT))
        pad_windows_acc_z = zero_pad(ds_filtered_windows_acc_z, 2048, number_windows=len(GT))

        # calculate the power spectrum
        power_spectrum_acc_x = create_power_spectrum(pad_windows_acc_x, number_windows=len(GT))
        power_spectrum_acc_y = create_power_spectrum(pad_windows_acc_y, number_windows=len(GT))
        power_spectrum_acc_z = create_power_spectrum(pad_windows_acc_z, number_windows=len(GT))

        # calculate the normalized power spectrum
        norm_power_spectrum_acc_x = normalize_power_spectrum(power_spectrum_acc_x, number_windows=len(GT))
        norm_power_spectrum_acc_y = normalize_power_spectrum(power_spectrum_acc_y, number_windows=len(GT))
        norm_power_spectrum_acc_z = normalize_power_spectrum(power_spectrum_acc_z, number_windows=len(GT))

        norm_power_spectrum_acc = np.mean([norm_power_spectrum_acc_x,
                                           norm_power_spectrum_acc_y,
                                           norm_power_spectrum_acc_z], axis=0)

        # extract the power spectrum in range between 0.6 and 3.3 Hz
        extract_power_spectrum_acc = extract_frequency(norm_power_spectrum_acc, 0.6, 3.3, number_windows=len(GT))

        # extract_power_spectrum_acc = np.expand_dims(extract_power_spectrum_acc, axis=2)
        power_spectra_acc.append(extract_power_spectrum_acc)

    ####################################################################################################################

        mean_intensity_acc_x = calc_mean_intensity_acc(ds_filtered_windows_acc_x)
        mean_intensity_acc_y = calc_mean_intensity_acc(ds_filtered_windows_acc_y)
        mean_intensity_acc_z = calc_mean_intensity_acc(ds_filtered_windows_acc_z)
        intensity_acc.append(np.mean([mean_intensity_acc_x, mean_intensity_acc_y, mean_intensity_acc_z], axis=0))

    ####################################################################################################################

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 37), strides=(4, 4),
                               activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                               input_shape=(2, 222, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1,
                               activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        tf.keras.layers.Reshape((19, 64)),  # reshape to (None, 19, 64) from 4D to 3D input shape
        tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="relu"),

        # Concatenation with acceleration intensity
        tf.keras.layers.Dense(1),
        tf.keras.layers.Reshape((1, 1)),
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x[1], axis=1)),

        tf.keras.layers.LSTM(512, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(222),
        tf.keras.layers.Dropout(0.2),
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

    labels_array = gaussian_heart_rate(ground_truth)

    PPG_data_array = np.zeros((len(power_spectra_ppg), np.max(number_windows), 222))
    ACC_data_array = np.zeros((len(power_spectra_acc), np.max(number_windows), 222))
    Labels_data_array = np.zeros((len(ground_truth), np.max(number_windows), 222))

    for i in range(0, len(power_spectra_ppg)):
        start = 0
        end = min(start + power_spectra_ppg[i].shape[0], PPG_data_array.shape[1])

        PPG_data_array[i][start:end, :] = power_spectra_ppg[i][:end - start, :]
        ACC_data_array[i][start:end, :] = power_spectra_acc[i][:end - start, :]
        Labels_data_array[i][start:end, :] = labels_array[i][:end - start, :]

    test = np.stack((PPG_data_array, ACC_data_array), axis=2)
