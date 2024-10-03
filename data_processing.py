import numpy as np
import matplotlib.pyplot as plt


def confiy(data, percentage=1):
    # Saturate data to resemble cone
    shape = data.shape
    N, nd = shape
    coneN = round(N*percentage)

    rng = 2/np.ptp(data)
    
    step = 1/coneN

    new_data = data.copy()

    for i in range(nd):
        for idx, _ in enumerate(data[:,i]):
            if idx<coneN:
                new_data[idx,i] *= step*idx
            else:
                new_data[idx,i] *= rng
    return new_data

def plot_fft(signal, Ts, N=None, title=None):
    N = signal[0].shape[0] if N is None else N

    # Compute the corresponding frequencies
    freqs = np.fft.fftfreq(N, Ts)

    for s in signal:
        # Compute the FFT
        fft_values = np.fft.fft(s)
        
        # Only take the positive part of the spectrum and corresponding frequencies
        positive_freqs = freqs[:N // 2]
        positive_fft_values = 2.0 / N * np.abs(fft_values[:N // 2])
        
        # plt.semilogy(positive_freqs, positive_fft_values)
        plt.plot(positive_freqs, 20*np.log10(positive_fft_values/np.max(positive_fft_values)))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.xlim([np.min(np.abs(freqs)), np.max(np.abs(freqs))])#
    plt.grid()
    plt.show()


def add_gaussian_noise(signal, snr_dB):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_dB / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    noisy_signal = signal + noise
    
    return noisy_signal, noise

def generate_multi_sin_signal(n, f_min, f_max, A, N, Ts):
    time = np.arange(0, N+1) * Ts
    frequencies = np.random.uniform(f_min, f_max, n)

    phases = np.random.uniform(0, np.pi, n)
    
    signal = np.zeros_like(time)
    
    for freq, phase in zip(frequencies, phases):
        signal += np.sin(2 * np.pi * freq * time + phase)
    
    signal = signal[1:]
    
    # fit within [-A, A]
    signal = signal / np.max(np.abs(signal)) * A
    
    return signal.reshape(-1,1), frequencies, phases