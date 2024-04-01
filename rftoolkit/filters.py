"""
Functions for returning filters on time domain and frequency domain waveforms.

@author: zackashm
"""

import numpy as np
from scipy.signal import butter, freqz

from .waveform import Waveform


def wienerFilter(freqHz, signal, noise, alpha=0):
    '''
    Return the frequency-domain Wiener filter using Welch's Method.
    
    Parameters
    ----------
    freqHz : array
        The frequency array corresponding with the signal and noise.
    signal : array(2d), Waveform
        The [time (s), voltage (V)] waveform of the signal. Alternatively, a Waveform object
        of the signal.
    noise : array(2d), Waveform
        The [time (s), voltage (V)] waveform of the noise. Alternatively, a Waveform object
        of the noise.  The SNR parameter in the Wiener Filter is determined at each frequency 
        the power spectrum of the signal compared to that of the noise. The parameter alpha 
        provides a renormalization shift.
    alpha : [-1,1]
        Increase the noise power spectra to "renormalize" by adding the average of signal
        below alpha times the signal max.
        Nxx = Nxx + S'_mean, where S' is Sxx when Sxx < alpha*Sxx_max
        For example, alpha = 0.001 will bring the noise up by the signal's 3dB level.
        If alpha < 0, then the equation above is changed to subtraction.
        By default, this is 0, and hence no change is made to the noise.

    Returns
    -------
    ndarray
        The frequency domain [0,1] window calculated representing the Wiener filter.
    '''
    
    # convert to Waveform objects
    sigWfm = signal if isinstance(signal, Waveform) else Waveform(data=signal)
    noiseWfm = noise if isinstance(noise, Waveform) else Waveform(data=noise)
    
    # Estimate power spectral density using Welch's method
    # -- signal
    sigV = sigWfm.vdata
    sig_dt = 1/sigWfm.samplerate
    sig_dft = [sigV[i] * np.exp(-1j * 2 * np.pi * freqHz * i * sig_dt) for i in range(len(sigV))]
    Sxx = (sig_dt * np.abs(np.sum(sig_dft, axis=0)))**2 / len(sigV)
    
    # -- noise
    noiseV = noiseWfm.vdata
    noise_dt = 1/noiseWfm.samplerate
    noise_dft = [noiseV[i] * np.exp(-1j * 2 * np.pi * freqHz * i * noise_dt) for i in range(len(noiseV))]
    Nxx = (noise_dt * np.abs(np.sum(noise_dft, axis=0)))**2 / len(noiseV)

    if alpha > 0:
        Nxx = Nxx + Sxx[Sxx < alpha * Sxx.max()].mean()
    elif alpha < 0:
        Nxx = Nxx - Sxx[Sxx < -1*alpha * Sxx.max()].mean()

    # Return Wiener filter
    return Sxx / (Sxx + Nxx)


def butterworthFilter(lowcut=None, highcut=None, sampleRate=1, order=3, size=512):
    '''
    Return the frequency domain Butterworth filter. Provide lowcut for a highpass filter, or 
    highcut for a lowpass filter, or both for a bandpass filter.

    Parameters
    ----------
    lowcut : float, optional
        The cutoff frequency in Hz on the low end.
    highcut : float, optional
        The cutoff frequency in Hz on the high end.
    sampleRate : float
        The sampling rate of the time-domain waveform.
    order : int
        See scipy.signal.butter.
    size : int
        The number of points in the returned window.

    Returns
    -------
    freqHz, window : ndarray, ndarray
        The frequency array in Hz and the [0,1] window calculated representing the Butterworth filter.
    '''

    # calculate filter coefficients
    if lowcut is not None and highcut is not None:
        b, a = butter(order, [lowcut, highcut], btype='band', fs=sampleRate)
    elif lowcut is not None:
        b, a = butter(order, lowcut, btype='high', fs=sampleRate)
    elif highcut is not None:
        b, a = butter(order, highcut, btype='low', fs=sampleRate)
    else:
        raise ValueError('One or both of "lowcut" or "highcut" must be provided.')
    
    # get the frequency domain response
    freqHz, window = freqz(b, a, worN=size, fs=sampleRate)

    return (freqHz, np.abs(window))