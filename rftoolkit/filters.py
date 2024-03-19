"""
Functions for returning filters on time domain and frequency domain waveforms.

@author: zackashm
"""

import numpy as np

from .waveform import Waveform


def wienerFilter(freqHz, signal, noise):
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
        of the noise.
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

    # Return Wiener filter
    return Sxx / (Sxx + Nxx)