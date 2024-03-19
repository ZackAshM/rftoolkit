"""
Gain and Impulse Repsonse functions from a transmitting antenna to receiving antenna measurement.

@author: zackashm
"""

from .waveform import Waveform
from .vna import S2P
from . import filters

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c

from warnings import warn

def gain(dataRx, dataPulse, dist, gainTx=None, freqStep='min', addComponentsS21=None, 
         subtractComponentsS21=None, verbose=1):
    '''
    Calculates the power gain [dB] in frequency from a time domain measurement where a pulse is sent from 
    Tx antenna to Rx antenna using the Friis Transmission Equation [1]:
        
        gainRx [dB] = FFT(P_Rx) [dB] - FFT(P_pulse) [dB] - gainTx [dB] + FSPL [dB]
        where 
        P_Rx = Vr^2 / 50 is the power of the received signal,
        P_pulse = Vp^2 / 50 is the power of the transmitted pulse,
        FSPL = (4 * pi * distance * frequency / c)^2 is the Free Space Path Loss, and
        dB = 10*log10( . )
    
    If gainTx is not given, it is assumed gainTx = gainRx and the equation adjusts to
        gainRx [dB] = 1/2 * ( FFT(P_Rx) [dB] - FFT(P_pulse) [dB] + FSPL [dB] )
    This is usually the case for identical antennas on both ends.
    
    If there are additional components such as cables or attenuators, their S21's should be measured over 
    the frequency band. In dB, these components are added directly.
    
    The calculation is evaluated over a common frequency space determined by the min and max frequency 
    of all given data, and frequency step set to the min or max of all data frequency steps (user's choice).
    
    In code, the steps for calculation are as follows:
    1. *Use rftoolkit.waveform.Waveform.calc_fft to get the relative gain of dataRx and dataPulse:
        dB = 10 * log10( abs(rfft)**2 / 50)
    2. Create scipy.interpolate.interp1d for each of the given data, using their own respective
       frequency values.
    3. Determine an evaluation frequency space using the min and max frequencies of all the given data
       and min or max frequency step of all given data.
    4. Calculate the gain using the Friis equation. Add the additional components directly.
    
    Phase is not considered for this calculation. For phase consideration, see rftoolkit.tx2rx.impulseResponse.
    
    *The data are converted to Waveform objects if not given as Waveform. A standard procedure would be
    cleaning the data as Waveform objects before calling this function.

    Parameters
    ----------
    dataRx : (2,1d) array, or Waveform
        An array containing [[time (s)], [voltage (V)]] data received. Can also be given as a
        rftoolkit.waveform.Waveform object.
    dataPulse : (2,1d) array, or Waveform
        An array containing [[time (s)], [voltage (V)]] data transmitted. Can also be given as a
        rftoolkit.waveform.Waveform object.
    dist : float, or (2, 1d) array
        The distance in meters used in the free space path loss. This can be a number to be used over
        all frequencies, or [[frequency (Hz)], [distance (m)]].
    gainTx : (2,1d) array, None
        An array containing [[frequency (Hz)], [gain (dBi)]] for the transmitting antenna. If None, it
        is assumed gainTx = gainRx, and the calculation is adjusted accordingly.
    freqStep : "min", "max"
        Determines the freqeuncy step for the evaluation frequency. If "min" ("max"), the frequency step will be 
        the minimum (maximum) frequency step of all the given data. Default is "min".
    addComponentsS21 : list[(2,1d) array], optional
        A list of S21 data in [[frequency in Hz], [gain in dBm]] associated with additional components such 
        as cables or attenuators which add to the Rx gain. All components are assumed 50 Ohms, so the S21's 
        are simply added (in dB).
    subtractComponentsS21 : list[(2,1d) array], optional
        A list of S21 data in [[frequency in Hz], [gain in dBm]] associated with additional components such 
        as cables or attenuators which subtract from the Rx gain. All components are assumed 50 Ohms, so 
        the S21's are simply subtracted (in dB).
    verbose : int, optional
        If 0: No print messages. If 1: Progress messages are printed at each step. Default is 1.

    Returns
    -------
    evalFreqHz : array
        Frequency array in Hz used for evaluation. This corresponds to the smallest band overlap and chosen
        min or max frequency step of all the data.
    gainRx : array
        Gain array in dB.
    
    See Also
    --------
    rftoolkit.tx2rx.impulseResponse
        For gain characterization with inclusion of phase.
    
    References
    ----------
    [1] H. T. Friis, "A Note on a Simple Transmission Formula," in Proceedings of the IRE, vol. 34, no. 5, pp. 
        254-256, May 1946, doi: 10.1109/JRPROC.1946.234568. 
    '''
    
    if verbose > 0:
        print('Starting gain from pulse calculation')
        
    # handle optional args
    identicalAntennas = (gainTx is None)
    addComps = not (addComponentsS21 is None)
    subtractComps = not (subtractComponentsS21 is None)
    if (identicalAntennas) and (verbose > 0):
        print('GainTx not given; assuming identical antenna calculation')
    if (not addComps) and (verbose > 0):
        print('Found no component S21s to add')
    if (not subtractComps) and (verbose > 0):
        print('Found no component S21s to subtract')
    if freqStep not in ['min', 'max']:
        raise AttributeError(f'Unrecognized argument "{freqStep}" for freqStep.')
    
    # Step 0. parse and convert data
    
    # -- convert to waveform objects
    wfmRx = dataRx if isinstance(dataRx, Waveform) else Waveform(data=dataRx)
    wfmPulse = dataPulse if isinstance(dataPulse, Waveform) else Waveform(data=dataPulse)
    
    # -- convert additional components to S2P objects
    if addComps:
        if verbose > 0:
            print('Checking and intializing addComponentsS21')
        addComponentsS21 = np.array([addComponentsS21]) if len(np.array(addComponentsS21).shape) == 2 else np.array(addComponentsS21)
        addCompsdB = []
        for comp in addComponentsS21:
            if isinstance(comp, S2P):
                addCompsdB.append([comp.fHz, comp.s21magdb])
            else:
                addCompsdB.append([comp[0], comp[1]])
    if subtractComps:
        if verbose > 0:
            print('Checking and intializing subtractComponentsS21')
        subtractComponentsS21 = np.array([subtractComponentsS21]) if len(np.array(subtractComponentsS21).shape) == 2 else np.array(subtractComponentsS21)
        subtractCompsdB = []
        for comp in subtractComponentsS21:
            if isinstance(comp, S2P):
                subtractCompsdB.append([comp.fHz, comp.s21magdb])
            else:
                subtractCompsdB.append([comp[0], comp[1]])
        
    # Step 1. Get FFTs
    if verbose > 0:
        print('Calculating FFTs')
    fftRx = wfmRx.calc_fft(rfft=True)[0::2] # Hz, dB
    fftPulse = wfmPulse.calc_fft(rfft=True)[0::2] # Hz, dB
    
    # Step 2. Interpolate
    if verbose > 0:
        print('Interpolating')
    
    # -- find the evaluation frequency:
    if freqStep == 'min':
        getStep = lambda freqSamp1, freqSamp2: min((freqSamp1, freqSamp2))
    elif freqStep == 'max':
        getStep = lambda freqSamp1, freqSamp2: max((freqSamp1, freqSamp2))
    
    evalFreqMin = max((fftRx[0][0], fftPulse[0][0]))
    evalFreqMax = min((fftRx[0][-1], fftPulse[0][-1]))
    evalFreqSamp = getStep(np.float64('{:0.5e}'.format(np.diff(fftRx[0]).mean())), 
                            np.float64('{:0.5e}'.format(np.diff(fftPulse[0]).mean())))
    
    if not identicalAntennas:
        evalFreqMin = max((evalFreqMin, gainTx[0][0]))
        evalFreqMax = min((evalFreqMax, gainTx[0][-1]))
        evalFreqSamp = getStep(evalFreqSamp, np.float64('{:0.5e}'.format(np.diff(gainTx[0]).mean())))
        
    if addComps:
        for addCompS21 in addCompsdB:
            evalFreqMin = max((evalFreqMin, addCompS21[0][0]))
            evalFreqMax = min((evalFreqMax, addCompS21[0][-1]))
            evalFreqSamp = getStep(evalFreqSamp, np.float64('{:0.5e}'.format(np.diff(addCompS21[0]).mean())))
    if subtractComps:
        for subtractCompS21 in subtractCompsdB:
            evalFreqMin = max((evalFreqMin, subtractCompS21[0][0]))
            evalFreqMax = min((evalFreqMax, subtractCompS21[0][-1]))
            evalFreqSamp = getStep(evalFreqSamp, np.float64('{:0.5e}'.format(np.diff(subtractCompS21[0]).mean())))
      
    evalFreqHz = np.arange(evalFreqMin+evalFreqSamp, evalFreqMax, evalFreqSamp)
    if verbose > 0:
        print(f'Obtained evaluation frequency of (min, max, step): ({evalFreqMin}, {evalFreqMax}, {evalFreqSamp}) Hz')
    
    # -- interpolate
    interpfftRx = interp1d(fftRx[0], fftRx[1])
    interpfftPulse = interp1d(fftPulse[0], fftPulse[1])
    if not identicalAntennas:
        interpGainTx = interp1d(gainTx[0], gainTx[1])
    if addComps:
        interpAddCompsS21 = []
        for addCompS21 in addCompsdB:
            interpAddCompsS21.append(interp1d(addCompS21[0],addCompS21[1]))
    if subtractComps:
        interpSubtractCompsS21 = []
        for subtractCompS21 in subtractCompsdB:
            interpSubtractCompsS21.append(interp1d(subtractCompS21[0],subtractCompS21[1]))
    
    # Step 3. Calculate Gain
    if verbose > 0:
        print('Interpolation done. Calculating gain')
    
    # -- prepare physical term
    if len(np.array(dist).shape) == 0: # just a number
        evalDist = dist * np.ones(evalFreqHz.size)
    else:
        evalDist = interp1d(dist[0], dist[1], fill_value='extrapolate', kind='quadratic')(evalFreqHz)
    
    # -- -- the friis term containing the info about the physical environment
    friisTerm = 20 * np.log10( np.abs((4 * np.pi * evalDist * evalFreqHz) / c ) ) # in dB
    
    # -- calculate gain
    gainRx = interpfftRx(evalFreqHz) - interpfftPulse(evalFreqHz) + friisTerm
    
    if addComps:
        for addCompS21 in interpAddCompsS21:
            gainRx += addCompS21(evalFreqHz)
    if subtractComps:
        for subtractCompS21 in interpSubtractCompsS21:
            gainRx += -1 * subtractCompS21(evalFreqHz)
    
    if identicalAntennas:
        gainRx *= 0.5
    else:
        gainRx += -1 * interpGainTx(evalFreqHz)
    
    # -- return freq and gain
    if verbose > 0:
        print('Gain calculation complete')
        
    return (evalFreqHz, gainRx)

def impulseResponse(dataRx, dataPulse, dist, gainTx=None, freqStep='min', addComponentsS21=None, 
                    subtractComponentsS21=None, wienerFilter=False, noise=None, 
                    returnDomain='time', verbose=1):
    '''
    Calculates the time domain impulse response from a time domain measurement where a pulse is sent from 
    Tx antenna to Rx antenna using the frequency domain formula [1]:
        
        h_Rx = (distance * c / j * frequency) * (V_received / V_pulse) * (1 / h_Tx)
        
    If gainTx (-->h_Tx) is not given, it is assumed gainTx = gainRx and the equation adjusts to
        h_Rx = sqrt( (distance * c / j * frequency) * (V_received / V_pulse) )
    with care taken on unwrapping complex phase before taking the square root.
    This is usually the case for identical antennas on both ends.
    
    If there are additional components such as cables or attenuators, their S21's should be measured over 
    the frequency band. The complex values of addComponentsS21 are multiplied, and the complex values
    of subtractComponentsS21 are divided.
    
    The calculation is evaluated over a common frequency space determined by the min and max frequency 
    of all given data, and frequency step set to the min or max of all data frequency steps (user's choice).
    
    In code, the steps for calculation are as follows:
    1. *Use rftoolkit.waveform.Waveform.calc_fft to get the frequency-domain complex FFT of dataRx and 
       dataPulse.
    2. Create scipy.interpolate.interp1d for each of the given data in frequency domain for the real and
       imaginary parts.
    3. Determine an evaluation frequency space using the min and max frequencies of all the given data
       and min or max frequency step of all given data.
    4. Calculate the impulse response using the formula above.
    5. Perform inverse FFT on the result to get the impulse response in the time domain.
    
    *The data are converted to Waveform objects if not given as Waveform. A standard procedure would be
    cleaning the data as Waveform objects before calling this function.

    Parameters
    ----------
    dataRx : (2,1d) array, or Waveform
        An array containing [[time (s)], [voltage (V)]] data received. Can also be given as a
        rftoolkit.waveform.Waveform object.
    dataPulse : (2,1d) array, or Waveform
        An array containing [[time (s)], [voltage (V)]] data transmitted. Can also be given as a
        rftoolkit.waveform.Waveform object.
    dist : float, or (2, 1d) array
        The distance in meters used in the free space path loss. This can be a number to be used over
        all frequencies, or [[frequency (Hz)], [distance (m)]].
    gainTx : (2,1d) array, None
        The impulse response array containing [[frequency (Hz)], [gain (dBi)], [phase (deg)]] for the 
        transmitting antenna. If None, it is assumed gainTx = gainRx, and the calculation is adjusted accordingly.
        Note that |h_Tx| = (c / freq) * sqrt( (1/4pi) 10**(gain/10) ) is the corresponding magnitude.
    freqStep : "min", "max"
        Determines the freqeuncy step for the evaluation frequency. If "min" ("max"), the frequency step will be 
        the minimum (maximum) frequency step of all the given data. Default is "min".
    addComponentsS21 : list[(2,1d) array], optional
        A list of S21 data in [[frequency (Hz)], [gain (dBm)], [phase (deg)]] associated with additional 
        components such as cables or attenuators which add to the Rx gain. All components are assumed 
        50 Ohms, so the complex valued S21's are multiplied.
    subtractComponentsS21 : list[(2,1d) array], optional
        A list of S21 data in [[frequency (Hz)], [gain (dBm)], [phase (deg)]] associated with additional 
        components such as cables or attenuators which subtract from the Rx gain. All components are assumed 
        50 Ohms, so the complex valued S21's are divided.
    wienerFilter : bool, optional
        If True, a wiener filter [2] is applied to account for noise and high frequency artifacts. For SNR estimates, 
        set the noise window using nsNoiseWindow. Default is False.
    noise : tuple(2), array(2d), Waveform, None
        When wienerFilter is True, this is the noise with which to determine SNR in the Wiener filter. A tuple of 
        length 2 can be given to describe a nanosecond window in the Rx waveform where a noise waveform is defined. 
        If the window is partially outside of the data, it will be truncated. If the window is completely outside 
        the data, or if None, the noise window is taken to be the first 5% of the waveform.
        Alternatively, a [time (s), voltage (V)] or Waveform object can be provided which are directly the noise
        waveforms to use.
    returnDomain : "time", "frequency", "both"
        The domain for which the impulse response is returned. If "time", the real-valued inverse FFT is returned 
        with a corresponding time domain array. If "frequency", the complex-valued FFT result is returned with
        the evaluation frequency array. If "both", then both time and frequency results are returned.
        Default is "time".
    verbose : int, optional
        If 0: No print messages. If 1: Only warning messages are given. 
        If 2: Progress messages are printed at each step. Default is 1.

    Returns
    -------
    time : array
        Returned when returnDomain is 'time' or 'both'. Time in seconds array starting at 0. The time step and 
        end time are determined by the determined evaluation frequency:
            time = np.arange(0,evalFreqHz.size*2-2) / (2*evalFreqHz.size*evalFreqSamp)
    impulseResponseRx : array
        Returned when returnDomain is 'time' or 'both'. Real valued impulse response in the time domain in 
        meters per second.
    evalFreqFFT : array
        Returned when returnDomain is 'frequency' or 'both'. Frequency array in Hz used for evaluation, 
        zeropadded to 0 Hz.
    impulseResponseRxFFT : array
        Returned when returnDomain is 'frequency' or 'both'. The complex valued impulse response in the 
        frequency domain in meters.
    
    See Also
    --------
    rftoolkit.tx2rx.gain
        For power gain characterization using only dB magnitude data.
    
    References
    ----------
    [1] B. Scheers, M. Acheroy and A. Vander Vorst, “Time domain simulation and
        characterisation of TEM horns using normalised impulse response,” IEE Proceedings
        - Microwaves, Antennas and Propagation, vol. 147, no. 6, pp. 463-468, Dec. 2000.
    [2] William H. Press, Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery. 2007. 
        Numerical Recipes 3rd Edition: The Art of Scientific Computing (3rd. ed.). Chapter 13.
        Cambridge University Press, USA.
    '''
    
    if verbose > 1:
        print('Starting impulse response calculation')
        
    # handle optional args
    identicalAntennas = (gainTx is None)
    addComps = not (addComponentsS21 is None)
    subtractComps = not (subtractComponentsS21 is None)
    if (identicalAntennas) and (verbose > 1):
        print('GainTx not given; assuming identical antenna calculation')
    if (not addComps) and (verbose > 1):
        print('Found no component S21s to add')
    if (not subtractComps) and (verbose > 1):
        print('Found no component S21s to subtract')
    if freqStep not in ['min', 'max']:
        raise AttributeError(f'Unrecognized argument "{freqStep}" for freqStep.')
    if returnDomain not in ['time', 'frequency', 'both']:
        raise AttributeError(f'Unrecognized argument "{returnDomain}" for returnDomain.')
    
    # Step 0. parse and convert data
    
    # -- convert to waveform objects
    wfmRx = dataRx if isinstance(dataRx, Waveform) else Waveform(data=dataRx)
    wfmPulse = dataPulse if isinstance(dataPulse, Waveform) else Waveform(data=dataPulse)
    
    # -- convert additional components to S2P objects
    if addComps:
        if verbose > 1:
            print('Checking and intializing addComponentsS21')
        addComponentsS21 = np.array([addComponentsS21]) if len(np.array(addComponentsS21).shape) == 2 else np.array(addComponentsS21)
        addCompsdB = []
        for comp in addComponentsS21:
            if isinstance(comp, S2P):
                addCompsdB.append([comp.fHz, comp.s21magdb, comp.s21phasedeg])
            else:
                addCompsdB.append([comp[0], comp[1], comp[2]])
    if subtractComps:
        if verbose > 1:
            print('Checking and intializing subtractComponentsS21')
        subtractComponentsS21 = np.array([subtractComponentsS21]) if len(np.array(subtractComponentsS21).shape) == 2 else np.array(subtractComponentsS21)
        subtractCompsdB = []
        for comp in subtractComponentsS21:
            if isinstance(comp, S2P):
                subtractCompsdB.append([comp.fHz, comp.s21magdb, comp.s21phasedeg])
            else:
                subtractCompsdB.append([comp[0], comp[1], comp[2]])
        
    # Step 1. Get FFTs
    if verbose > 1:
        print('Calculating FFTs')
    fftRx = wfmRx.calc_fft(rfft=True)[0:2] # Hz, Complex mag
    fftPulse = wfmPulse.calc_fft(rfft=True)[0:2] # Hz, Complex mag
    
    # Step 2. Interpolate
    if verbose > 1:
        print('Interpolating')
    
    # -- find the evaluation frequency:
    if freqStep == 'min':
        getStep = lambda freqSamp1, freqSamp2: min((freqSamp1, freqSamp2))
    elif freqStep == 'max':
        getStep = lambda freqSamp1, freqSamp2: max((freqSamp1, freqSamp2))
    
    evalFreqMin = max((fftRx[0][0], fftPulse[0][0]))
    evalFreqMax = min((fftRx[0][-1], fftPulse[0][-1]))
    evalFreqSamp = getStep(np.float64('{:0.5e}'.format(np.diff(fftRx[0]).mean())), 
                            np.float64('{:0.5e}'.format(np.diff(fftPulse[0]).mean())))
    
    if not identicalAntennas:
        evalFreqMin = max((evalFreqMin, gainTx[0][0]))
        evalFreqMax = min((evalFreqMax, gainTx[0][-1]))
        evalFreqSamp = getStep(evalFreqSamp, np.float64('{:0.5e}'.format(np.diff(gainTx[0]).mean())))
        
    if addComps:
        for addCompS21 in addCompsdB:
            evalFreqMin = max((evalFreqMin, addCompS21[0][0]))
            evalFreqMax = min((evalFreqMax, addCompS21[0][-1]))
            evalFreqSamp = getStep(evalFreqSamp, np.float64('{:0.5e}'.format(np.diff(addCompS21[0]).mean())))
    if subtractComps:
        for subtractCompS21 in subtractCompsdB:
            evalFreqMin = max((evalFreqMin, subtractCompS21[0][0]))
            evalFreqMax = min((evalFreqMax, subtractCompS21[0][-1]))
            evalFreqSamp = getStep(evalFreqSamp, np.float64('{:0.5e}'.format(np.diff(subtractCompS21[0]).mean())))
      
    evalFreqHz = np.arange(evalFreqMin+evalFreqSamp, evalFreqMax, evalFreqSamp)
    if verbose > 1:
        print(f'Obtained evaluation frequency of (min, max, step): ({evalFreqMin}, {evalFreqMax}, {evalFreqSamp}) Hz')
    
    # Step 3. Interpolate real and imag data
    if verbose > 1:
        print('Interpolating real and imaginary values of the data FFTs')
    interpRxReal = interp1d(fftRx[0], np.real(fftRx[1]))
    interpRxImag = interp1d(fftRx[0], np.imag(fftRx[1]))
    interpPulseReal = interp1d(fftPulse[0], np.real(fftPulse[1]))
    interpPulseImag = interp1d(fftPulse[0], np.imag(fftPulse[1]))
    
    toComplex = lambda dB, deg: 10**(dB / 20) * np.exp(1j * np.unwrap(np.deg2rad(deg), period=np.pi))
    if not identicalAntennas:
        hTxMagdB = gainTx[1] - 10*np.log10(4*np.pi*(gainTx[0]/c)**2) # = |hTx|^2 in dB, i.e. linmag is 10**(dB/20)
        complexGainTx = toComplex(hTxMagdB, gainTx[2])
        interpGainTxReal = interp1d(gainTx[0], np.real(complexGainTx))
        interpGainTxImag = interp1d(gainTx[0], np.imag(complexGainTx))
    if addComps:
        interpAddCompsS21Real = []
        interpAddCompsS21Imag = []
        for addCompS21 in addCompsdB:
            complexAddCompS21 = toComplex(addCompS21[1], addCompS21[2])
            interpAddCompsS21Real.append(interp1d(addCompS21[0], np.real(complexAddCompS21)))
            interpAddCompsS21Imag.append(interp1d(addCompS21[0], np.imag(complexAddCompS21)))
    if subtractComps:
        interpSubtractCompsS21Real = []
        interpSubtractCompsS21Imag = []
        for subtractCompS21 in subtractCompsdB:
            complexSubtractCompS21 = toComplex(subtractCompS21[1], subtractCompS21[2])
            interpSubtractCompsS21Real.append(interp1d(subtractCompS21[0], np.real(complexSubtractCompS21)))
            interpSubtractCompsS21Imag.append(interp1d(subtractCompS21[0], np.imag(complexSubtractCompS21)))
    
    if verbose > 1:
        print('Interpolation done')
    
    # -- prepare physical term
    if len(np.array(dist).shape) == 0: # just a number
        evalDist = dist * np.ones(evalFreqHz.size)
    else:
        evalDist = interp1d(dist[0], dist[1], fill_value='extrapolate', kind='linear')(evalFreqHz)
    
    
    # Step 4. Perform deconvolution
    if verbose > 1:
        print('Performing deconvolution via FFT division')
            
    VrFFT = interpRxReal(evalFreqHz) + 1j*interpRxImag(evalFreqHz) # received data
    VpFFT = interpPulseReal(evalFreqHz) + 1j*interpPulseImag(evalFreqHz) # pulse data
    physFactor = (c * evalDist) / (1j * evalFreqHz) # physical factor from path loss, etc.
    
    addCompResponse = 1
    subtractCompResponse = 1
    if addComps:
        for addCompReal, addCompImag in zip(interpAddCompsS21Real, interpAddCompsS21Imag):
            addCompResponse *= addCompReal(evalFreqHz) + 1j*addCompImag(evalFreqHz)
    if subtractComps:
        for subtractCompReal, subtractCompImag in zip(interpSubtractCompsS21Real, interpSubtractCompsS21Imag):
            subtractCompResponse *= subtractCompReal(evalFreqHz) + 1j*subtractCompImag(evalFreqHz)
    
    impulseResponseRxFFT_1 = physFactor * (addCompResponse / subtractCompResponse) * (VrFFT / VpFFT)
        
    # -- Calculate Wiener Filter
    if wienerFilter:
        if verbose > 1:
            print('Applying Wiener filter.')

        # Get noise data from Rx
        if noise is None: # default is first 5% of window
            nsTime = wfmRx.tdata * 1e9
            nsNoiseWindow = (nsTime[0], nsTime[int(0.05*len(nsTime))])
            if verbose > 1:
                print(f'Noise window defaulting to {nsNoiseWindow}')
            wfmRxNoise = wfmRx.copy() # new waveform set to hold noise data
            wfmRxNoise.truncate(nsNoiseWindow)
        elif isinstance(noise, tuple): # check if the noise window is outside of the domain
            nsTime = wfmRx.tdata * 1e9
            if (noise[1] < nsTime[0]) or (noise[0] > nsTime[-1]): 
                nsNoiseWindow = (nsTime[0], nsTime[int(0.05*len(nsTime))])
                if verbose > 0:
                    warn(f'Requested noise window fully outside signal time domain. Defaulting to {nsNoiseWindow}')
            else:
                nsNoiseWindow = noise
            wfmRxNoise = wfmRx.copy() # new waveform set to hold noise data
            wfmRxNoise.truncate(nsNoiseWindow)
        else:
            wfmRxNoise = noise if isinstance(noise, Waveform) else Waveform(data=noise)
        
        wienerFilterFactor = filters.wienerFilter(evalFreqHz, wfmRx, wfmRxNoise)
        
        impulseResponseRxFFT_1 *= wienerFilterFactor

        if verbose > 1:
            print('Completed Wiener filtering.')
    
    # -- Complete calculation based on presence of separate gain antenna
    if identicalAntennas:
        if verbose > 1:
            print('No gain of a transmitting antenna given - performing square root.')
        impulseResponseRxFFTmag = np.sqrt(np.abs(impulseResponseRxFFT_1))
        impulseResponseRxFFTphase = np.unwrap(np.angle(impulseResponseRxFFT_1), period=np.pi) / 2
        impulseResponseRxFFT = impulseResponseRxFFTmag * np.exp(1j * impulseResponseRxFFTphase)
    else:
        if verbose > 1:
            print('Gain of a transmitting antenna given - performing division.')
        Gt = interpGainTxReal(evalFreqHz) + 1j*interpGainTxImag(evalFreqHz)
        impulseResponseRxFFT = impulseResponseRxFFT_1 / Gt
    
    if verbose > 1:
        print('Completed impulse response calculation in frequency domain.')
        
    if returnDomain == 'frequency':
        if verbose > 1:
            print('Returning frequency domain result.')
        return (evalFreqHz, impulseResponseRxFFT)
    
    # Step 5. Convert to time domain
    if verbose > 1:
        print('Calculating in time domain.')
        
    # -- zero pad to 0 Hz
    freqPad = -1 * np.arange(-1*evalFreqHz[0], evalFreqSamp, evalFreqSamp)[-1:0:-1]
    freqPad[0] = 0 # set zero-closest freq to 0
    if verbose > 1:
        print(f'Zero-padding frequencies {freqPad}')
    evalFreqPadded = np.append(freqPad, evalFreqHz, axis=0)
    
    impulseResponseRxFFTPadded = np.pad(impulseResponseRxFFT, (len(freqPad),0), constant_values=0)
    
    dt = 1 / (2*evalFreqPadded.size*evalFreqSamp)
    time = np.arange(0,evalFreqPadded.size*2-2) * dt
    
    impulseResponseRx = np.real(np.fft.irfft(impulseResponseRxFFTPadded))*2 / dt
    
    if returnDomain == 'time':
        return (time, impulseResponseRx)
    elif returnDomain == 'both':
        return (time, impulseResponseRx, evalFreqHz, impulseResponseRxFFT)
