"""
Toolkit for handling 2 port s-parameter data from a VNA.

@author: zackashm
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from warnings import warn

class S2P:
    '''
    Data parsing and analysis functions for data from an s2p file saved from a VNA.
    Manually inputted data can also be fed in.
    
    Parameters
    ----------
    file : str, optional
        The full file path of the s2p data. This overrides manual data input.
    data : array(9), optional
        A manual input array of 9 data arrays. The order of each data array matches the order
        of data arrays in an s2p file. Arrays should match size. If an empty array is found, it is replaced
        by an array of zeros corresponding to the frequency (0th) array. Not used if file is given.
    
    Attributes
    ----------
    header : str
        The header data in the s2p data.
    fullData : numpy.ndarray(9)
        A list of the data arrays: [freq, s11 mag, s11 phase, s12 mag, s12 phase, 
                                    s21 mag, s21 phase, s22 mag, s22 phase]
    fHz : numpy.ndarray
        The frequency array in Hz.
    s[XX]mag[db/lin] : numpy.ndarray
        The s-parameter magnitude array. Replace XX with the requested s-parameter, and choose between 
        "db" and "lin" for db magnitude or voltage magnitude.
    s[XX]phase[deg/rad] : numpy.ndarray
        The s-parameter wrapped phase array. Replace XX with the requested s-parameter, and choose between 
        "deg" and "rad" for degrees or radians.
    s[XX]complex : numpy.ndarray
        The s-parameter complex-valued array. Calculated by
            10**(magdb / 20) * np.exp(i * np.unwrap(phaserad))
        Replace XX with the requested s-parameter.
    s[XX]groupdelay : numpy.ndarray
        The s-parameter calculated group delay in seconds. Equal to -d(phase / 2pi)/df
    freqstep : float
        The average frequency step size in Hz.
    
    Methods
    -------
    smooth
        Smooth the given s-parameter curves via rolling average.
    plotAll
        Plot each s-parameter, mag and phase separated.
    
    '''
    
    def __init__(self, file=None, data=None):
        
        if file:
            self.file = file
            
            # header info
            self.header = ''
            for line_number, line in enumerate(open(file)):
                self.header += line[1:]
                if line_number > 10: break
            
            # save copy of data
            self._RAWDATA = np.loadtxt(file, skiprows=12, unpack=True)
            self.fullData = self._RAWDATA.copy()
            
        elif data: # input data
        
            # handle input errors
            if np.array(data, dtype=object).shape[0] != 9:
                raise ValueError('Data must have 9 associated arrays in the same order as an s2p file.')
            if not list(data[0]):
                raise ValueError('Data must have a frequency array in its first index.')
            for ind, d in enumerate(data[1:]):
                if not (len(d) == len(data[0])) and (len(d)>0):
                    warn(f'The data[{ind}] array does not match the given frequency array. It has been discarded.')
                    
            self.file = ''
            self.header = ''
            self._RAWDATA = np.array([np.array(d) if len(d)==len(data[0]) else np.zeros(len(data[0])) for d in data]).copy()
        
        else: # empty data
            self.file = ''
            self.header = ''
            self._RAWDATA = np.zeros((9,1))
            
        
        self.fullData = self._RAWDATA.copy()
        
        # parse data
        self.fHz = self.fullData[0]
        self.s11magdb, self.s11phasedeg = self.fullData[1:3]
        self.s12magdb, self.s12phasedeg = self.fullData[3:5]
        self.s21magdb, self.s21phasedeg = self.fullData[5:7]
        self.s22magdb, self.s22phasedeg = self.fullData[7:9]
        
    @property
    def s11maglin(self):
        return 10**(self.s11magdb / 20)
    
    @property
    def s12maglin(self):
        return 10**(self.s12magdb / 20)
    
    @property
    def s21maglin(self):
        return 10**(self.s21magdb / 20)
    
    @property
    def s22maglin(self):
        return 10**(self.s22magdb / 20)
    
    @property
    def s11phaserad(self):
        return np.deg2rad(self.s11phasedeg)
    
    @property
    def s12phaserad(self):
        return np.deg2rad(self.s12phasedeg)
    
    @property
    def s21phaserad(self):
        return np.deg2rad(self.s21phasedeg)
    
    @property
    def s22phaserad(self):
        return np.deg2rad(self.s22phasedeg)
    
    @property
    def s11complex(self):
        return self.s11maglin * np.exp(1j * np.unwrap(self.s11phaserad, period=2*np.pi))
    
    @property
    def s12complex(self):
        return self.s12maglin * np.exp(1j * np.unwrap(self.s12phaserad, period=2*np.pi))
    
    @property
    def s21complex(self):
        return self.s21maglin * np.exp(1j * np.unwrap(self.s21phaserad, period=2*np.pi))
    
    @property
    def s22complex(self):
        return self.s22maglin * np.exp(1j * np.unwrap(self.s22phaserad, period=2*np.pi))
    
    @property
    def s11groupdelay(self):
        return -1*np.gradient(np.unwrap(self.s11phaserad, period=2*np.pi), 2*np.pi*self.freqstep)
    
    @property
    def s12groupdelay(self):
        return -1*np.gradient(np.unwrap(self.s12phaserad, period=2*np.pi), 2*np.pi*self.freqstep)
    
    @property
    def s21groupdelay(self):
        return -1*np.gradient(np.unwrap(self.s21phaserad, period=2*np.pi), 2*np.pi*self.freqstep)
    
    @property
    def s22groupdelay(self):
        return -1*np.gradient(np.unwrap(self.s22phaserad, period=2*np.pi), 2*np.pi*self.freqstep)

    @property
    def freqstep(self):
        return np.float64('{:0.5e}'.format(np.diff(self.fHz).mean()))
    
    @property
    def size(self):
        return self.fullData.shape[1]
    
    
    def reset(self):
        '''
        Set the data back to its original state.
        '''
        del self.fullData
        self.fullData = self._RAWDATA.copy()
        
        self.fHz = self.fullData[0]
        self.s11magdb, self.s11phasedeg = self.fullData[1:3]
        self.s12magdb, self.s12phasedeg = self.fullData[3:5]
        self.s21magdb, self.s21phasedeg = self.fullData[5:7]
        self.s22magdb, self.s22phasedeg = self.fullData[7:9]
    
    
    def smooth(self, sparms=['11', '12', '21', '22'], window_length=None, polyorder=2, **kwargs):
        '''
        Smooth the given s-parameter data curves via scipy.signal.savgol_filter.
        
        Parameters
        ----------
        sparms : list['11', '12', '21', '22']
            A list of requested s-parameter data to smooth.
        window_length : int, None
            See scipy.signal.savgol_filter. If None, the window length is chosen to
            be 10% of the data length.
        polyorder : int
            See scipy.signal.savgol_filter.
        kwargs
            Kwargs for scipy.signal.savgol_filter.
        '''
        
        window_length = int(0.1 * len(self.fHz))
        if window_length % 2 == 0: # make odd for behaved savgol_filter result
            window_length += 1
        _smooth = lambda data: savgol_filter(data, window_length, polyorder)
        _rewrap = lambda theta: (theta + np.pi) % (2 * np.pi) - np.pi # wrap between -pi and pi
        
        if '11' in sparms:
            self.fullData[1] = _smooth(self.fullData[1])
            self.fullData[2] = np.rad2deg(_rewrap(_smooth(np.deg2rad(np.unwrap(self.fullData[2], period=360)))))
            self.s11magdb, self.s11phasedeg = self.fullData[1:3]
        if '12' in sparms:
            self.fullData[3] = _smooth(self.fullData[3])
            self.fullData[4] = np.rad2deg(_rewrap(_smooth(np.deg2rad(np.unwrap(self.fullData[4], period=360)))))
            self.s12magdb, self.s12phasedeg = self.fullData[3:5]
        if '21' in sparms:
            self.fullData[5] = _smooth(self.fullData[5])
            self.fullData[6] = np.rad2deg(_rewrap(_smooth(np.deg2rad(np.unwrap(self.fullData[6], period=360)))))
            self.s21magdb, self.s21phasedeg = self.fullData[5:7]
        if '22' in sparms:
            self.fullData[7] = _smooth(self.fullData[7])
            self.fullData[8] = np.rad2deg(_rewrap(_smooth(np.deg2rad(np.unwrap(self.fullData[8], period=360)))))
            self.s22magdb, self.s22phasedeg = self.fullData[7:9]

    def plotAll(self, fscale=1, unwrap=True, maglin=False, phaserad=False, overlay=False, title=None, ax=None, label=None):
        '''
        Plot all s-parameters.
        
        Parameters
        ----------
        fscale : float, optional
            A scaling factor on the frequency. The units will adjust to common scalings:
            '1' --> 'Hz', '1e-6' --> 'MHz', '1e-9' --> 'GHz'
            Default is 1.
        unwrap : bool, optional
            If True, the plotted phases are unwrapped.
        maglin : bool, optional
            If True, plot using linear units of magnitude.
        phaserad : bool, optional
            If True, plot using radian units for phase.
        overlay : bool, optional
            If True, all s-parameter magnitudes are plotted together and all phases are plotted together.
        title : str, None, optional
            Plot title. If None, the file name is used. Default is None.
        ax : matplotlib.pyplot.Axes, None, optional
            The Axes object to plot on. If None, an Axes object is created.
        label : str, None, optional
            If overlay is False, this is the label for the plot traces. This is relevant when plotting
            multiple systems' s-parameters on the same Axes.
        
        Returns
        -------
        ax : matplotlib.pyplot.Axes
            The Axes which the data is plotted on.
        '''
        
        if ax is None:
            nrows, ncols = (1,2) if overlay else (4,2)
            figsize = (60,20) if overlay else (55,60)
            fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        else:
            fig = ax[0,0].get_figure()
        plottitle = self.file if title is None else title
        
        # freq
        freq = self.fHz*fscale
        funits = '[Hz]' if fscale == 1 else '[MHz]' if fscale == 1e-6 else '[GHz]' if fscale == 1e-9 else ''
        
        # data for plot
        s11m = self.s11maglin if maglin else self.s11magdb
        s11p = self.s11phaserad if phaserad else self.s11phasedeg
        s12m = self.s12maglin if maglin else self.s12magdb
        s12p = self.s12phaserad if phaserad else self.s12phasedeg
        s21m = self.s21maglin if maglin else self.s21magdb
        s21p = self.s21phaserad if phaserad else self.s21phasedeg
        s22m = self.s22maglin if maglin else self.s22magdb
        s22p = self.s22phaserad if phaserad else self.s22phasedeg
        magunits = '[V]' if maglin else '[dB]'
        phaseunits = '[rad]' if phaserad else r'[$\degree$]'
        
        # unwrap
        if unwrap:
            period = 2*np.pi if phaserad else 360
            s11p = np.unwrap(s11p, period=period)
            s12p = np.unwrap(s12p, period=period)
            s21p = np.unwrap(s21p, period=period)
            s22p = np.unwrap(s22p, period=period)
        
        # plot
        if overlay:
            ax[0].plot(freq, s11m, label='S11', lw=3)
            ax[0].plot(freq, s12m, label='S12', lw=3)
            ax[0].plot(freq, s21m, label='S21', lw=3)
            ax[0].plot(freq, s22m, label='S22', lw=3)
            ax[1].plot(freq, s11p, label='S11', lw=3)
            ax[1].plot(freq, s12p, label='S12', lw=3)
            ax[1].plot(freq, s21p, label='S21', lw=3)
            ax[1].plot(freq, s22p, label='S22', lw=3)
            
            ax[0].legend(loc='best')
            ax[1].legend(loc='best')
            
            ax[0].set(title='Magnitude', ylabel=f'Mag {magunits}', xlabel=f'Freq {funits}')
            ax[1].set(title='Phase', ylabel=f'Phase {phaseunits}', xlabel=f'Freq {funits}')
            
        else:
            ax[0,0].plot(freq, s11m, lw=3, label=label)
            ax[0,1].plot(freq, s11p, lw=3)
            ax[1,0].plot(freq, s12m, lw=3, label=label)
            ax[1,1].plot(freq, s12p, lw=3)
            ax[2,0].plot(freq, s21m, lw=3, label=label)
            ax[2,1].plot(freq, s21p, lw=3)
            ax[3,0].plot(freq, s22m, lw=3, label=label)
            ax[3,1].plot(freq, s22p, lw=3)
            
            ax[0,0].set(title=f'Magnitude {magunits}', ylabel='S11')
            ax[0,1].set(title=f'Phase {phaseunits}')
            ax[1,0].set(ylabel='S12')
            ax[2,0].set(ylabel='S21')
            ax[3,0].set(ylabel='S22', xlabel=f'Freq {funits}')
            
            if label is not None:
                for subax in ax[:,0]:
                    subax.legend(loc='lower left')
        
        fig.suptitle(plottitle)
        fig.tight_layout()
        
        return ax