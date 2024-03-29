"""
Toolkit for plotting data handled with rftoolkit modules.

@author: zackashm
"""

import numpy as np

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
sns.set_theme(font_scale=5, rc={'font.family':'Helvetica', 'axes.facecolor':'#e5ebf6',
                                'legend.facecolor':'#dce3f4', 'legend.edgecolor':'#000000',
                                'axes.prop_cycle':cycler('color', 
                                                         ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', 
                                                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
                                                          '#FF97FF', '#FECB52']),
                                'axes.xmargin':0, 'axes.ymargin':0.05})
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from itertools import cycle
pio.renderers.default='png'     # for displaying on noninteractive programs

# helpful, but not necessary imports
from warnings import warn

# UPDATE THIS
def plotBeampattern(beamDataAngles, beamDataGains, freqs, funit='Hz', angleOffset=0, 
                    polar=True, plot_object=None, dBlim=(None,None), label_add='',
                    title='', close_gap=False, colors=None, scatterpolar_kwargs={}, **kwargs):
    """
    Plot a beampattern, the gain response [dB] vs angle [degree] at select frequencies.
    
    Parameters
    ----------
    beamDataAngles : ndarray, list[ndarray]
        An array or list of arrays corresponding to the angles in degrees. If only one array, but beamDataGains
        contains multiple arrays, this array is assumed the same for all gain arrays. If multiple arrays, the
        index should correspond to the selected frequency array.
    beamDataGains : ndarray, list[ndarray]
        An array or list of arrays corresponding to the gain in dB. If multiple arrays, the
        index should correspond to the selected frequency array.
    freqs : float, list[floats]
        A float or list of frequency data which corresponds via index to the list of beamData arrays.
    funit : str, optional
        The unit of the frequencies in freqs. Default is "Hz".
    angleOffset : float
        Give an offset in degrees to shift the angles of the pattern.
    polar : bool, optional
        If True, the plot will be in polar coordinates and use plotly.graph_objects.Figure.
        Otherwise, the plot will use rectangular coordinates and use matplotlib.Axes.
        If plot_object is given, this should match the corresponding object with polar.
    plot_object : matplotlib.Axes or plotly.graph_objects.Figure, optional
        The plot object to use. This should match the corresponding object
        that polar would use: True => plotly.graph_objects.Figure, False => matplotlib.Axes.
        If not given, the plot object is created internally.
    dBlim : tuple(2), optional
        The (min,max) dB limits. The default is the full range.
    label_add : str, optional
        This is added to the frequency labels in the legend if needed.
    title_add : str, optional
        This is added to the plot title if needed.
    close_gap : bool, optional
        Connect the first and last points. Default is False. Usually relevant to polar plotting.
    colors : list
        List of colors to use for plotting.
    scatterpolar_kwargs : Dict
        When polar is True, this is a dict containing kwargs for plotly.Figure.Scatterpolar().
    **kwargs
        If polar is True, this is plotly.Figure.update_layout() kwargs. 
        Otherwise, this is matplotlib.Axes.plot() kwargs.
        
    Returns
    -------
    fig : matplotlib.Figure or plotly.graph_objects.Figure
        The plotting object.
    ax : matplotlib.Axes
        If not plotting polar, the matplotlib Axes.
    """
    
    # ensure data shapes for iterations
    if len(np.array(beamDataGains).shape) == 1:
        beamDataGains = np.array([beamDataGains])       # force 2d array
    else:
        beamDataGains = np.array(beamDataGains)
        
    if len(np.array(beamDataAngles).shape) == 1:
        beamDataAngles = np.array([beamDataAngles])     # force 2d array
    else:
        beamDataAngles = np.array(beamDataAngles)
        
    if len(np.array(freqs).shape) == 0:
        freqs = np.array([freqs])                       # force 1d array
    else:
        freqs = np.array(freqs)
    
    # handle data mismatch
    if freqs.shape[0] != beamDataGains.shape[0]:
        raise ValueError(f'List of arrays must match length of frequency array. {beamDataGains.shape[0]} vs {freqs.shape[0]}')
    
    # zip data
    beamData = {}
    for freqind, freq in enumerate(freqs):
        angleind = 0 if beamDataAngles.shape[0] == 1 else freqind    # common angles or not
        beamData[freq] = [beamDataAngles[angleind], beamDataGains[freqind]]
    
    # set the plot object
    if polar:  # plotly much nicer for polar plots
    
        # polar should use go.Figure()
        if isinstance(plot_object, plt.Axes):
            warn("WARNING [antenna_response.plot_beampattern()]: Polar is set to True, but " +
                  "the plot object is a matplotlib Axes instead of plotly graph object. " +
                  "Defaulting to plot_object to None.")
            plot_object = None
            
        fig = plot_object if plot_object is not None else go.Figure()
        
    else:      # standard rectangular coords
    
        # not polar should use plt.Axes
        if isinstance(plot_object, go.Figure):
            warn("WARNING [antenna_response.plot_beampattern()]: Polar is set to False, but " +
                  "the plot object is a plotly graph object instead of matplotlib Axes. " +
                  "Defaulting to plot_object to None.")
            plot_object = None
        
        fig, ax = plot_object if plot_object is not None else plt.subplots(figsize=[30,20])
        ax.set(xlabel=r"theta [$\degree$]", ylabel="Gain [dB]", title=title,ylim=dBlim)
    
    # set colors and polar args for plotting
    COLORS = cycle(px.colors.sequential.OrRd[1:] + ['#530000', '#210000']) if colors is None else cycle(colors)
    if polar:
        spmode = scatterpolar_kwargs.pop('mode', 'lines')
        spline = scatterpolar_kwargs.pop('line', dict(width=3))
        spmarker = scatterpolar_kwargs.pop('marker', dict(symbol='0',size=8,opacity=1))
    else:
        pltls = kwargs.pop('ls', '-')
        pltmarker = kwargs.pop('marker', 'o')
        pltlw = kwargs.pop('lw', 4)
        pltms = kwargs.pop('ms', 13)
    
    # now plot the gain response vs angle at each frequency of interest
    for freq in freqs:
        
        # set data
        angles = beamData[freq][0]
        gains = beamData[freq][1]
        
        if close_gap:
            angles = np.append(angles, angles[0])
            gains = np.append(gains, gains[0])
        
        # and plot
        if polar:
            
            fig.add_trace(
                go.Scatterpolar(
                    r = gains,
                    theta = angles + angleOffset,
                    name = '{:.1f} {} {}'.format(freq,funit,label_add),
                    line_color = next(COLORS),
                    mode = spmode,
                    line = spline,
                    marker = spmarker,
                    **scatterpolar_kwargs,
                    ),
                )
        else:
            ax.plot(angles + angleOffset, gains, ls=pltls, marker=pltmarker, lw=pltlw, ms=pltms, 
                    label='{:.1f} {} {}'.format(freq,funit,label_add), **kwargs)
    
    # plot settings
    if polar:
        
        # add solid lines for main grid / major ticks
        rmin,rmax = (beamDataGains.min(), beamDataGains.max()) if dBlim == (None,None) else dBlim 
        for deg in np.arange(-180,180,30):
            fig.add_trace(
                go.Scatterpolar(
                    r = (rmin,rmax),
                    theta = (deg,deg),
                    mode = 'lines',
                    name = None,
                    line=dict(color='#cccccc',width=1),
                    showlegend=False,
                    )
                )
        circ = np.linspace(-180,180,500)
        for dB in range(int(rmin), int(rmax)):
            if dB % 5 == 0:
                fig.add_trace(
                    go.Scatterpolar(
                        r = dB*np.ones(circ.size),
                        theta = circ,
                        mode = 'lines',
                        name = None,
                        line=dict(color='#cccccc',width=1),
                        showlegend=False,
                        )
                    )
        
        _makePlotlyPolarLookNice(fig, rlim=dBlim, rlabel='', alltitle=title, **kwargs)
        fig.show()
    else:
        legend = ax.legend(loc='upper right')
        for line in legend.get_lines(): 
            line.set_linewidth(15)
        
    # return plot object
    if polar:
        return fig
    else:
        return fig, ax



def _makePlotlyPolarLookNice(fig, rlim=(None,None), rlabel='', sector=(0,180), alltitle='', 
                             kwargs_for_trace={}, **kwargs):
    '''
    Makes plotly's polar plot look nice, to the extent it can on Python...

    Parameters
    ----------
    fig : plotly.Figure
        The figure to make look nice.
    rlim : tuple(2)
        Radial axis (min,max) limits.
    rlabel : str
        The radial axis label. The default is ''.
    sector : tuple(2)
        Set the angular sector of the chart. Default is (0,180), i.e. upper half.
    alltitle : str
        The overall title. The default is ''.
    kwargs_for_trace : dict
        Kwargs specifically for plotly.Figure.update_traces(). I planned to use this
        if I wanted multiple data set ups plotted and wanted to distinguish them by
        changing the traces to a certain style.
    **kwargs
        plotly.Figure.update_layout() kwargs. 

    Returns
    -------
    None.

    '''
    
    # kwarg defaults
    polar = kwargs.pop('polar', 
                       dict(                                # polar setting
                           sector = sector,                 # set chart shape (half)
                           bgcolor='#fcfcfc',
                           angularaxis = dict(              # angle axis settings
                               dtick = 5,                   # angle tick increment
                               ticklabelstep=6,             # tick label increment
                               gridcolor='#d6d6d6',         # tick line color
                               griddash='dot',              # tick line style
                               # rotation = 90,               # rotates data to be on upper half
                               direction = "counterclockwise",     # -90 to 90 right to left
                               tickfont=dict(size=20),      # angle tick size
                               ),
                           radialaxis=dict(                 # radial axis settings
                               angle=0,                     # angle where the tick axis is
                               tickangle=0,                 # rotation of ticklabels
                               dtick=1,                     # radial tick increment
                               ticklabelstep=10,             # tick label increment
                               gridcolor='#d6d6d6',         # tick line color
                               griddash='dot',              # tick line style
                               linecolor='#d6d6d6',         # axis color
                               exponentformat="E",          # tick label in exponential form
                               title=dict(                  # radial axis label settings
                                   text=rlabel,             # the label text
                                   font=dict(               # label font settings
                                       size = 23,           # label text size
                                       ),
                                   ),
                               tickfont=dict(size=20),      # radial tick size
                               range=rlim,                  # radial limits
                                )
                           ),
                       )
    
    autosize = kwargs.pop('autosize', False)                # we'll custom size the figure
    
    width = kwargs.pop('width', 1100)                       # canvas width
    
    height = kwargs.pop('height', 820)                      # canvas height
    
    legend = kwargs.pop('legend',                           # legend settings
                        dict(                        
                             bgcolor="#d8e4f5",             # legend bg color
                             xanchor="right",               # reference for x pos
                             x=1.05,                        # legend x pos percentage
                             yanchor="top",                 # reference for x pos
                             y=1.05,                        # legend y pos percentage
                             font=dict(size=20),            # font size
                             itemsizing='constant',         # symbol size to not depend on data traces
                             itemwidth=30,                  # symbol size
                             )
                        )
    
    margin = kwargs.pop('margin',                           # margins
                        dict(                               
                            b=30,
                            t=70,
                            l=40,
                            r=60,
                            ),
                        )
    
    title = kwargs.pop('title',                             # title settings
                       dict(                                
                           text=alltitle,                   # title text
                           x = 0.03,                        # title position
                           y = 0.98,
                           font=dict(
                               size = 24,                   # title font size
                               ),
                           ),
                       )
    
    # update plotly figure
    fig.update_layout(
            polar=polar,
            autosize=autosize,
            width=width,
            height=height,
            legend=legend,
            margin=margin,
            title=title,
            **kwargs,       # other user specific kwargs
            )
    
    # for the traces themselves
    fig.update_traces(**kwargs_for_trace)
