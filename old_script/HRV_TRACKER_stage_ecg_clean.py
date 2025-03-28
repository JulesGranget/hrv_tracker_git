
import copy
import numpy as np
import pandas as pd
import scipy.interpolate
import warnings
import scipy.signal


def compute_median_mad(data, axis=0):
    """
    Compute median and mad
    Parameters
    ----------
    data: np.array
        An array
    axis: int (default 0)
        The axis 
    Returns
    -------
    med: 
        The median
    mad: 
        The mad
    """
    med = np.median(data, axis=axis)
    mad = np.median(np.abs(data - med), axis=axis) / 0.6744897501960817
    return med, mad


def detect_peak(traces, srate, thresh=5, exclude_sweep_ms=4.0):
    """
    Simple positive peak detector.
    Parameters
    ----------
    traces: np.array
        An array
    srate: float
        Sampling rate of the traces
    thresh: float (default 5)
        The threhold as mad factor
        abs_threholds = med + thresh * mad
    exclude_sweep_ms: float
        Zone to exclude multiple peak detection when noisy.
        If several peaks or detected in the same sweep the best is the winner.
    Returns
    -------
    inds: np.array
        Indices on the peaks
    """
    
    exclude_sweep_size = int(exclude_sweep_ms / 1000. * srate)
    
    traces_center = traces[exclude_sweep_size:-exclude_sweep_size]
    length = traces_center.shape[0]
    
    med, mad = compute_median_mad(traces)
    abs_threholds = med + thresh * mad
    
    peak_mask = traces_center > abs_threholds
    for i in range(exclude_sweep_size):
        peak_mask &= traces_center > traces[i:i + length]
        peak_mask &= traces_center >= traces[exclude_sweep_size +
                                             i + 1:exclude_sweep_size + i + 1 + length]
    
    inds,  = np.nonzero(peak_mask)
    inds += exclude_sweep_size
    
    return inds

def get_empirical_mode(traces, nbins=200):
    """
    Get the emprical mode of a distribution.
    This is a really lazy implementation that make an histogram
    of the traces inside quantile [0.25, 0.75] and make an histogram
    of 200 bins and take the max.
    Parameters
    ----------
    traces: np.array
        The traces
    nbins: int (default 200)
        Number of bins for the histogram
    Returns
    -------
    mode: float
        The empirical mode.
    """
    
    q0 = np.quantile(traces, 0.25)
    q1 = np.quantile(traces, 0.75)


    mask = (traces > q0)  & (traces < q1)
    traces = traces[mask]
    count, bins = np.histogram(traces, bins=np.arange(q0, q1, (q1 - q0) / nbins))

    ind_max = np.argmax(count)

    mode = bins[ind_max]

    return mode
    



# convoultion suff to keep in mind
# sweep = np.arange(-60, 60)
# wfs = ecg_clean[some_peaks[:, None] + sweep]
# wfs.shape
# kernel = m / np.sum(np.sqrt(m**2))
# kernel -= np.min(kernel)

# kernel -= np.mean(kernel)

# fig, ax = plt.subplots()
# ax.plot(kernel)

# kernel_cmpl = kernel * + kernel * 1j

# ecg_conv = scipy.signal.fftconvolve(ecg_clean, kernel, mode='same')
# ecg_conv = np.convolve(ecg_clean, kernel, mode='same')
# ecg_conv = np.convolve(ecg_clean, kernel_cmpl, mode='same')
# print(ecg_conv.shape, ecg_conv.dtype)
# ecg_conv = np.abs(ecg_conv)



def preprocess(traces, srate, band=[5., 45.], btype='bandpass', ftype='bessel', order=5, normalize=True):
    """
    Apply simple filter using scipy
    
    For ECG bessel 5-50Hz and order 5 is maybe a good choice.
    By default also normalize the signal using median and mad.
    Parameters
    ----------
    traces: np.array
        Input signal.
    srate: float
        Sampling rate
    band: list of float or float
        Tha band in Hz or scalar if high/low pass.
    btype: 'bandpass', 'highpass', 'lowpass'
        The band pass type
    ftype: str (dfault 'bessel')
        The filter type used to generate coefficient using scipy.signal.iirfilter
    order: int (default 5)
        The order
    normalize: cool (default True)
        Aplly or not normalization
    Returns
    -------
    traces_clean: np.array
        The preprocess traces
    """
    
    if np.isscalar(band):
        Wn = band / srate * 2
    else:
        Wn = [e / srate * 2 for e in band]
    filter_coeff = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output='sos')
    
    traces_clean = scipy.signal.sosfiltfilt(filter_coeff, traces, axis=0)
    if normalize:
        med, mad = compute_median_mad(traces_clean)
        traces_clean -= med
        traces_clean /= mad

    return traces_clean 


def smooth_signal(trace, srate, win_shape='gaussian', sigma_ms=5.0):
    """
    A simple signal smoother using gaussian/rect kernel.
    Parameters
    ----------
    traces: np.array
        Input signal.
    srate: float
        Sampling rate
    win_shape: 'gaussian' / 'rect'
        The shape of the kernel
    sigma_ms: float (default 5ms)
        The length of the kernel. Sigma for the gaussian case.
    Returns
    -------
    trace_smooth: np.array
        The smoothed traces
    """

    size = int(srate * sigma_ms / 1000.)
    if win_shape == 'gaussian':
        times = np.arange(- 5 * size, 5 * size + 1)
        kernel = np.exp(- times ** 2 / size ** 2)
        kernel /= np.sum(kernel)

    elif win_shape == 'rect':
        kernel = np.ones(size, dtype='folat64') / size
    else:
        raise ValueError(f'Bad win_shape {win_shape}')

    trace_smooth = scipy.signal.fftconvolve(trace, kernel, mode='same', axes=0)

    return trace_smooth



def compute_ecg(raw_ecg, srate, parameter_preset='human_ecg', parameters=None):
    """
    Function for ECG that:
      * preprocess the ECG
      * detect R peaks
      * apply some cleaning to remove too small ECG interval
    Parameters
    ----------
    raw_ecg: np.array
        Raw traces of ECG signal
    srate: float
        Sampling rate
    parameter_preset: str or None
        Name of parameters set like 'human_ecg'
        This use the automatic parameters you can also have with get_ecg_parameters('human_ecg')
    parameters : dict or None
        When not None this overwrite the parameter set.
        
    Returns
    -------
    clean_ecg: np.array
        preprocess and normalized ecg traces
    ecg_peaks: pd.DataFrame
        dataframe with indices of R peaks, times of R peaks.
    """
    if parameter_preset is None:
        params = {}
    else:
        params = get_ecg_parameters(parameter_preset)
    if parameters is not None:
        recursive_update(params, parameters)

    
    clean_ecg = preprocess(raw_ecg, srate, **params['preprocess'])

    params_detection = params['peak_detection']
    if params_detection.get('thresh', None) == 'auto':
        # automatic threhold = half of the 99 pecentile (=max less artifcat)
        # empirical and naive but work more or less
        params_detection = copy.copy(params_detection)
        thresh = np.quantile(clean_ecg, 0.99) / 2.
        if thresh < 4:
            print(f'Automatic threshold for ECG is too low {thresh:0.2f} setting to 4')
            thresh = 4
        params_detection['thresh'] = thresh
    
    raw_ecg_peak = detect_peak(clean_ecg, srate, **params_detection)


    ecg_R_peaks = clean_ecg_peak(clean_ecg, srate, raw_ecg_peak, **params['peak_clean'])

    ecg_peaks = pd.DataFrame()
    ecg_peaks['peak_index'] = ecg_R_peaks
    ecg_peaks['peak_time'] = ecg_R_peaks / srate

    return clean_ecg, ecg_peaks



def clean_ecg_peak(ecg, srate, raw_peak_inds, min_interval_ms=400., max_clean_loop=4):
    """
    Clean peak with ultra simple idea: remove short interval.
    Parameters
    ----------
    ecg: np.array
        preprocess traces of ECG signal
    srate: float
        Sampling rate
    raw_peak_inds: np.array
        Array of peaks indices to be cleaned
    min_interval_ms: float (dfault 400ms)
        Minimum interval for cleaning
    Returns
    -------
    peak_inds: np.array
        Cleaned array of peaks indices 
    """
    peak_inds = raw_peak_inds.copy()

    for i in range(max_clean_loop):
        # when two peaks are too close :  remove the smaller peaks in amplitude
        peak_ms = (peak_inds / srate * 1000.)
        bad_peak, = np.nonzero(np.diff(peak_ms) < min_interval_ms)
        if bad_peak.size == 0:
            break
        # trick to keep the best amplitude when to peak are too close
        bad_ampl  = ecg[peak_inds[bad_peak]]
        bad_ampl_next  = ecg[peak_inds[bad_peak + 1]]
        bad_peak +=(bad_ampl > bad_ampl_next).astype(int)
        
        keep = np.ones(peak_inds.size, dtype='bool')
        keep[bad_peak] = False
        peak_inds = peak_inds[keep]
    
    return peak_inds




def compute_ecg_metrics(ecg_peaks, min_interval_ms=500., max_interval_ms=2000., verbose = False):
    """
    Compute metrics on ecg peaks: HRV_Mean, HRV_SD, HRV_Median, ...
    
    This metrics are a bit more robust that neurokit2 ones because strange interval
    are skiped from the analysis.
    Parameters
    ----------
    ecg_peaks: pr.DataFrame
        Datfarame containing ecg R peaks.
    min_interval_ms: float (default 500ms)
        Minimum interval inter R peak
    max_interval_ms: float (default 2000ms)
        Maximum interval inter R peak
    verbose: bool (default False)
        Control verbosity
    Returns
    -------
    metrics: pd.Series
        A table contaning metrics
    """
    
    peak_ms = ecg_peaks['peak_time'].values * 1000.
    
    remove = np.zeros(peak_ms.size, dtype='bool')
    d = np.diff(peak_ms) 
    bad, = np.nonzero((d > max_interval_ms)| (d < min_interval_ms))
    remove[bad] = True
    remove[bad+1] = True
    
    peak_ms[remove] = np.nan

    if verbose:
        print(f'{sum(np.isnan(peak_ms))} peaks removed')

    
    delta_ms = np.diff(peak_ms)
    
    # keep = delta_ms < max_interval_ms
    
    # delta_ms = delta_ms[keep]
    
    
    metrics = pd.Series(dtype=float)
    
    metrics['HRV_Mean'] = np.nanmean(delta_ms)
    metrics['HRV_SD'] = np.nanstd(delta_ms)
    metrics['HRV_Median'], metrics['HRV_Mad'] = compute_median_mad(delta_ms[~np.isnan(delta_ms)])
    metrics['HRV_CV'] = metrics['HRV_SD'] / metrics['HRV_Mean']
    metrics['HRV_MCV'] = metrics['HRV_Mad'] / metrics['HRV_Median']
    metrics['HRV_Asymmetry'] = metrics['HRV_Median'] - metrics['HRV_Mean']

    
    # TODO
    metrics['HRV_RMSSD'] = np.sqrt(np.nanmean(np.diff(delta_ms)**2))

    # return pd.DataFrame(metrics).T
    return metrics



def compute_instantaneous_rate(ecg_peaks, new_times, limits=None, units='bpm', interpolation_kind='linear'):
    """
    
    Parameters
    ----------
    ecg_peaks: pr.DataFrame
        Datfarame containing ecg R peaks.
    new_times : np.array
        Time vector for interpolating the instanteneous rate.
    limits : list or None
        Limits for removing outliers.
    units : 'bpm' / 'Hz' / 'ms' / 's'
        Units of the rate. can be interval or rate.
    interpolation_kind : 'linear'/ 'cubic'
    """
    peak_times = ecg_peaks['peak_time'].values

    delta = np.diff(peak_times)

    if units == 's':
        delta = delta
    elif units == 'ms':
        delta = delta * 1000.
    elif units == 'Hz':
        delta = 1.  / delta
    elif units == 'bpm':
        delta = 60.  / delta
    else:
        raise ValueError(f'Bad units {units}')

    if limits is not None:
        lim0, lim1 = limits
        keep,  = np.nonzero((delta > lim0) & (delta < lim1))
        peak_times = peak_times[keep]
        delta = delta[keep]
    else:
        peak_times = peak_times[:-1]

    interp = scipy.interpolate.interp1d(peak_times, delta, kind=interpolation_kind, axis=0,
                                        bounds_error=False, fill_value='extrapolate')
    
    instantaneous_rate = interp(new_times)

    return instantaneous_rate
    

def compute_hrv_psd(ecg_peaks, sample_rate=100., limits=None, units='bpm',
                                        freqency_bands = {'lf': (0.04, .15), 'hf' : (0.15, .4)},
                                        window_s=250., interpolation_kind='cubic'):
    """
    Compute hrv power spectrum density and extract some metrics:
      * lf power
      * hf power
    
    Please note:
        1. The duration of the signal and the window are important parameters to estimate low frequencies
           in a spectrum. Some warnings or errors should popup if they are too short.
        2. Given that the hrv is mainly driven by the respiration the frequency boudaries are often innacurate!
           For instance a slow respiration at 0.1Hz is moving out from the 'hf' band wheras this band should capture
           the respiratory part of the hrv.
        3. The instataneous rate is computed by interpolating eccg peak interval, the interpolation method
           'linear' or 'cubic' are a real impact of the dynamic and signal smoothness and so the spectrum should differ
           because of the wieight of the harmonics
        4. The units of the instantaneous hrv (bpm, interval in second, interval in ms) have a high impact on the
           magnitude of metrics. Many toolboxes (neurokit2, ) differ a lot on this important detail.
        5. Here we choose the classical welch method for spectrum density estimation. Some parameters have also small
           impact on the results : dentend, windowing, overlap.
    
    
    Parameters
    ----------
    ecg_peaks: pr.DataFrame
        Datfarame containing ecg R peaks.
    
    sample_rate=100.
    
    limits=None
    
    units='bpm'
    interpolation_kind
    
    """
    
    ecg_duration_s = ecg_peaks['peak_time'].values[-1]
    
    
    times = np.arange(0, ecg_duration_s, 1 / sample_rate)
    
    instantaneous_rate = compute_instantaneous_rate(ecg_peaks, times, limits=limits, units=units,
                                                    interpolation_kind=interpolation_kind)
    
    # some check on the window
    min_freq = min(freqs[0] for freqs in freqency_bands.values())
    if window_s <  (1 / min_freq) * 5:
        raise ValueError(f'The window is too short {window_s}s compared to the lowest frequency {min_freq}Hz')
    if ecg_duration_s <  (1 / min_freq) * 5:
        raise ValueError(f'The duration is too short {ecg_duration_s}s compared to the lowest frequency {min_freq}Hz')

    if window_s <  (1 / min_freq) * 10 or (1 / min_freq) * 10:
        warnings.warn(f'The window is not optimal {window_s}s compared to the lowest frequency {min_freq}Hz')
    if ecg_duration_s <  (1 / min_freq) * 10 or (1 / min_freq) * 10:
        warnings.warn(f'The duration is not optimal {ecg_duration_s}s compared to the lowest frequency {min_freq}Hz')

    
    # See https://github.com/scipy/scipy/issues/8368 about density vs spectrum

    # important note : when using welch with scaling='density'
    # then the integrale (trapz) must be aware of the dx to take in account
    # so the metrics scale is invariant given against sampling rate and also sample_rate
    nperseg = int(window_s * sample_rate)
    nfft = nperseg
    psd_freqs, psd = scipy.signal.welch(instantaneous_rate, detrend='constant', fs=sample_rate, window='hann',
                                                            scaling='density', nperseg=nperseg, noverlap=0, nfft=nfft)

    metrics = pd.Series(dtype=float)
    delta_freq = np.mean(np.diff(psd_freqs))
    for name, freq_band in freqency_bands.items():
        f0, f1 = freq_band
        area = np.trapz(psd[(psd_freqs >= f0) & (psd_freqs < f1)], dx=delta_freq)
        metrics[name] = area

    return psd_freqs, psd, metrics


def recursive_update(d, u):
    """
    Recurssive dict update.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



def get_respiration_parameters(parameter_preset):
    """
    Get parameters nested dict for a given predefined parameter preset.
    """
    return copy.deepcopy(_resp_parameters[parameter_preset])


def get_ecg_parameters(parameter_preset):
    """
    Get parameters nested dict for a given predefined parameter preset.
    """
    return copy.deepcopy(_ecg_parameters[parameter_preset])


###################################################
# Resp preset

_resp_parameters = {}

# this parameters works with airflow sensor for a human
_resp_parameters['human_airflow'] = dict(
    preprocess=dict(band=7., btype='lowpass', ftype='bessel', order=5, normalize=False),
    smooth=dict(win_shape='gaussian', sigma_ms=60.0),
    cycle_detection=dict(inspiration_adjust_on_derivative=False),
    baseline=dict(baseline_mode='median'),
    cycle_clean=dict(low_limit_log_ratio=4.),
)


_resp_parameters['rat_plethysmo'] = dict(
    preprocess=dict(band=50., btype='lowpass', ftype='bessel', order=5, normalize=False),
    # smooth=dict(win_shape='gaussian', sigma_ms=5.0),
    smooth=None,
    cycle_detection=dict(inspiration_adjust_on_derivative=False),
    baseline=dict(baseline_mode='manual', baseline=0.),
    cycle_clean=dict(low_limit_log_ratio=5.),
)


###################################################
#ECG preset

_ecg_parameters = {}

# this parameters works well with simple ecg signal with positive peaks
_ecg_parameters['human_ecg'] = dict(
    preprocess=dict(band=[5., 45.], ftype='bessel', order=5, normalize=True),
    peak_detection=dict(thresh='auto', exclude_sweep_ms=4.0),
    peak_clean=dict(min_interval_ms=400.),
)

_ecg_parameters['rat_ecg'] = dict(
    preprocess=dict(band=[5., 200.], ftype='bessel', order=5, normalize=True),
    peak_detection=dict(thresh='auto', exclude_sweep_ms=4.0),
    peak_clean=dict(min_interval_ms=50.),
)





