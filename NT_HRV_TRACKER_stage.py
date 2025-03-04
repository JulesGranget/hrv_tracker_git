

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from NT_HRV_TRACKER_stage_ecg_clean import *
from NT_HRV_TRACKER_stage_metrics import *

debug = False





################################
######## FUNCTIONS ######## 
################################


from itertools import repeat


###################################################################################################
###################################################################################################

def check_param_range(param, label, bounds):
    """Check a parameter value is within an acceptable range.

    Parameters
    ----------
    param : float
        Parameter value to check.
    label : str
        Label of the parameter being checked.
    bounds : list of [float, float]
       Bounding range of valid values for the given parameter.

    Raises
    ------
    ValueError
        If a parameter that is being checked is out of range.
    """

    if (param < bounds[0]) or (param > bounds[1]):
        msg = "The provided value for the {} parameter is out of bounds. ".format(label) + \
        "It should be between {:1.1f} and {:1.1f}.".format(*bounds)
        raise ValueError(msg)



from operator import gt, lt

###################################################################################################
###################################################################################################

def find_zerox(sig, peaks, troughs):
    """Find zero-crossings within each cycle, from identified peaks and troughs.

    Parameters
    ----------
    sig : 1d array
        Time series.
    peaks : 1d array
        Samples of oscillatory peaks.
    troughs : 1d array
        Samples of oscillatory troughs.

    Returns
    -------
    rises : 1d array
        Samples at which oscillatory rising zero-crossings occur.
    decays : 1d array
        Samples at which oscillatory decaying zero-crossings occur.

    Notes
    -----
    - Zero-crossings are defined as when the voltage crosses midway between one extrema and
      the next. For example, a 'rise' is halfway from the trough to the peak.
    - If this halfway voltage is crossed at multiple times, the temporal median is taken
      as the zero-crossing.
    - Sometimes, due to noise in estimating peaks and troughs when the oscillation
      is absent, the estimated peak might be lower than an adjacent trough. If this
      occurs, the rise and decay zero-crossings will be set to be halfway between
      the peak and trough.
    - Burst detection should be used to restrict phase estimation to periods with oscillations
      present, in order to ignore periods of the signal in which estimation is poor.

    Examples
    --------
    Find the rise and decay zero-crossings locations of a simulated signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from bycycle.cyclepoints import find_extrema
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> peaks, troughs = find_extrema(sig, fs, f_range=(8, 12))
    >>> rises, decays = find_zerox(sig, peaks, troughs)
    """

    # Calculate the number of rises and decays
    n_rises = len(peaks)
    n_decays = len(troughs)
    idx_bias = 0

    # Offset values, depending on order of peaks & troughs
    if peaks[0] < troughs[0]:
        n_rises -= 1
    else:
        n_decays -= 1
        idx_bias += 1

    rises = _find_flank_midpoints(sig, 'rise', n_rises, troughs, peaks, idx_bias)
    decays = _find_flank_midpoints(sig, 'decay', n_decays, peaks, troughs, idx_bias)

    return rises, decays


def find_flank_zerox(sig, flank, midpoint=None):
    """Find zero-crossings on rising or decaying flanks of a filtered signal.

    Parameters
    ----------
    sig : 1d array
        Time series to detect zero-crossings in.
    flank : {'rise', 'decay'}
        Which flank, rise or decay, to use to get zero crossings.

    Returns
    -------
    zero_xs : 1d array
        Samples of the zero crossings.

    Examples
    --------
    Find rising flanks in a filtered signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from neurodsp.filt import filter_signal
    >>> sig = sim_bursty_oscillation(10, 500, freq=10)
    >>> sig_filt = filter_signal(sig, 500, 'lowpass', 30)
    >>> rises_flank = find_flank_zerox(sig_filt, 'rise')
    """

    if midpoint is None:
        midpoint = 0

    assert flank in ['rise', 'decay']
    pos = sig <= midpoint if flank == 'rise' else sig > midpoint

    zero_xs = (pos[:-1] & ~pos[1:]).nonzero()[0]

    # If no zero-crossing's found (peak and trough are same voltage), output dummy value
    zero_xs = [int(len(sig) / 2)] if len(zero_xs) == 0 else zero_xs

    return zero_xs


def _find_flank_midpoints(sig, flank, n_flanks, extrema_start, extrema_end, idx_bias):
    """Helper function for find_zerox."""

    assert flank in ['rise', 'decay']
    idx_bias = -idx_bias + 1 if flank == 'rise' else idx_bias
    comp = gt if flank == 'rise' else lt

    flanks = np.zeros(n_flanks, dtype=int)
    for idx in range(n_flanks):

        sig_temp = sig[extrema_start[idx]:extrema_end[idx + idx_bias] + 1]
        midpoint = (sig_temp[0] + sig_temp[-1]) / 2.

        # If data is all zeros, just set the zero-crossing to be halfway between
        if np.sum(np.abs(sig_temp)) == 0:
            flanks[idx] = extrema_start[idx] + int(len(sig_temp) / 2.)

        # If flank is actually an extrema, just set the zero-crossing to be halfway between
        elif comp(sig_temp[0], sig_temp[-1]):
            flanks[idx] = extrema_start[idx] + int(len(sig_temp) / 2.)
        else:
            midpoint = (sig_temp[0] + sig_temp[-1]) / 2.
            flanks[idx] = extrema_start[idx] + \
                int(np.median(find_flank_zerox(sig_temp, flank, midpoint)))

    return flanks



from neurodsp.filt import filter_signal
from neurodsp.filt.fir import compute_filter_length


###################################################################################################
###################################################################################################

def find_extrema(sig, fs, f_range, boundary=0, first_extrema='peak',
                 filter_kwargs=None, pass_type='bandpass', pad=True):
    """Identify peaks and troughs in a time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range, in Hz, to narrowband filter the signal, used to find zero-crossings.
    boundary : int, optional, default: 0
        Number of samples from edge of the signal to ignore.
    first_extrema: {'peak', 'trough', None}
        If 'peak', then force the output to begin with a peak and end in a trough.
        If 'trough', then force the output to begin with a trough and end in peak.
        If None, force nothing.
    filter_kwargs : dict, optional, default: None
        Keyword arguments to :func:`~neurodsp.filt.filter.filter_signal`,
        such as 'n_cycles' or 'n_seconds' to control filter length.
    pass_type : str, optional, default: 'bandpass'
        Which kind of filter pass_type is consistent with the frequency definition provided.
    pad : bool, optional, default: True
        Whether to pad ``sig`` with zeros to prevent missed cyclepoints at the edges.

    Returns
    -------
    peaks : 1d array
        Indices at which oscillatory peaks occur in the input ``sig``.
    troughs : 1d array
        Indices at which oscillatory troughs occur in the input ``sig``.

    Notes
    -----
    This function assures that there are the same number of peaks and troughs
    if the first extrema is forced to be either peak or trough.

    Examples
    --------
    Find the locations of peaks and burst in a signal:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> peaks, troughs = find_extrema(sig, fs, f_range=(8, 12))
    """

    # Ensure arguments are within valid range
    check_param_range(fs, 'fs', (0, np.inf))

    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    # Get the original signal and filter lengths
    sig_len = len(sig)
    filt_len = 0

    # Pad beginning of signal with zeros to prevent missing cyclepoints
    if pad:

        filt_len = compute_filter_length(fs, pass_type, f_range[0], f_range[1],
                                         n_seconds=filter_kwargs.get('n_seconds', None),
                                         n_cycles=filter_kwargs.get('n_cycles', 3))

        # Pad the signal
        sig = np.pad(sig, int(np.ceil(filt_len/2)), mode='constant')

    # Narrowband filter signal
    sig_filt = filter_signal(sig, fs, pass_type, f_range, remove_edges=False, **filter_kwargs)

    # Find rising and decaying zero-crossings (narrowband)
    rise_xs = find_flank_zerox(sig_filt, 'rise')
    decay_xs = find_flank_zerox(sig_filt, 'decay')

    # Compute number of peaks and troughs
    if rise_xs[-1] > decay_xs[-1]:
        n_peaks = len(rise_xs) - 1
        n_troughs = len(decay_xs)
    else:
        n_peaks = len(rise_xs)
        n_troughs = len(decay_xs) - 1

    # Calculate peak samples
    peaks = np.zeros(n_peaks, dtype=int)
    _decay_xs = decay_xs.copy()
    for p_idx in range(n_peaks):

        # Calculate the sample range between the most recent zero rise and the next zero decay
        last_rise = rise_xs[p_idx]

        for idx, decay in enumerate(_decay_xs):
            if decay > last_rise:
                _decay_xs = _decay_xs[idx:]
                break

        next_decay = _decay_xs[0]

        # Identify time of peak
        peaks[p_idx] = np.argmax(sig[last_rise:next_decay]) + last_rise

    # Calculate trough samples
    troughs = np.zeros(n_troughs, dtype=int)
    _rise_xs = rise_xs.copy()
    for t_idx in range(n_troughs):

        # Calculate the sample range between the most recent zero decay and the next zero rise
        last_decay = decay_xs[t_idx]

        for idx, rise in enumerate(_rise_xs):
            if rise > last_decay:
                _rise_xs = _rise_xs[idx:]
                break

        next_rise = _rise_xs[0]

        # Identify time of trough
        troughs[t_idx] = np.argmin(sig[last_decay:next_rise]) + last_decay

    # Remove padding
    peaks = peaks - int(np.ceil(filt_len/2))
    troughs = troughs - int(np.ceil(filt_len/2))

    # Remove peaks and trough outside the boundary limit
    peaks = peaks[np.logical_and(peaks > boundary, peaks < sig_len - boundary)]
    troughs = troughs[np.logical_and(troughs > boundary, troughs < sig_len - boundary)]

    # Force the first extrema to be as desired & assure equal # of peaks and troughs
    if first_extrema == 'peak':
        troughs = troughs[1:] if peaks[0] > troughs[0] else troughs
        peaks = peaks[:-1] if peaks[-1] > troughs[-1] else peaks
    elif first_extrema == 'trough':
        peaks = peaks[1:] if troughs[0] > peaks[0] else peaks
        troughs = troughs[:-1] if troughs[-1] > peaks[-1] else troughs
    elif first_extrema is None:
        pass
    else:
        raise ValueError('Parameter "first_extrema" is invalid')

    return peaks, troughs




################################
######## FUNCTIONS ######## 
################################


def get_peaks_from_ppg(sig, srate, btype='bandpass', ftype='bessel', order=5, debug=False):

    """"
    sig : np.array
    """

    #### zscore
    sig = (sig - sig.mean()) / sig.std()

    #### identify heart rate for filtering
    nwind = int(5*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx = scipy.signal.welch(sig, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    if debug:
        plt.plot(hzPxx, Pxx)
        plt.show()

    band = [hzPxx[Pxx.argmax()] - 0.5, hzPxx[Pxx.argmax()] + 0.5] 

    #### filter signal
    if np.isscalar(band):
        Wn = band / srate * 2
    else:
        Wn = [e / srate * 2 for e in band]

    filter_coeff = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output='sos')

    sig_clean = scipy.signal.sosfiltfilt(filter_coeff, sig, axis=0)

    #### get peaks
    peaks, troughs = find_extrema(sig_clean, srate, band)

    if debug:
        plt.plot(sig, label='raw')
        plt.plot(sig_clean, label='clean')
        plt.vlines(peaks, ymin=sig_clean.min(), ymax=sig_clean.max(), colors='r')
        plt.legend()
        plt.show()

    return peaks



#sujet_i = 1
#sig_physio, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict = sig_physio_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict
def hrv_tracker_svm(sig_physio, classifier, prms_tracker, sig_mode):

    srate = prms_tracker['srate']

    win_size_sec, metric_used, odor_trig_n_bpm = prms_tracker['win_size_sec'], prms_tracker['metric_list'], prms_tracker['odor_trig_n_bpm']
    win_size = int(win_size_sec*srate)

    #### load cR
    if sig_mode == 'ecg':
        sig_physio, ecg_peaks = compute_ecg(sig_physio, srate) # recuperation des pics
        cR_val = ecg_peaks/srate
    if sig_mode == 'ppg':
        ppg_peaks = get_peaks_from_ppg(sig_physio, srate) # recuperation des pics
        cR_val = ppg_peaks/srate

    RRI = np.diff(cR_val) # recuperation des intervalles R-R

    #### verif
    if debug:
        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((cR_val[0], cR_val[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, cR_val[cR_initial])
        cR_initial += 1

    #### first point sliding win
    RRI_win = np.diff(ecg_cR_sliding_win)
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
    predictions = classifier.predict(df_res.values)
    trig_odor = [0]
    times = [cR_val[cR_initial]]

    #### progress bar
    # bar = IncrementalBar('Countdown', max = len(cR_val)-cR_initial)

    #### sliding on other cR
    for cR_i in range(len(cR_val)-cR_initial):

        # bar.next()

        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], cR_val[cR_i])
        RRI_win = np.diff(ecg_cR_sliding_win)
        df_slide = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
        predictions = np.append(predictions, classifier.predict(df_slide.values))
        df_res = pd.concat([df_res, df_slide], axis=0)

        if predictions.shape[0] >= odor_trig_n_bpm:
            trig_odor_win = predictions.copy()[-odor_trig_n_bpm:]
            trig_odor_win[trig_odor_win < 2] = 0
            trig_odor_win[(trig_odor_win == prms_tracker['cond_label_tracker']['MECA']) | (trig_odor_win == prms_tracker['cond_label_tracker']['CO2'])] = 1
            trig_odor_pred_i = np.round(np.mean(trig_odor_win))

            if trig_odor_pred_i != 0:
                trig_odor.append(1)
            else:
                trig_odor.append(0)
        
        else:
            trig_odor.append(0)

        times.append(cR_val[cR_i])

    # bar.finish()

    times = np.array(times)
    trig_odor = np.array(trig_odor)

    return df_res, times, predictions, trig_odor






def get_data_hrv_tracker(sig_physio, prms_tracker, sig_mode):

    #### extract params
    srate = prms_tracker['srate']
    win_size_sec, jitter = prms_tracker['win_size_sec'], prms_tracker['jitter']
    win_size = int(win_size_sec*srate)

    #### precompute sig_physio
    if sig_mode == 'ecg':
        sig_physio, ecg_peaks = compute_ecg(sig_physio, srate)
        cR_val = ecg_peaks/srate
    if sig_mode == 'ppg':
        ppg_peaks = get_peaks_from_ppg(sig_physio, srate)
        cR_val = ppg_peaks/srate # valeur des peaks en secondes
    
    RRI = np.diff(cR_val)

    #### verif
    if debug:
        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((cR_val[0], cR_val[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, cR_val[cR_initial])
        cR_initial += 1

    #### first point sliding win
    RRI_win = np.diff(ecg_cR_sliding_win)
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
    times = [cR_val[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(cR_val)-cR_initial):
        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], cR_val[cR_i])
        RRI_win = np.diff(ecg_cR_sliding_win)
        df_slide = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
        
        df_res = pd.concat([df_res, df_slide], axis=0)

        times.append(cR_val[cR_i])

    times = np.array(times)

    return df_res, times











################################
######## COMPUTE ######## 
################################


def hrv_tracker_exploration(sig_mode):
    
    #### params
    prms_tracker = {
    'metric_list' : ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_COV'],
    'win_size_sec' : 30,
    'odor_trig_n_bpm' : 75,
    'jitter' : 0,
    'srate' : 500,
    'cond_label_tracker' : {'FR_CV_1' : 1, 'MECA' : 2, 'CO2' : 3, 'FR_CV_2' : 1},
    }

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    conditions = ['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']

    train_value = 0.8

    ######## LOAD DATA ########
    # os.chdir('/home/jules/Bureau/')
    
    # à changer par mes fichiers que je vais load

    if sig_mode == 'ecg':
        sig_physio = np.load('ecg_datas.npy')
        label_vec = np.load('label_hrv_tracker.npy')
    if sig_mode == 'ppg':
        sig_physio = np.load('ppg_hrv_tracker.npy')
        label_vec = np.load('label_hrv_tracker.npy') # il faut créer le label vec

    if debug:

        sig_physio = np.load('ecg_hrv_tracker.npy')
        label_vec = np.load('label_hrv_tracker.npy') # il faut créer le label vec

    with open('trig_hrv_tracker.pkl', 'rb') as f:
        trig = pickle.load(f)

    if debug:

        plt.plot(sig_physio)
        plt.show()

        plt.plot(label_vec)
        plt.show()

    #### get metrics
    df_hrv, times = get_data_hrv_tracker(sig_physio, prms_tracker, sig_mode)
    label_vec = label_vec[(times*prms_tracker['srate']).astype('int')]

    #### split values
    X, y = df_hrv.values, label_vec.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_value, random_state=5)
    
    #### make pipeline
    #SVC().get_params()
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps)

    #### find best model
    params = {
    # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
    # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
    'SVM__C' : [0.001, 0.1, 1, 10, 100, 10e5], 
    'SVM__gamma' : [0.1, 0.01]
    }

    print('train', flush=True)

    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=5)
    grid.fit(X_train, y_train)
    classifier_score = grid.best_score_
    classifier = grid.best_estimator_
    
    print('train done', flush=True)

    #### test model
    df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(sig_physio, classifier, prms_tracker, sig_mode)

    #### trim vectors
    #cond_i, cond = 1, conditions[1]
    for cond_i, cond in enumerate(conditions):

        if cond_i == 0:

            start = trig[cond][0]/prms_tracker['srate'] - trim_edge
            stop = trig[cond][1]/prms_tracker['srate'] + trim_between

            mask_start = (start <= predictions_time) & (predictions_time <= stop)

            predictions_trim = predictions[mask_start] 
            label_vec_trim = label_vec[mask_start]
            trig_odor_trim = trig_odor[mask_start] 

        elif cond_i == len(conditions)-1:

            start = trig[cond][0]/prms_tracker['srate'] - trim_between
            stop = trig[cond][1]/prms_tracker['srate'] + trim_edge

            mask_start = (start <= predictions_time) & (predictions_time <= stop) 

            predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
            label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
            trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

        else:

            start = trig[cond][0]/prms_tracker['srate'] - trim_between
            stop = trig[cond][1]/prms_tracker['srate'] + trim_between

            mask_start = (start <= predictions_time) & (predictions_time <= stop)

            predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
            label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
            trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

        if debug:

            plt.plot(predictions_trim, label='prediction', linestyle='--')
            plt.plot(label_vec_trim, label='real')
            plt.plot(trig_odor_trim, label='odor_trig')
            plt.legend()
            plt.show()

    #### resample
    f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions_trim.shape[-1]), predictions_trim, kind='linear')
    predictions_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

    f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec_trim.shape[-1]), label_vec_trim, kind='linear')
    label_vec_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

    f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor_trim.shape[-1]), trig_odor_trim, kind='linear')
    trig_odor_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

    #### plot
    fig_whole, ax = plt.subplots(figsize=(18,9))
    ax.plot(label_vec, color='k', label='real', linestyle=':', linewidth=3)
    ax.plot(predictions, color='y', label='prediction')
    ax.plot(trig_odor, color='r', label='odor_trig', linestyle='--')
    ax.set_title(f"perf : {np.round(classifier_score, 3)}")
    plt.suptitle(f'RAW {sig_mode}')
    plt.legend()
    fig_whole.show()
    # plt.close()

    fig_trim, ax = plt.subplots(figsize=(18, 9))
    ax.plot(label_vec_trim_resampled, color='k', label='real', linestyle=':', linewidth=3)
    ax.plot(predictions_trim_resampled, color='y', label='prediction')
    ax.plot(trig_odor_trim_resampled, color='r', label='odor_trig', linestyle='--')
    ax.set_title(f"perf : {np.round(classifier_score, 3)}")
    plt.suptitle(f'TRIMMED {sig_mode}')
    plt.legend()
    fig_trim.show() 
    # plt.close()












################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':

    sig_mode = 'ecg'
    sig_mode = 'ppg'

    hrv_tracker_exploration(sig_mode)




