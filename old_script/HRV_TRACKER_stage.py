

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

from HRV_TRACKER_stage_ecg_clean import *
from HRV_TRACKER_stage_metrics import *

debug = False









################################
######## FUNCTIONS ######## 
################################


#sujet_i = 1
#ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict = ecg_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict
def hrv_tracker_svm(ecg, classifier, prms_tracker):

    srate = prms_tracker['srate']

    win_size_sec, metric_used, odor_trig_n_bpm = prms_tracker['win_size_sec'], prms_tracker['metric_list'], prms_tracker['odor_trig_n_bpm']
    win_size = int(win_size_sec*srate)

    #### load cR
    ecg, ecg_peaks = compute_ecg(ecg, srate)

    ecg_cR = np.zeros((ecg.shape[0]))
    ecg_cR[ecg_peaks['peak_index']] = 10

    ecg_cR_val = np.where(ecg_cR != 0)[0]/srate
    RRI = np.diff(ecg_cR_val)

    #### verif
    if debug:
        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((ecg_cR_val[0], ecg_cR_val[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, ecg_cR_val[cR_initial])
        cR_initial += 1

    #### first point sliding win
    RRI_win = np.diff(ecg_cR_sliding_win)
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
    predictions = classifier.predict(df_res.values)
    trig_odor = [0]
    times = [ecg_cR_val[cR_initial]]

    #### progress bar
    # bar = IncrementalBar('Countdown', max = len(ecg_cR_val)-cR_initial)

    #### sliding on other cR
    for cR_i in range(len(ecg_cR_val)-cR_initial):

        # bar.next()

        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR_val[cR_i])
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

        times.append(ecg_cR_val[cR_i])

    # bar.finish()

    times = np.array(times)
    trig_odor = np.array(trig_odor)

    return df_res, times, predictions, trig_odor






def get_data_hrv_tracker(ecg, prms_tracker):

    #### extract params
    srate = prms_tracker['srate']
    win_size_sec, jitter = prms_tracker['win_size_sec'], prms_tracker['jitter']
    win_size = int(win_size_sec*srate)

    #### precompute ecg
    ecg, ecg_peaks = compute_ecg(ecg, srate)

    ecg_cR = np.zeros((ecg.shape[0]))
    ecg_cR[ecg_peaks['peak_index'].values] = 10

    #### load cR
    ecg_cR_val = np.where(ecg_cR != 0)[0]/srate
    RRI = np.diff(ecg_cR_val)

    #### verif
    if debug:
        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((ecg_cR_val[0], ecg_cR_val[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, ecg_cR_val[cR_initial])
        cR_initial += 1

    #### first point sliding win
    RRI_win = np.diff(ecg_cR_sliding_win)
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
    times = [ecg_cR_val[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(ecg_cR_val)-cR_initial):
        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR_val[cR_i])
        RRI_win = np.diff(ecg_cR_sliding_win)
        df_slide = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
        
        df_res = pd.concat([df_res, df_slide], axis=0)

        times.append(ecg_cR_val[cR_i])

    times = np.array(times)

    return df_res, times











################################
######## COMPUTE ######## 
################################


def hrv_tracker_exploration():
    
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

    #### load data
    # os.chdir('/home/jules/Bureau/')

    ecg = np.load('ecg_hrv_tracker.npy')

    label_vec = np.load('label_hrv_tracker.npy')

    with open('trig_hrv_tracker.pkl', 'rb') as f:
        trig = pickle.load(f)

    if debug:

        plt.plot(ecg)
        plt.show()

        plt.plot(label_vec)
        plt.show()

    #### get metrics
    df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
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
    'SVM__C' : [1e5, 1e6], 
    'SVM__gamma' : [0.1, 0.01]
    }

    print('train', flush=True)

    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=5)
    grid.fit(X_train, y_train)
    classifier_score = grid.best_score_
    classifier = grid.best_estimator_
    
    print('train done', flush=True)

    #### test model
    df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(ecg, classifier, prms_tracker)

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
    plt.suptitle(f'RAW')
    plt.legend()
    fig_whole.show()
    # plt.close()

    fig_trim, ax = plt.subplots(figsize=(18, 9))
    ax.plot(label_vec_trim_resampled, color='k', label='real', linestyle=':', linewidth=3)
    ax.plot(predictions_trim_resampled, color='y', label='prediction')
    ax.plot(trig_odor_trim_resampled, color='r', label='odor_trig', linestyle='--')
    ax.set_title(f"perf : {np.round(classifier_score, 3)}")
    plt.suptitle(f'TRIMMED')
    plt.legend()
    fig_trim.show() 
    # plt.close()












################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':


    hrv_tracker_exploration()


