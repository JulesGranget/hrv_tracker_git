

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal 
import os
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
import xarray as xr
import neurokit2 as nk
import mne



import joblib 
import seaborn as sns
import pandas as pd

import gc

from n00_analysis_functions import *

debug = False




################################
######## LOAD DATA ########
################################


def load_data_stress_relax():

    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/HRV_Tracking/data/')
    
    file_to_open = [file_i for file_i in os.listdir() if file_i.find('.vhdr') != -1]

    load_data = {}
    for file_i in file_to_open:
        load_data[file_i.split('.')[0]] = {}
        raw = mne.io.read_raw_brainvision(file_i, preload=True)
        chan_list_drop = [] 
        for nchan in raw.info['ch_names']:
            if nchan == 'ECG' or nchan == 'RespiVentrale':
                continue
            else:
                chan_list_drop.append(nchan)
        raw.drop_channels(chan_list_drop)
        load_data[file_i.split('.')[0]]['data'] = raw.get_data()

    #### verif
    if debug:
        plt.plot(load_data[1][0])
        plt.show()

    #data_ecg_stress = np.zeros((len(load_data), load_data[0].shape[0], load_data[0].shape[1])) 
    #for sujet_i in range(len(load_data)):
    #    data_ecg_stress[sujet_i,:,:] = load_data[sujet_i]
    #return data_ecg_stress

    data_shape_stress = []
    data_shape_train = []
    data_shape_prediction = []

    for file_i in load_data.keys():

        if file_i.find('hrv_stress') != -1:
        
            data_shape_stress.append(load_data[file_i]['data'].shape[1])

        elif file_i.find('hrv_tracker_train') != -1:
            
            data_shape_train.append(load_data[file_i]['data'].shape[1])

        elif file_i.find('hrv_tracker_prediction') != -1:
            
            data_shape_prediction.append(load_data[file_i]['data'].shape[1])
    
    for file_i in load_data.keys():

        if file_i == 'hrv_stress_1':
            load_data[file_i]['data'] = load_data[file_i]['data'][:, :np.min(data_shape_stress)]
            load_data[file_i]['trig'] = [3.126e5, 6.189e5, 9.146e5, 1.2116e6, 1.5194e6, 1.8166e6]

        elif file_i == 'hrv_stress_2':
            load_data[file_i]['data'] = load_data[file_i]['data'][:, :np.min(data_shape_stress)]
            load_data[file_i]['trig'] = [3.126e5, 6.189e5, 9.146e5, 1.2116e6, 1.5194e6, 1.8166e6]

        elif file_i == 'hrv_tracker_train_1':
            load_data[file_i]['data'] = load_data[file_i]['data'][:, :np.min(data_shape_train)]
            load_data[file_i]['trig'] = [3.126e5, 6.189e5, 9.146e5, 1.2116e6, 1.5194e6, 1.8166e6]

        elif file_i == 'hrv_tracker_train_2':
            load_data[file_i]['data'] = load_data[file_i]['data'][:, :np.min(data_shape_train)]
            load_data[file_i]['trig'] = [3.126e5, 6.189e5, 9.146e5, 1.2116e6, 1.5194e6, 1.8166e6]

        elif file_i == 'hrv_tracker_prediction_1':
            load_data[file_i]['data'] = load_data[file_i]['data'][:, :np.min(data_shape_prediction)]
            load_data[file_i]['trig'] = [3.051e5, 6.025e5, 8.949e5, 1.2060e6, 1.5032e6, 1.8009e6, 2.1125e6, 2.4059e6, 2.7026e6, 3.0077e6, 3.3069e6]

        elif file_i == 'hrv_tracker_prediction_2':
            load_data[file_i]['data'] = load_data[file_i]['data'][:, :np.min(data_shape_prediction)]
            load_data[file_i]['trig'] = [3.051e5, 6.025e5, 8.949e5, 1.2060e6, 1.5032e6, 1.8009e6, 2.1125e6, 2.4059e6, 2.7026e6, 3.0077e6, 3.3069e6]

    #### verif sig
    if debug:

        for file_i in load_data.keys():
            if file_i.find('hrv_tracker_prediction') != -1:
                plt.plot(zscore(load_data[file_i]['data'][-1, :]))
        plt.show()

    #### adjust ecg
    for file_i in load_data.keys():
        if file_i == 'hrv_stress_1':
            load_data[file_i]['data'][0,:] *= -1

        if debug:
            plt.plot(load_data[1][0])
            plt.plot(load_data[1][0,:])
            plt.vlines(ymin=np.min(load_data[0][1]), ymax=np.max(load_data[0][1]), color='r')
            plt.show()

        return load_data


################################
######## COMPUTE ########
################################


def generate_xr_data_compact(xr_data):

    xr_ecg = xr_data.loc[:, 'ses02', ['free', 'confort', 'coherence'], :, 'ecg', :]
    order = []

    #sujet_i = xr_ecg['participant'].values[1]
    for sujet_i in xr_ecg['participant'].values:
        
        #trial_i = xr_ecg['trial'].values[0]
        for trial_i in xr_ecg['trial'].values:

            #bloc_i = xr_ecg['bloc'].values[0]
            for bloc_i in xr_ecg['bloc'].values: 

                if trial_i == xr_ecg['trial'].values[0] and bloc_i == xr_ecg['bloc'].values[0]:
                    
                    ecg_sig_i = xr_ecg.loc[sujet_i, bloc_i, trial_i, :].values

                else:

                    ecg_sig_i = np.concatenate((ecg_sig_i, xr_ecg.loc[sujet_i, bloc_i, trial_i, :].values), axis=0)
                
                if sujet_i == xr_ecg['participant'].values[0]:
                
                    order.append(f'{trial_i}_{bloc_i}_{int(xr_ecg.loc[sujet_i, bloc_i, trial_i, :].values.shape[0]/srate/60)}min')

        if sujet_i == xr_ecg['participant'].values[0]:

            data_ecg = ecg_sig_i.reshape(1,len(ecg_sig_i))

        else:

            ecg_sig_i = ecg_sig_i.reshape(1,len(ecg_sig_i))
            data_ecg = np.concatenate((data_ecg, ecg_sig_i), axis=0)

    return data_ecg, order






########################################
######## TRACKING FUNCTION ########
########################################


def zscore(data):
    zscore_data = (data - np.mean(data))/np.std(data)
    return zscore_data


#sujet_i = 1
#ecg = data_ecg_stress[sujet_i]['data'][0, :]
def hrv_tracker(ecg, win_size, srate, srate_resample_hrv, compute_PSD=False):

    #### load cR
    ecg = scipy.signal.detrend(ecg)
    ecg = mne.filter.filter_data(ecg, srate, 8, 15, verbose='CRITICAL')
    ecg = zscore(ecg)

    ecg_cR = scipy.signal.find_peaks(ecg, distance=srate*0.5, prominence=np.mean(ecg)+np.std(ecg)*0.5)[0]
    ecg_cR = ecg_cR/srate

    RRI = np.diff(ecg_cR)

    #### verif
    if debug:
        #### ECG_cR
        times = np.arange(ecg.shape[0])/srate
        plt.plot(times, ecg)
        plt.vlines(ecg_cR, ymin=np.min(ecg) ,ymax=np.max(ecg), colors='r')
        plt.show()

        #### threshold
        times = np.arange(ecg.shape[0])/srate
        plt.plot(times, ecg)
        plt.hlines(np.mean(ecg), xmin=0 ,xmax=ecg.shape[0]/srate, colors='r')
        plt.hlines(np.mean(ecg)+np.std(ecg)*0.5, xmin=0 ,xmax=ecg.shape[0]/srate, colors='r')
        plt.hlines(np.mean(ecg)-np.std(ecg)*0.5, xmin=0 ,xmax=ecg.shape[0]/srate, colors='r')
        plt.show()

        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((ecg_cR[0], ecg_cR[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, ecg_cR[cR_initial])
        cR_initial += 1

    #### first point sliding win
    if compute_PSD:
        df_res = ecg_analysis_homemade(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
    else:
        df_res = ecg_analysis_homemade_stats(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
    times = [ecg_cR[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(ecg_cR)-cR_initial):
        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR[cR_i])
        if compute_PSD:
            df_slide = ecg_analysis_homemade(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
        else:
            df_slide = ecg_analysis_homemade_stats(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)

        df_res = pd.concat([df_res, df_slide], axis=0)

        times.append(ecg_cR[cR_i])

    return df_res, times




#sujet_i = 1
#ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict = ecg_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict
def hrv_tracker_svm(ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict, compute_PSD=False):

    #### load cR
    ecg = scipy.signal.detrend(ecg)
    ecg = mne.filter.filter_data(ecg, srate, 8, 15, verbose='CRITICAL')
    ecg = zscore(ecg)

    ecg_cR = scipy.signal.find_peaks(ecg, distance=srate*0.5, prominence=np.mean(ecg)+np.std(ecg)*0.5)[0]
    ecg_cR = ecg_cR/srate

    RRI = np.diff(ecg_cR)

    #### verif
    if debug:
        #### ECG_cR
        times = np.arange(ecg.shape[0])/srate
        plt.plot(times, ecg)
        plt.vlines(ecg_cR, ymin=np.min(ecg) ,ymax=np.max(ecg), colors='r')
        plt.show()

        #### threshold
        times = np.arange(ecg.shape[0])/srate
        plt.plot(times, ecg)
        plt.hlines(np.mean(ecg), xmin=0 ,xmax=ecg.shape[0]/srate, colors='r')
        plt.hlines(np.mean(ecg)+np.std(ecg)*0.5, xmin=0 ,xmax=ecg.shape[0]/srate, colors='r')
        plt.hlines(np.mean(ecg)-np.std(ecg)*0.5, xmin=0 ,xmax=ecg.shape[0]/srate, colors='r')
        plt.show()

        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((ecg_cR[0], ecg_cR[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, ecg_cR[cR_initial])
        cR_initial += 1

    #### first point sliding win
    if compute_PSD:
        df_res = ecg_analysis_homemade(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
    else:
        df_res = ecg_analysis_homemade_stats(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
        df_res = df_res[metric_used]
        predictions = classifier.predict(df_res.values)
        trig_odor = [labels_dict['FR_CV']]
    times = [ecg_cR[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(ecg_cR)-cR_initial):

        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR[cR_i])
        if compute_PSD:
            df_slide = ecg_analysis_homemade(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
        else:
            df_slide = ecg_analysis_homemade_stats(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
            df_slide = df_slide[metric_used]
            predictions = np.append(predictions, classifier.predict(df_slide.values))
        df_res = pd.concat([df_res, df_slide], axis=0)

        if predictions.shape[0] >= odor_trig_n_bpm:
            trig_odor_win = predictions.copy()[-odor_trig_n_bpm:]
            trig_odor_win[trig_odor_win == labels_dict['RD_SV']] = labels_dict['FR_CV']
            trig_odor_pred_i = np.round(np.mean(trig_odor_win))

            if trig_odor_pred_i != labels_dict['RD_FV']:
                trig_odor.append(labels_dict['FR_CV'])
            else:
                trig_odor.append(labels_dict['RD_FV'])
        
        else:
            trig_odor.append(labels_dict['FR_CV'])

        times.append(ecg_cR[cR_i])

    return df_res, times, predictions, trig_odor




def get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter):


    #### params
    win_size = int(win_size_sec*srate)

    #### compute tracking data
    df_allsujet = {}
    for sujet_i in data_ecg_stress.keys():

        _ecg = data_ecg_stress[sujet_i]['data'][0, :]
        df_res, times = hrv_tracker(_ecg, win_size, srate, srate_resample_hrv, compute_PSD=False)
        df_allsujet[sujet_i] = {}
        df_allsujet[sujet_i]['tracker'] = [df_res, times]

    #### compute label names
    if jitter == 0:
        for sujet_i in data_ecg_stress.keys():
            
            if sujet_i.find('hrv_stress') != -1:
                trig = data_ecg_stress[sujet_i]['trig']    
                period_label = []
                for i in df_allsujet[sujet_i]['tracker'][1]:
                    if i <= trig[0]/srate :
                        period_label.append('FR_CV')
                    elif i >= trig[0]/srate and i <= trig[1]/srate:
                        period_label.append('RD_FV')
                    elif i >= trig[1]/srate and i <= trig[2]/srate:
                        period_label.append('FR_CV')
                    elif i >= trig[2]/srate and i <= trig[3]/srate:
                        period_label.append('RD_FV')
                    elif i >= trig[3]/srate and i <= trig[4]/srate:
                        period_label.append('FR_CV')
                    elif i >= trig[4]/srate and i <= trig[5]/srate:
                        period_label.append('RD_FV')
                    if i >= trig[5]/srate:
                        period_label.append('FR_CV')

                df_allsujet[sujet_i]['labels_names'] = np.array(period_label)

            elif sujet_i.find('hrv_tracker_train') != -1:
                trig = data_ecg_stress[sujet_i]['trig']    
                period_label = []
                for i in df_allsujet[sujet_i]['tracker'][1]:
                    if i <= trig[0]/srate :
                        period_label.append('FR_CV')
                    elif i >= trig[0]/srate and i <= trig[1]/srate:
                        period_label.append('RD_FV')
                    elif i >= trig[1]/srate and i <= trig[2]/srate:
                        period_label.append('FR_CV')
                    if i >= trig[2]/srate:
                        period_label.append('RD_SV')

                df_allsujet[sujet_i]['labels_names'] = np.array(period_label)

            elif sujet_i.find('hrv_tracker_prediction') != -1:
                trig = data_ecg_stress[sujet_i]['trig']    
                period_label = []
                for i in df_allsujet[sujet_i]['tracker'][1]:
                    if i <= trig[0]/srate :
                        period_label.append('FR_CV')
                    elif i >= trig[0]/srate and i <= trig[1]/srate:
                        period_label.append('RD_FV')
                    elif i >= trig[1]/srate and i <= trig[2]/srate:
                        period_label.append('FR_CV')
                    elif i >= trig[2]/srate and i <= trig[3]/srate:
                        period_label.append('RD_SV')

                    elif i >= trig[3]/srate and i <= trig[4]/srate:
                        period_label.append('FR_CV')
                    elif i >= trig[4]/srate and i <= trig[5]/srate:
                        period_label.append('RD_FV')
                    elif i >= trig[5]/srate and i <= trig[6]/srate:
                        period_label.append('FR_CV')
                    elif i >= trig[6]/srate and i <= trig[7]/srate:
                        period_label.append('RD_SV')

                    elif i >= trig[7]/srate and i <= trig[8]/srate:
                        period_label.append('FR_CV')
                    elif i >= trig[8]/srate and i <= trig[9]/srate:
                        period_label.append('RD_FV')
                    elif i >= trig[9]/srate and i <= trig[10]/srate:
                        period_label.append('FR_CV')
                    if i >= trig[10]/srate:
                        period_label.append('RD_SV')

                df_allsujet[sujet_i]['labels_names'] = np.array(period_label)

    else:
        for sujet_i in data_ecg_stress.keys():
            
            if sujet_i.find('hrv_stress') != -1:
                trig = data_ecg_stress[sujet_i]['trig']    
                period_label = []
                for i in df_allsujet[sujet_i]['tracker'][1]:
                    if i <= trig[0]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[0]/srate + jitter and i <= trig[1]/srate + jitter:
                        period_label.append('RD_FV')
                    elif i >= trig[1]/srate + jitter and i <= trig[2]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[2]/srate + jitter and i <= trig[3]/srate + jitter:
                        period_label.append('RD_FV')
                    elif i >= trig[3]/srate + jitter and i <= trig[4]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[4]/srate + jitter and i <= trig[5]/srate + jitter:
                        period_label.append('RD_FV')
                    if i >= trig[5]/srate + jitter:
                        period_label.append('FR_CV')

                df_allsujet[sujet_i]['labels_names'] = np.array(period_label)

            elif sujet_i.find('hrv_tracker_train') != -1:
                trig = data_ecg_stress[sujet_i]['trig']    
                period_label = []
                for i in df_allsujet[sujet_i]['tracker'][1]:
                    if i <= trig[0]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[0]/srate + jitter and i <= trig[1]/srate + jitter:
                        period_label.append('RD_FV')
                    elif i >= trig[1]/srate + jitter and i <= trig[2]/srate + jitter:
                        period_label.append('FR_CV')
                    if i >= trig[2]/srate + jitter:
                        period_label.append('RD_SV')

                df_allsujet[sujet_i]['labels_names'] = np.array(period_label)

            elif sujet_i.find('hrv_tracker_prediction') != -1:
                trig = data_ecg_stress[sujet_i]['trig']    
                period_label = []
                for i in df_allsujet[sujet_i]['tracker'][1]:
                    if i <= trig[0]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[0]/srate + jitter and i <= trig[1]/srate + jitter:
                        period_label.append('RD_FV')
                    elif i >= trig[1]/srate + jitter and i <= trig[2]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[2]/srate + jitter and i <= trig[3]/srate + jitter:
                        period_label.append('RD_SV')

                    elif i >= trig[3]/srate + jitter and i <= trig[4]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[4]/srate + jitter and i <= trig[5]/srate + jitter:
                        period_label.append('RD_FV')
                    elif i >= trig[5]/srate + jitter and i <= trig[6]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[6]/srate + jitter and i <= trig[7]/srate + jitter:
                        period_label.append('RD_SV')

                    elif i >= trig[7]/srate + jitter and i <= trig[8]/srate + jitter:
                        period_label.append('FR_CV')
                    elif i >= trig[8]/srate + jitter and i <= trig[9]/srate + jitter:
                        period_label.append('RD_FV')
                    elif i >= trig[9]/srate + jitter and i <= trig[10]/srate + jitter:
                        period_label.append('FR_CV')
                    if i >= trig[10]/srate + jitter:
                        period_label.append('RD_SV')

                df_allsujet[sujet_i]['labels_names'] = np.array(period_label)  


    #### compute label numeric
    label_correspondances = {'FR_CV' : 1, 'RD_FV' : 2, 'RD_SV' : 0}
    for sujet_i in data_ecg_stress.keys():
    
        numeric_labels = []
        period_label = df_allsujet[sujet_i]['labels_names']
        
        for label_i in period_label: 
            numeric_labels.append(label_correspondances[label_i])

        df_allsujet[sujet_i]['labels_numeric'] = np.array(numeric_labels)
            
    #### verif
    if debug: 

        for sujet_i in data_ecg_stress.keys():

            plt.plot(df_allsujet[sujet_i]['labels_numeric'])
            plt.title(sujet_i)
            plt.show()

    return df_allsujet




















if __name__ == '__main__':


    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn import metrics
    from sklearn.decomposition import PCA
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler, RobustScaler

    from sklearn.pipeline import make_pipeline, Pipeline

    from sklearn.metrics import confusion_matrix




    ########################
    ######## PARAMS ########
    ########################

    sujet = 1

    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 30
    odor_trig_n_bpm = 75
    win_size = int(win_size_sec*srate)
    jitter = 0

    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    # labels_used = ['HRV_MeanNN','HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    labels_used = ['HRV_MeanNN','HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']


    
    ########################################
    ######## LOAD DATA STRESS ########
    ########################################

    data_ecg_stress = load_data_stress_relax()
    df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, 0)

    X_train = df_allsujet[f'hrv_tracker_train_{sujet}']['tracker'][0].values
    y_train_name_labels = df_allsujet[f'hrv_tracker_train_{sujet}']['labels_names']
    y_train = LabelEncoder().fit_transform(y_train_name_labels)
    y_labels = np.unique(df_allsujet[f'hrv_tracker_train_{sujet}']['labels_names'])

    labels_dict = {key:value for (key,value) in zip(y_labels, range(y_labels.shape[0]))}

    target_test = LabelEncoder().fit_transform(df_allsujet[f'hrv_tracker_prediction_{sujet}']['labels_names'])

    ecg_train = data_ecg_stress[f'hrv_tracker_train_{sujet}']['data'][0,:]
    ecg_test = data_ecg_stress[f'hrv_tracker_prediction_{sujet}']['data'][0,:]



    ########################################
    ######## IDENTIFY JITTER ########
    ########################################

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_train)

    #### plot
    plt.figure()
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.title("PCA")
    targets = ['FR_CV', 'RD_FV', 'RD_SV']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        mask = np.where(y_train_name_labels[-X_train.shape[0]:] == target)[0]
        plt.scatter(principalComponents[mask,0], principalComponents[mask,1], c = color, s = 50, label=target)
    plt.legend()
    plt.show()

    #### apply jitter
    label_jitter = y_train_name_labels[-X_train.shape[0]:]
    n_jitter = 15
    label_jitter = np.concatenate([np.array(['FR_CV']*n_jitter), label_jitter[:-n_jitter]])
    
    #### plot
    for jitter_i in range(10):
    
        if jitter_i == 0:
            label_jitter = y_train_name_labels[-X_train.shape[0]:]
        else:
            label_jitter = np.concatenate([np.array(['FR_CV']*n_jitter), label_jitter[:-n_jitter]])

        plt.figure()
        plt.xlabel('Principal Component - 1')
        plt.ylabel('Principal Component - 2')
        plt.title(f"PCA : {jitter_i}")
        targets = ['FR_CV', 'RD_FV', 'RD_SV']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            mask = np.where(label_jitter == target)[0]
            plt.scatter(principalComponents[mask,0], principalComponents[mask,1], c = color, s = 50, label=target)
        plt.legend()
        plt.show()

    #### plot
    time_chunk = 30
    time_chunk_list = np.arange(0, int(ecg_train.shape[0]/srate), time_chunk)

    # time_chunk_i = 1
    for time_chunk_i, time_chunk_val in enumerate(time_chunk_list):
        plt.figure()
        plt.xlabel('Principal Component - 1')
        plt.ylabel('Principal Component - 2')
        plt.title("PCA")
        if time_chunk_i == 0:
            mask = np.where(np.array(df_allsujet[f'hrv_tracker_train_{sujet}']['tracker'][1]) <= time_chunk_val)[0]
        elif time_chunk_i == len(time_chunk_list)-1:
            mask = np.where(np.array(df_allsujet[f'hrv_tracker_train_{sujet}']['tracker'][1]) >= time_chunk_val)[0]
        else:
            mask = np.where((np.array(df_allsujet[f'hrv_tracker_train_{sujet}']['tracker'][1]) >= time_chunk_val) & (np.array(df_allsujet[f'hrv_tracker_train_{sujet}']['tracker'][1]) <= time_chunk_list[time_chunk_i+1]))[0]

        mask_FR_CV = np.where(df_allsujet[f'hrv_tracker_train_{sujet}']['labels_names'].reshape(-1)[mask] == 'FR_CV')[0]
        mask_RD_FV = np.where(df_allsujet[f'hrv_tracker_train_{sujet}']['labels_names'].reshape(-1)[mask] == 'RD_FV')[0]
        mask_RD_SV = np.where(df_allsujet[f'hrv_tracker_train_{sujet}']['labels_names'].reshape(-1)[mask] == 'RD_SV')[0]
        
        plt.scatter(principalComponents[mask,0][mask_FR_CV], principalComponents[mask,1][mask_FR_CV], c = 'r', s = 25, label='FR_CV')
        plt.scatter(principalComponents[mask,0][mask_RD_FV], principalComponents[mask,1][mask_RD_FV], c = 'g', s = 25, label='RD_FV')
        plt.scatter(principalComponents[mask,0][mask_RD_SV], principalComponents[mask,1][mask_RD_SV], c = 'b', s = 25, label='RD_SV')

        plt.xlim(np.min(principalComponents[:,0]), np.max(principalComponents[:,0]))
        plt.ylim(np.min(principalComponents[:,1]), np.max(principalComponents[:,1]))
        plt.legend()
        plt.show(block=False)

        plt.pause(2)

        plt.close()


    ################################
    ######## PIPELINE ########
    ################################
    
    #### make pipeline
    #SVC().get_params()
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps)

    #### find best model
    params = {

    # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
    'SVM__kernel' : ['linear', 'poly', 'rbf'],    
    'SVM__C' : [0.001, 0.1, 1, 10, 100, 10e5], 
    'SVM__gamma' : [0.1, 0.01]

    }

    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=6)
    grid.fit(X_train, y_train)
    grid.best_score_
    model = grid.best_estimator_

    #### plot confusion matrix
    X_train_cm, X_test_cm, y_train_cm, y_test_cm = train_test_split(X_train, y_train, test_size=0.4, random_state=5)
    conf_mat = confusion_matrix(y_test_cm, model.predict(X_test_cm))
    fig, ax = plt.subplots()
    ax.matshow(conf_mat)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[0]):
            c = conf_mat[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    ax.set_title(f'Confusion matrix : {np.round(model.score(X_test_cm, y_test_cm), 4)}')
    ax.set_yticklabels(['']+y_labels.tolist())
    #fig.show()
    plt.close('all')




    ################################
    ######## SVM PREDICTION ########
    ################################


    #### get values
    df_res, times, predictions, trig_odor = hrv_tracker_svm(ecg_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict)

    #### plot predictions
    plt.plot(target_test, label='real')
    plt.plot(predictions, label='prediction')
    plt.plot(trig_odor, label='odor_trig')
    plt.title(f'sujet{sujet}_win{win_size_sec}_jitter{jitter}_metric{len(labels_used)}')
    plt.legend()
    # plt.show()

    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/HRV_Tracking/results/')
    plt.savefig(f'sujet{sujet}_win{win_size_sec}_jitter{jitter}_metric{len(labels_used)}.png')
    plt.show()


