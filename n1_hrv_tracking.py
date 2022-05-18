

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal 
import os
import pandas as pd
import xarray as xr
import neurokit2 as nk
import mne

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import joblib 
import seaborn as sns
import pandas as pd

import gc

from progress.bar import IncrementalBar

from n0_analysis_functions import *

debug = False




################################
######## LOAD DATA ########
################################


def load_data_anis():

    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/Etudiants/NBuonviso202201_trigeminal_sna_Anis/Data_Preprocessed/all_subjects')
    xr_data = xr.open_dataarray('da_time.nc')

    return xr_data


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
            load_data[sujet_i][0,:] = load_data[sujet_i][0,:]*-1

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
#ecg, win_size, srate, srate_resample_hrv, classifier, odor_trig_win, metric_used = data_ecg_stress[f'hrv_tracker_prediction_{sujet_i}']['data'][0,:], 30000, srate, srate_resample_hrv, clf, 10, labels_used
def hrv_tracker_svm(ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, compute_PSD=False):

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
        trig_odor = [1]
    times = [ecg_cR[cR_initial]]

    #### progress bar
    #bar = IncrementalBar('Countdown', max = len(ecg_cR)-cR_initial)

    #### sliding on other cR
    for cR_i in range(len(ecg_cR)-cR_initial):

        #bar.next()

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
            trig_odor_mean = np.round(np.mean(predictions[-odor_trig_n_bpm:]))
            if trig_odor_mean >= 1.5:
                trig_odor.append(2)
            else:
                trig_odor.append(1)
        else:
            trig_odor.append(1)

        times.append(ecg_cR[cR_i])

    #bar.finish()

    return df_res, times, predictions, trig_odor




def get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter):


    #### params
    win_size = int(win_size_sec*srate)

    #### compute tracking data
    df_allsujet = {}
    for sujet_i in data_ecg_stress.keys():

        df_res, times = hrv_tracker(data_ecg_stress[sujet_i]['data'][0, :], win_size, srate, srate_resample_hrv, compute_PSD=False)
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

    
    ########################################
    ######## LOAD DATA STRESS ########
    ########################################
    
    
    
    #### load data
    data_ecg_stress = load_data_stress_relax()
    
    
    ################################
    ######## DATA ANIS ########
    ################################
    
    
    
    #### params
    srate = 500
    srate_resample_hrv = 10
    win_size_min = 3
    win_size = int(win_size_min*srate*60)



    #### load_data
    xr_data = load_data_anis()
    data_ecg, order = generate_xr_data_compact(xr_data)


    #### compute for all sujet
    df_allsujet = []
    for sujet_i in range(data_ecg.shape[0]):

        df_res, times = hrv_tracker(data_ecg[sujet_i,:], win_size, srate, srate_resample_hrv, compute_PSD=True)
        df_allsujet.append([df_res, times])



    #### plot results for all metrics
    order_times = [i*300 for i in range(len(order))]

    #metric_i = df_res.columns.values[0]
    for metric_i in df_res.columns.values:
        scales = {'max' : [], 'min' : []}
        for sujet_i in range(data_ecg.shape[0]):
            plt.plot(df_allsujet[sujet_i][1], df_allsujet[sujet_i][0][metric_i].values, label=f's{sujet_i}')
            scales['max'].append(np.max(df_allsujet[sujet_i][0][metric_i].values))
            scales['min'].append(np.min(df_allsujet[sujet_i][0][metric_i].values))
        plt.vlines(order_times, ymin=np.min(scales['min']) ,ymax=np.max(scales['max']), colors='r')
        plt.legend()
        plt.title(metric_i)
        plt.show()







    ########################################
    ######## METRIC EVOLUTION ########
    ########################################

    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 60
    jitter = 0

    df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter)


    #metric_i = df_res.columns.values[0]

    sujet_i = 1
    for metric_i in df_res.columns.values:
        times = np.arange(data_ecg_stress[sujet_i][0].shape[0])/srate
        scales = {'max' : [], 'min' : []}

        plt.plot(times, zscore(scipy.signal.detrend(data_ecg_stress[sujet_i][1:])).reshape(-1), label=f'respi')
        plt.plot(df_allsujet[sujet_i][1], zscore(df_allsujet[sujet_i][0][metric_i].values), label=f'ecg')
        
        scales['max'].append(np.max(df_allsujet[sujet_i][0][metric_i].values))
        scales['min'].append(np.min(df_allsujet[sujet_i][0][metric_i].values))
        #plt.vlines(order_times, ymin=np.min(scales['min']) ,ymax=np.max(scales['max']), colors='r')
        plt.legend()
        plt.title(metric_i)
        plt.show()







    
    ################################
    ######## PCA WITH JITTER ########
    ################################

    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 60
    jitter = 0

    df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter)

    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    labels_used = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    #labels_used = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    #### plot protocol
    plt.plot(df_allsujet['hrv_tracker_train_1']['tracker'][1], df_allsujet['hrv_tracker_train_1']['labels_numeric'])
    plt.show()

    plt.plot(df_allsujet['hrv_tracker_prediction_1']['tracker'][1], df_allsujet['hrv_tracker_prediction_1']['labels_numeric'])
    plt.show()

    #### plot one metric
    plt.plot(df_allsujet['hrv_tracker_prediction_1']['tracker'][1], df_allsujet['hrv_tracker_prediction_1']['tracker'][0]['HRV_SDNN'].values)
    plt.title('HRV_SDNN')
    plt.show()



    #### PCA
    data_set = 'hrv_tracker_prediction_1'
    x = df_allsujet[data_set]['tracker'][0][labels_used].values

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    pca.explained_variance_ratio_

    #### plot
    plt.figure()
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.title("PCA")
    targets = ['FR_CV', 'RD_FV', 'RD_SV']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        mask = np.where(df_allsujet[data_set]['labels_names'] == target)[0]
        plt.scatter(principalComponents[mask,0], principalComponents[mask,1], c = color, s = 50, label=target)
    plt.legend()
    plt.show()

    
    #### check coeef for PCA
    coeff = np.transpose(pca.components_)

    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.2, width=1e-4)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels_used[i], color = 'g', ha = 'center', va = 'center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.show()



    
    
    
    
    
    ################################
    ######## PCA WITH TIME ########
    ################################


    data_set = 'hrv_tracker_prediction_1'

    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    labels_used = ['HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    x = df_allsujet[data_set]['tracker'][0][labels_used].values

    #### PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    #### plot
    time_chunk = 60
    time_chunk_list = np.arange(0, int(data_ecg_stress[data_set]['data'].shape[1]/srate), time_chunk)

    # time_chunk_i = 1
    for time_chunk_i, _ in enumerate(time_chunk_list):
        plt.figure()
        plt.xlabel('Principal Component - 1')
        plt.ylabel('Principal Component - 2')
        plt.title("PCA")
        if time_chunk_i == 0:
            mask = np.where(np.array(df_allsujet[data_set]['tracker'][1]) <= time_chunk_list[time_chunk_i])[0]
        elif time_chunk_i == len(time_chunk_list)-1:
            mask = np.where(np.array(df_allsujet[data_set]['tracker'][1]) >= time_chunk_list[time_chunk_i])[0]
        else:
            mask = np.where((np.array(df_allsujet[data_set]['tracker'][1]) >= time_chunk_list[time_chunk_i]) & (np.array(df_allsujet[data_set]['tracker'][1]) <= time_chunk_list[time_chunk_i+1]))[0]

        mask_FR_CV = np.where(df_allsujet[data_set]['labels_names'].reshape(-1)[mask] == 'FR_CV')[0]
        mask_RD_FV = np.where(df_allsujet[data_set]['labels_names'].reshape(-1)[mask] == 'RD_FV')[0]
        mask_RD_SV = np.where(df_allsujet[data_set]['labels_names'].reshape(-1)[mask] == 'RD_SV')[0]
        
        plt.scatter(principalComponents[mask,0][mask_FR_CV], principalComponents[mask,1][mask_FR_CV], c = 'r', s = 25, label='FR_CV')
        plt.scatter(principalComponents[mask,0][mask_RD_FV], principalComponents[mask,1][mask_RD_FV], c = 'g', s = 25, label='RD_FV')
        plt.scatter(principalComponents[mask,0][mask_RD_SV], principalComponents[mask,1][mask_RD_SV], c = 'b', s = 25, label='RD_SV')

        plt.xlim(np.min(principalComponents[:,0]), np.max(principalComponents[:,0]))
        plt.ylim(np.min(principalComponents[:,1]), np.max(principalComponents[:,1]))
        plt.legend()
        plt.show(block=False)

        plt.pause(1)

        plt.close()


    










    ################################
    ######## SVM TEST ########
    ################################

    #### get data
    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 30

    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    labels_used = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

    score_all_sujet = {}
    jitter_list = [0, 30, 60, 90]

    #jitter = 0
    for jitter in jitter_list:

        print(f'compute jitter : {jitter}')

        score_all_sujet[jitter] = {}
    
        df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter)

        #### params
        all_sujet = ['hrv_stress_1', 'hrv_stress_2', 'hrv_tracker_prediction_1', 'hrv_tracker_prediction_2']
        sujet_i = 'hrv_tracker_prediction_1'

        def conpute_svm_score(sujet_i):

            score_sujet_i = {}
            
            #### prepare data
            labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
            labels_used = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']

            target_svm = df_allsujet[sujet_i]['labels_numeric']
            data_svm = df_allsujet[sujet_i]['tracker'][0][labels_used].values

            data_train, data_test, target_train, target_test = train_test_split(data_svm, target_svm, test_size=0.4,random_state=109) # 70% training and 30% test

            #### apply svm
            for kernel_i in ['linear', 'poly', 'rbf']:

                print(f'{sujet_i} : {kernel_i}')

                if kernel_i == 'rbf':
                    clf = svm.SVC(kernel=kernel_i, gamma=2, C=1)
                else:
                    clf = svm.SVC(kernel=kernel_i)

                clf.fit(data_train, target_train)

                #Predict the response for test dataset
                target_pred = clf.predict(data_test)

                #Import scikit-learn metrics module for accuracy calculation
                accuracy_svm = metrics.accuracy_score(target_test, target_pred)

                score_sujet_i[kernel_i] = np.round(accuracy_svm, 2)

            return score_sujet_i

        compilation_score_sujet = joblib.Parallel(n_jobs = 6, prefer = 'processes')(joblib.delayed(conpute_svm_score)(sujet_i) for sujet_i in all_sujet)

        #### reorganize data
        for sujet_i, sujet in enumerate(all_sujet):
            score_all_sujet[jitter][sujet] = compilation_score_sujet[sujet_i]

    #### create dataframe
    df_svm = pd.DataFrame(columns=['sujet', 'jitter', 'kernel', 'score'])
    for sujet_i in all_sujet:
        for kernel_i in ['linear', 'poly', 'rbf']:
            for jitter in jitter_list:
                data_dict = {'sujet':[sujet_i], 'jitter':[jitter], 'kernel':[kernel_i], 'score':[score_all_sujet[jitter][sujet][kernel_i]]}
                df_svm = pd.concat([df_svm, pd.DataFrame(data=data_dict)])

    #### plot
    g = sns.catplot(x="jitter", y="score", hue="sujet", col="kernel", kind="point", data=df_svm)
    plt.show()













    ################################
    ######## SVM PREDICTION ########
    ################################

    #### params
    all_sujet = ['hrv_stress_1', 'hrv_stress_2', 'hrv_tracker_prediction_1', 'hrv_tracker_prediction_2', 'hrv_tracker_train_1', 'hrv_tracker_train_2']
    sujet_i = 2

    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 30
    odor_trig_n_bpm = 75
    win_size = int(win_size_sec*srate)
    jitter = 0


    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    # labels_used = ['HRV_MeanNN','HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    labels_used = ['HRV_MeanNN', 'HRV_S']
            
    #### select data
    df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter)
    data_train = df_allsujet[f'hrv_tracker_train_{sujet_i}']['tracker'][0][labels_used].values
    target_train = df_allsujet[f'hrv_tracker_train_{sujet_i}']['labels_numeric']

    df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, 0)
    data_test = df_allsujet[f'hrv_tracker_prediction_{sujet_i}']['tracker'][0][labels_used].values
    target_test = df_allsujet[f'hrv_tracker_prediction_{sujet_i}']['labels_numeric']

    #### prepare svm
    clf = svm.SVC(kernel='linear')
    # clf = svm.SVC(kernel='poly') 
    # clf = svm.SVC(kernel='rbf', gamma=2, C=1)
    
    clf.fit(data_train, target_train)


    #### get values
    df_res, times, predictions, trig_odor = hrv_tracker_svm(data_ecg_stress[f'hrv_tracker_prediction_{sujet_i}']['data'][0,:], win_size, srate, srate_resample_hrv, clf, labels_used, odor_trig_n_bpm, compute_PSD=False)


    #### plot predictions
    plt.plot(target_test, label='real')
    plt.plot(predictions, label='prediction')
    plt.plot(trig_odor, label='odor_trig')
    plt.title(f'sujet{sujet_i}_win{win_size_sec}_jitter{jitter}_metric{len(labels_used)}.png')
    plt.legend()
    # plt.show()
    
    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/HRV_Tracking/results/')
    plt.savefig(f'sujet{sujet_i}_win{win_size_sec}_jitter{jitter}_metric{len(labels_used)}.png')
    plt.show()














    ########################################
    ######## SVM PREDICTION SCORES ########
    ########################################

    #### params
    all_sujet = ['hrv_stress_1', 'hrv_stress_2', 'hrv_tracker_prediction_1', 'hrv_tracker_prediction_2', 'hrv_tracker_train_1', 'hrv_tracker_train_2']
    sujet_i = 1

    srate = 1000
    srate_resample_hrv = 10
    odor_trig_n_bpm = 75
    win_size_sec_list = [30, 60, 120]

    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    labels_used = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']


    def compute_svm_score_prediction(sujet_i, srate, srate_resample_hrv, win_size_sec, jitter, kernel, labels_used):

        print(f'sujet{sujet_i} win{win_size_sec} jitter{jitter} {kernel}')
        
        win_size = int(win_size_sec*srate)

        #### prepare data
        df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter)
        data_train = df_allsujet[f'hrv_tracker_train_{sujet_i}']['tracker'][0][labels_used].values
        target_train = df_allsujet[f'hrv_tracker_train_{sujet_i}']['labels_numeric']

        df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, 0)
        target_test = df_allsujet[f'hrv_tracker_prediction_{sujet_i}']['labels_numeric']

        #### apply svm
        if kernel == 'rbf':
            clf = svm.SVC(kernel=kernel, gamma=2, C=1)
        else:
            clf = svm.SVC(kernel=kernel)
        
        clf.fit(data_train, target_train)

        #### get values
        df_res, times, predictions, trig_odor = hrv_tracker_svm(data_ecg_stress[f'hrv_tracker_prediction_{sujet_i}']['data'][0,:], win_size, srate, srate_resample_hrv, clf, labels_used, odor_trig_n_bpm, compute_PSD=False)

        #Import scikit-learn metrics module for accuracy calculation
        accuracy_svm = metrics.accuracy_score(target_test, predictions)

        score_sujet_i = np.round(accuracy_svm, 2)

        #### prepare df
        df_res = pd.DataFrame(data={'sujet':[f'sujet{sujet_i}'], 'win_size':[win_size_sec], 'jitter':[jitter], 'labels':[len(labels_used)], 'kernel':[kernel], 'score':[score_sujet_i]})

        return df_res

    #### generate params
    params_fun = []
    for sujet_i in [1, 2]:
        for kernel_i in ['linear', 'poly', 'rbf']:
            for win_size_sec_list_i in win_size_sec_list:
                for jitter_i in [0, 30, win_size_sec_list_i]:
                    params_fun.append([sujet_i, srate, srate_resample_hrv, win_size_sec_list_i, jitter_i, kernel_i, labels_used])

    #### compute
    #compute_svm_score_prediction(sujet_i, srate, srate_resample_hrv, win_size_sec_list_i, jitter_i, kernel_i, labels_used)
    compilation_score_sujet = joblib.Parallel(n_jobs = 6, prefer = 'processes')(joblib.delayed(compute_svm_score_prediction)(sujet_i, srate, srate_resample_hrv, win_size_sec_list_i, jitter_i, kernel_i, labels_used) for sujet_i, srate, srate_resample_hrv, win_size_sec_list_i, jitter_i, kernel_i, labels_used in params_fun)

    #### extract data
    df_res = pd.DataFrame(columns=['sujet', 'win_size', 'jitter', 'labels', 'kernel', 'score']) 
    
    for df_i in compilation_score_sujet:
        df_res = pd.concat([df_res, df_i])

    #### plot
    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/HRV_Tracking/results/')

    sujet_i = 2
    g = sns.catplot(x="jitter", y="score", hue="win_size", col='kernel', kind="point", data=df_res[df_res['sujet'] == f'sujet{sujet_i}'])
    # plt.show()

    plt.savefig(f'comparison score sujet{sujet_i}.png')
    plt.show()


    




















    ########################################################
    ######## DECISION FUNCTION FOR TWO FEATURES ########
    ########################################################

    #### params
    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 60
    win_size = int(win_size_sec*srate)

    df_allsujet = get_data_tracking(data_ecg_stress, srate, srate_resample_hrv, win_size_sec, jitter)

    sujet_i = 1
            
    target_svm = df_allsujet[f'hrv_stress_{sujet_i+1}']['labels_numeric']
    data_svm = df_allsujet[f'hrv_stress_{sujet_i+1}']['tracker'][0].values

    #### select features
    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    feat_A = 'HRV_MeanNN'
    feat_B = 'HRV_pNN50'

    feat_A_i = labels.index(feat_A)
    feat_B_i = labels.index(feat_B)

        
    h = 0.02  # step size in the mesh
    # preprocess dataset, split into training and test part
    X, y = data_svm[:, [feat_A_i, feat_B_i]], target_svm
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = matplotlib.colors.ListedColormap(["#FF0000", "#0000FF"])

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

    plt.show()



    clf = svm.SVC(kernel='linear')
    # clf = svm.SVC(kernel='rbf', gamma=2, C=1)


    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)


    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

    
    
    plt.show()
    

    print(score)





    ########################################
    ######## IDENTIFY BEST PAIR ########
    ########################################



    #### params
    all_sujet = ['hrv_stress_1', 'hrv_stress_2', 'hrv_tracker_prediction_1', 'hrv_tracker_prediction_2', 'hrv_tracker_train_1', 'hrv_tracker_train_2']
    sujet_i = 'hrv_tracker_train_1'

    srate = 1000
    srate_resample_hrv = 10
    win_size_sec = 60
    win_size = int(win_size_sec*srate)

    best_pairs = {}
    for sujet_i in all_sujet:

        #### load data  
        target_svm = df_allsujet[sujet_i]['labels_numeric']
        data_svm = df_allsujet[sujet_i]['tracker'][0].values

        #### compute features
        score_pair = np.zeros((len(labels), len(labels)))

        labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
        for A_i, feat_A in enumerate(labels):
            for B_i, feat_B in enumerate(labels):

                feat_A_i = labels.index(feat_A)
                feat_B_i = labels.index(feat_B)

                    
                h = 0.02  # step size in the mesh
                # preprocess dataset, split into training and test part
                X, y = data_svm[:, [feat_A_i, feat_B_i]], target_svm
                X = StandardScaler().fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

                #clf = svm.SVC(kernel='linear')
                clf = svm.SVC(kernel='rbf', gamma=2, C=1)


                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                score_pair[A_i, B_i] = score

        best_pairs[sujet_i] = {}

        best_pairs[sujet_i]['mat'] = score_pair
        best_feat_A, best_feat_B = labels[np.where(score_pair == score_pair.max())[0][0]], labels[np.where(score_pair == score_pair.max())[0][1]]
        best_pairs[sujet_i]['best'] = f'best pair : {best_feat_A} & {best_feat_B} with : {score_pair.max()}'
        

    #### plot
    for sujet_i in all_sujet:

        fig, ax = plt.subplots()
        ax.matshow(best_pairs[sujet_i]['mat'])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f'{sujet_i} = \n' + best_pairs[sujet_i]['best'])
        plt.show()


