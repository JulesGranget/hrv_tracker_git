

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pandas as pd
import xarray as xr
import mne
import glob
import neo

from n00_analysis_functions import *
from n00_config_params import *

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA

debug = False



################################
######## LOAD DATA ########
################################









def adjust_trig(sujet, data_aux, trig):


    #### adjust
    if sujet == 'CHEe':
        
        trig_name = ['CV_start', 'CV_stop', '31',   '32',   '11',   '12',   '71',   '72',   '11',   '12',   '51',    '52',    '11',    '12',    '51',    '52',    '31',    '32']
        trig_time = [0,          153600,    463472, 555599, 614615, 706758, 745894, 838034, 879833, 971959, 1009299, 1101429, 1141452, 1233580, 1285621, 1377760, 1551821, 1643948]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)    

    if sujet == 'GOBc':

        trig_name = ['MV_start', 'MV_stop']
        trig_time = [2947600,    3219072]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        index_append = [len(trig.name), len(trig.name)+1]
        trig_append = pd.DataFrame(trig_load, index=index_append)
        trig = trig.append(trig_append)
    
    if sujet == 'MAZm':
        
        trig_name = ['CV_start','CV_stop',  '31',   '32',   '11',   '12',   '31',   '32',   '11',   '12',   '51',   '52',   '11',   '12',   '51',   '52',   '61',    '62',    '61',    '62',    'MV_start', 'MV_stop']
        trig_time = [0,         164608,     164609, 240951, 275808, 367946, 396690, 488808, 529429, 621558, 646959, 739078, 763014, 855141, 877518, 969651, 1102256, 1194377, 1218039, 1310170, 1391000,    1558000]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)

    if sujet == 'TREt':

        trig_name = ['61',    '62',    'MV_start', 'MV_stop']
        trig_time = [1492770, 1584894, 1679260,    1751039]
        
        trig = trig.drop(labels=range(51,59), axis=0, inplace=False)

        trig_load = {'name' : trig_name, 'time' : trig_time}
        index_append = [i for i in range(len(trig.name), (len(trig.name)+len(trig_name)))]
        trig_append = pd.DataFrame(trig_load, index=index_append)
        trig = trig.append(trig_append)

    if sujet == 'MUGa':
        
        trig_name = ['CV_start', 'CV_stop']
        trig_time = [17400,       98700]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)    

    if sujet == 'BANc':

        trig_name = ['CV_start', 'CV_stop']
        trig_time = [0,       data_aux[0,:].shape[0]]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)  

    if sujet == 'KOFs':

        trig_name = ['CV_start', 'CV_stop']
        trig_time = [0,       data_aux[0,:].shape[0]]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)  

    if sujet == 'LEMl':

        trig_name = ['CV_start', 'CV_stop']
        trig_time = [0,       data_aux[0,:].shape[0]]

        trig_load = {'name' : trig_name, 'time' : trig_time}
        trig = pd.DataFrame(trig_load)  

    return trig





def get_ecg(sujet_list):

    data_allsujet = {}

    #sujet_i = sujet_list[3]
    for sujet_i in sujet_list:

        data_aux, chan_list_aux, trig, srate = extract_data_trc(sujet_i)
        trig = adjust_trig(sujet_i, data_aux, trig)

        data_allsujet[sujet_i] = {'ecg' : data_aux[-1,:], 'trig' : trig, 'srate' : srate}

        del data_aux

    return data_allsujet





########################################
######## TRACKING FUNCTION ########
########################################


def zscore(data):
    zscore_data = (data - np.mean(data))/np.std(data)
    return zscore_data







#sujet_i = 1
#ecg = data_allsujet[sujet_i]['ecg']
def hrv_tracker(ecg, win_size, srate, srate_resample_hrv, compute_PSD=True):

    #### load cR
    ecg = scipy.signal.detrend(ecg)
    try:
        ecg = mne.filter.filter_data(ecg, srate, 8, 15, verbose='CRITICAL')
    except:
        ecg = np.array([np.float64(i) for i in ecg])
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
#ecg = data_ecg_stress[sujet_i][0,:]
def hrv_tracker_svm(ecg, win_size, srate, srate_resample_hrv, classifier, compute_PSD=False):

    #### load cR
    ecg = scipy.signal.detrend(ecg)
    try:
        ecg = mne.filter.filter_data(ecg, srate, 8, 15, verbose='CRITICAL')
    except:
        ecg = np.array([np.float64(i) for i in ecg])
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
        predictions = classifier.predict(df_res.values)
    times = [ecg_cR[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(ecg_cR)-cR_initial):
        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR[cR_i])
        if compute_PSD:
            df_slide = ecg_analysis_homemade(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
        else:
            df_slide = ecg_analysis_homemade_stats(ecg[int(ecg_cR_sliding_win[0]*srate):int(ecg_cR_sliding_win[-1]*srate)], srate, srate_resample_hrv, fig_token=False)
            predictions = np.append(predictions, classifier.predict(df_slide.values))
        df_res = pd.concat([df_res, df_slide], axis=0)

        times.append(ecg_cR[cR_i])

    return df_res, times, predictions
























################################
######## COMPUTE ########
################################

if __name__ == '__main__':


    ################################
    ######## TRACKER DATA ########
    ################################


    #### get data
    data_allsujet = get_ecg(sujet_list)


    #### params
    srate_resample_hrv = 10
    win_size_sec = 30

    #### compute
    df_allsujet = {}
    for sujet_i in data_allsujet.keys():

        srate = data_allsujet[sujet_i]['srate']
        win_size = int(win_size_sec*srate)
        df_res, times = hrv_tracker(data_allsujet[sujet_i]['ecg'], win_size, srate, srate_resample_hrv, compute_PSD=False)
        df_allsujet[sujet_i] = {'df_res' : df_res, 'times' : times}


    #### identify when is the cR
    df_allsujet_label = {}
    for sujet_i in data_allsujet.keys():
        period_label = []
        trig_sujet_i = data_allsujet[sujet_i]['trig']
        #i = df_allsujet[sujet_i]['times'][1000]
        for i in df_allsujet[sujet_i]['times']:
            
            time_i = i*data_allsujet[sujet_i]['srate']

            if time_i >= trig_sujet_i['time'].values[-1]:
                period_label.append('NO_COND')
                continue

            else:
                for trigger_i, trigger_time in enumerate(trig_sujet_i['time'].values):
                    if time_i <= trigger_time:
                        break
                    else:
                        continue
                trigger_name_i = trig_sujet_i['name'].values[trigger_i-1]
                if trigger_name_i in identify_trig.keys():
                    period_label.append(identify_trig[trigger_name_i])
                else:
                    period_label.append('NO_COND')

        period_label_number = [label_condition[label_i] for label_i in period_label]
        
        df_allsujet_label[sujet_i] = {'label_name' : np.array(period_label), 'label_number' : np.array(period_label_number)}




    #### verify protocol for sujet
    if debug:

        for sujet_i in data_allsujet.keys():
            plt.plot(df_allsujet_label[sujet_i]['label_number'])
            plt.show()











    
    ################################
    ######## PCA ALL TIME ########
    ################################

    #### compute PCA


    #sujet_i = 'CHEe'
    #sujet_i = 'GOBc' 
    #sujet_i = 'MAZm' 
    #sujet_i = 'TREt' 

    x = df_allsujet[sujet_i]['df_res']

    #### PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    pca.explained_variance_ratio_

    #### plot PCA

    plt.figure()
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.title('PCA')

    #targets = ['RD_CV' , 'RD_FV', 'RD_SV', 'RD_AV', 'FR_CV', 'FR_MV', 'NO_COND']    
    #colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c']

    targets = ['RD_CV' , 'RD_FV', 'RD_SV', 'RD_AV', 'FR_CV', 'FR_MV']
    colors = ['r', 'g', 'b', 'k', 'm', 'y']

    for target, color in zip(targets,colors):
        mask = np.where(df_allsujet_label[sujet_i]['label_name'] == target)[0]
        plt.scatter(principalComponents[mask,0], principalComponents[mask,1], c = color, s = 20, label=target)
    plt.legend()
    plt.show()
    
    

    #### check coeef for PCA
    labels = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    coeff = np.transpose(pca.components_)

    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.show()



    
    
    
    
    
    ################################
    ######## PCA WITH TIME ########
    ################################


    #sujet_i = 'CHEe'
    #sujet_i = 'GOBc' 
    #sujet_i = 'MAZm' 
    #sujet_i = 'TREt' 


    x = df_allsujet[sujet_i]['df_res']

    #### PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    #### plot
    time_chunk = 60
    time_chunk_list = np.arange(0, data_allsujet[sujet_i]['ecg'].shape[0]/data_allsujet[sujet_i]['srate'], time_chunk)

    sujet_label = df_allsujet_label[sujet_i]['label_name']

    targets = ['RD_CV' , 'RD_FV', 'RD_SV', 'RD_AV', 'FR_CV', 'FR_MV', 'NO_COND']    
    colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c']

    for time_chunk_i in range(len(time_chunk_list))[:]:
        plt.figure()
        plt.xlabel('Principal Component - 1')
        plt.ylabel('Principal Component - 2')
        plt.title("PCA")
        if time_chunk_i == 0:
            mask = np.where(np.array(df_allsujet[sujet_i]['times']) <= time_chunk_list[time_chunk_i])[0]
        elif time_chunk_i == len(time_chunk_list)-1:
            mask = np.where(np.array(df_allsujet[sujet_i]['times']) >= time_chunk_list[time_chunk_i])[0]
        else:
            mask = np.where((np.array(df_allsujet[sujet_i]['times']) >= time_chunk_list[time_chunk_i]) & (np.array(df_allsujet[sujet_i]['times']) <= time_chunk_list[time_chunk_i+1]))[0]

        for cond_i, cond in enumerate(targets):
            mask_cond = np.where(sujet_label[mask] == cond)[0]
            plt.scatter(principalComponents[mask,0][mask_cond], principalComponents[mask,1][mask_cond], c = colors[cond_i], s = 25, label=cond)
        
        plt.xlim(np.min(principalComponents[:,0]), np.max(principalComponents[:,0]))
        plt.ylim(np.min(principalComponents[:,1]), np.max(principalComponents[:,1]))
        plt.legend()
        plt.show(block=False)

        plt.pause(2)

        plt.close()


    



    ################################
    ######## SVM WHOLE DATA ########
    ################################


    #sujet_i = 'CHEe'
    #sujet_i = 'GOBc' 
    #sujet_i = 'MAZm' 
    #sujet_i = 'TREt' 

    #### prepare data
    target_svm = df_allsujet_label[sujet_i]['label_name']
    data_svm = df_allsujet[sujet_i]['df_res']

    data_train, data_test, target_train, target_test = train_test_split(data_svm, target_svm, test_size=0.3,random_state=109) # 70% training and 30% test


    #### SVM
    #clf = svm.SVC(kernel='linear')
    #clf = svm.SVC(kernel='poly') 
    clf = svm.SVC(kernel='rbf', gamma=2, C=1)
    clf.fit(data_train, target_train)

    target_pred = clf.predict(data_test)

    accuracy_score = metrics.accuracy_score(target_test, target_pred)
    
    print("Accuracy:", accuracy_score)








    ########################################
    ######## SVM PREDICTIONS ########
    ########################################


    #sujet_i = 'CHEe'
    #sujet_i = 'GOBc' 
    #sujet_i = 'MAZm' 
    #sujet_i = 'TREt' 

    #### prepare data            
    target_svm = df_allsujet_label[sujet_i]['label_number']
    data_svm = df_allsujet[sujet_i]['df_res']

    data_sel_train = .6
    data_train = data_svm[:int(data_svm.shape[0]*data_sel_train)]
    data_test = data_svm[int(data_svm.shape[0]*(1-data_sel_train)):]
    target_train = target_svm[:int(data_svm.shape[0]*data_sel_train)]
    target_test = target_svm[int(data_svm.shape[0]*(1-data_sel_train)):]

    #### SVM
    #classifier = svm.SVC(kernel='linear') 
    #classifier = svm.SVC(C=1 ,kernel='poly') 
    classifier = svm.SVC(kernel='rbf', gamma=2, C=1)
    
    classifier.fit(data_train, target_train)


    #### get values
    win_size_sec = 30
    srate = data_allsujet[sujet_i]['srate']
    win_size = int(win_size_sec*srate)
    df_res, times, predictions = hrv_tracker_svm(data_allsujet[sujet_i]['ecg'], win_size, srate, srate_resample_hrv, classifier, compute_PSD=False)


    #### plot predictions
    plt.plot(target_svm, label='real')
    plt.plot(predictions, label='prediction')
    plt.legend()
    plt.show()
   



