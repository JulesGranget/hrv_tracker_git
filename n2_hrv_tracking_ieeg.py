

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

from n0_analysis_functions import *
from n0_config import *

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA

debug = False



################################
######## LOAD DATA ########
################################

#path_data, sujet = 'D:\LPPR_CMO_PROJECT\Lyon\Data\iEEG', 'LYONNEURO_2019_CAPp'
def extract_data_trc(sujet):

    os.chdir(os.path.join(path_data,sujet))

    #### identify number of trc file
    trc_file_names = glob.glob('*.TRC')

    #### sort order TRC
    trc_file_names_ordered = []
    [trc_file_names_ordered.append(file_name) for file_name in trc_file_names if file_name.find('FR_CV') != -1]
    [trc_file_names_ordered.append(file_name) for file_name in trc_file_names if file_name.find('PROTOCOLE') != -1]

    #### extract file one by one
    print('#### EXTRACT TRC ####')
    data_whole = []
    chan_list_whole = []
    srate_whole = []
    events_name_whole = []
    events_time_whole = []
    #file_i, file_name = 1, trc_file_names[1]
    for file_i, file_name in enumerate(trc_file_names_ordered):

        #### current file
        print(file_name)

        #### extract segment with neo
        reader = neo.MicromedIO(filename=file_name)
        seg = reader.read_segment()
        print('len seg : ' + str(len(seg.analogsignals)))
        
        #### extract data
        data_whole_file = []
        chan_list_whole_file = []
        srate_whole_file = []
        events_name_file = []
        events_time_file = []
        #anasig = seg.analogsignals[2]
        for seg_i, anasig in enumerate(seg.analogsignals):
            
            chan_list_whole_file.append(anasig.array_annotations['channel_names'].tolist()) # extract chan
            data_whole_file.append(anasig[:, :].magnitude.transpose()) # extract data
            srate_whole_file.append(int(anasig.sampling_rate.rescale('Hz').magnitude.tolist())) # extract srate

        if srate_whole_file != [srate_whole_file[i] for i in range(len(srate_whole_file))] :
            print('srate different in segments')
            exit()
        else :
            srate_file = srate_whole_file[0]

        #### concatenate data
        for seg_i in range(len(data_whole_file)):
            if seg_i == 0 :
                data_file = data_whole_file[seg_i]
                chan_list_file = chan_list_whole_file[seg_i]
            else :
                data_file = np.concatenate((data_file,data_whole_file[seg_i]), axis=0)
                [chan_list_file.append(chan_list_whole_file[seg_i][i]) for i in range(np.size(chan_list_whole_file[seg_i]))]


        #### event
        if len(seg.events[0].magnitude) == 0 : # when its VS recorded
            events_name_file = ['CV_start', 'CV_stop']
            events_time_file = [0, len(data_file[0,:])]
        else : # for other sessions
            #event_i = 0
            for event_i in range(len(seg.events[0])):
                events_name_file.append(seg.events[0].labels[event_i])
                events_time_file.append(int(seg.events[0].times[event_i].magnitude * srate_file))

        #### fill containers
        data_whole.append(data_file)
        chan_list_whole.append(chan_list_file)
        srate_whole.append(srate_file)
        events_name_whole.append(events_name_file)
        events_time_whole.append(events_time_file)

    #### verif
    #file_i = 1
    #data = data_whole[file_i]
    #chan_list = chan_list_whole[file_i]
    #events_time = events_time_whole[file_i]
    #srate = srate_whole[file_i]

    #chan_name = 'p19+'
    #chan_i = chan_list.index(chan_name)
    #file_stop = (np.size(data,1)/srate)/60
    #start = 0 *60*srate 
    #stop = int( file_stop *60*srate )
    #plt.plot(data[chan_i,start:stop])
    #plt.vlines( np.array(events_time)[(np.array(events_time) > start) & (np.array(events_time) < stop)], ymin=np.min(data[chan_i,start:stop]), ymax=np.max(data[chan_i,start:stop]))
    #plt.show()

    #### concatenate 
    print('#### CONCATENATE ####')
    data = data_whole[0]
    chan_list = chan_list_whole[0]
    events_name = events_name_whole[0]
    events_time = events_time_whole[0]
    srate = srate_whole[0]

    if len(trc_file_names) > 1 :
        #trc_i = 0
        for trc_i in range(len(trc_file_names)): 

            if trc_i == 0 :
                len_trc = np.size(data_whole[trc_i],1)
                continue
            else:
                    
                data = np.concatenate((data,data_whole[trc_i]), axis=1)

                [events_name.append(events_name_whole[trc_i][i]) for i in range(len(events_name_whole[trc_i]))]
                [events_time.append(events_time_whole[trc_i][i] + len_trc) for i in range(len(events_time_whole[trc_i]))]

                if chan_list != chan_list_whole[trc_i]:
                    print('not the same chan list')
                    exit()

                if srate != srate_whole[trc_i]:
                    print('not the same srate')
                    exit()

                len_trc += np.size(data_whole[trc_i],1)

    
    #### no more use
    del data_whole
    del data_whole_file
    del data_file
    
    #### events in df
    event_dict = {'name' : events_name, 'time' : events_time}
    columns = ['name', 'time']
    trig = pd.DataFrame(event_dict, columns=columns)

    #### identify iEEG / respi / ECG

    print('#### AUX IDENTIFICATION ####')
    nasal_i = chan_list.index(aux_chan.get(sujet).get('nasal'))
    ecg_i = chan_list.index(aux_chan.get(sujet).get('ECG'))
    
    if aux_chan.get(sujet).get('ventral') == None:
        _data_ventral = np.zeros((data[nasal_i, :].shape[0]))
        data_aux = np.stack((data[nasal_i, :], _data_ventral, data[ecg_i, :]), axis = 0)
    else:
        ventral_i = chan_list.index(aux_chan.get(sujet).get('ventral'))
        data_aux = np.stack((data[nasal_i, :], data[ventral_i, :], data[ecg_i, :]), axis = 0)

    chan_list_aux = ['nasal', 'ventral', 'ECG']

    #### adjust ECG
    if sujet_ecg_adjust.get(sujet) == 'inverse':
        data_aux[-1,:] = data_aux[-1,:] * -1

    return data_aux, chan_list_aux, trig, srate








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
   



