


import chunk
from distutils.util import execute
from unittest import expectedFailure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pandas as pd
from bycycle.cyclepoints import find_extrema
import seaborn as sns
import pingouin as pg
import xarray as xr
import neurokit2 as nk

from n0_analysis_functions import *

debug = False





################################
######## LOAD DATA ########
################################


def load_data():

    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/Etudiants/NBuonviso202201_trigeminal_sna_Anis/Data_Preprocessed/all_subjects')
    xr_data = xr.open_dataarray('da_time.nc')

    return xr_data





################################
######## COMPUTE ########
################################

def compute_df(xr_data, srate, srate_resample_hrv, chunk):

    print(chunk/srate)

    xr_chunk = xr_data.loc[:, 'ses02', ['free', 'confort', 'coherence'], :, 'ecg', :chunk/srate]

    df_res = pd.DataFrame(columns=['chunk', 'sujet', 'cond', 'trial', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_LFHF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2', 'HRV_S'])

    #sujet_i, sujet = 0,xr_chunk['participant'].data[0]
    for sujet_i, sujet in enumerate(xr_chunk['participant'].data):
        #cond_i, cond = 0, xr_chunk['bloc'].data[0]
        for cond_i, cond in enumerate(xr_chunk['bloc'].data):
            #trial_i, trial = 0, xr_chunk['trial'].data[0]
            for trial_i, trial in enumerate(xr_chunk['trial'].data):

                hrv_metrics_homemade = ecg_analysis_homemade(xr_chunk[sujet_i, cond_i, trial_i, :].data, srate, srate_resample_hrv, fig_token=False)
                #hrv_metrics = nk_analysis(xr_chunk[sujet_i, cond_i, trial_i, :].data, srate)

                data_dict = {'chunk' : chunk/srate, 'sujet' : sujet, 'cond' : cond, 'trial' : trial}
                df_ID = pd.DataFrame(data_dict, index=[0], columns=['chunk', 'sujet', 'cond', 'trial'])
                df_concat = pd.concat([df_ID, hrv_metrics_homemade], axis=1)
                df_res = pd.concat([df_res, df_concat], axis=0)


    return df_res








################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### params
    srate = 500
    srate_resample_hrv = 10
    within = 'cond'
    seuil = 0.05

    #### load_data
    xr_data = load_data()

    #### initiate df
    df_allchunk = pd.DataFrame(columns=['chunk', 'sujet', 'cond', 'trial', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2', 'HRV_S'])

    chunk_list =  [2.5*srate*60, 3*srate*60, 3.5*srate*60, 4*srate*60, 4.5*srate*60, 5*srate*60]
    #chunk = chunk_list[0]
    for chunk in chunk_list:
    
        #### compute
        chunk=int(chunk)
        df_res = compute_df(xr_data, srate, srate_resample_hrv, chunk)
        df_res = df_res.set_index(['chunk', 'sujet', 'cond', 'trial'])

        df_allchunk = pd.concat([df_allchunk, df_res.reset_index()], axis=0)

        #### stats
        df_pre, pre_signif, df_post, post_signif, conclusions = smart_stats(df_res, within, seuil, 'sujet')

        if chunk == chunk_list[0]:
            df_pre.insert(0, 'chunk', [int(chunk/srate)]*len(df_pre))
            df_pre_allchunk = df_pre
            df_post.insert(0, 'chunk', [int(chunk/srate)]*len(df_post))
            df_post_allchunk = df_post

        else:
            df_pre.insert(0, 'chunk', [int(chunk/srate)]*len(df_pre))
            df_post.insert(0, 'chunk', [int(chunk/srate)]*len(df_post))
            df_pre_allchunk = pd.concat([df_pre_allchunk, df_pre], axis=0)
            df_post_allchunk = pd.concat([df_pre_allchunk, df_post], axis=0)

    #### save
    os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/Etudiants/NBuonviso202201_trigeminal_sna_Anis/Analyses/Test_hrv_time')

    df_allchunk.to_excel('df_allchunk.xlsx')
    df_pre_allchunk.to_excel('df_pre_allchunk.xlsx')
    df_post_allchunk.to_excel('df_post_allchunk.xlsx')

    #### plot
    hrv_metrics = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2', 'HRV_S']
    #metric_i = hrv_metrics[0]
    for metric_i in hrv_metrics:
        g = sns.catplot(x="chunk", y=metric_i, hue="cond", capsize=.2, height=6, aspect=.75, kind="point", data=df_allchunk)
        g.savefig(f'{metric_i}.jpeg')

    g = sns.catplot(x="chunk", y='pval', hue="metric", capsize=.2, height=6, aspect=.75, kind="point", data=df_pre_allchunk)
    g.savefig(f'stats_pre_all.jpeg')

    g = sns.catplot(x="chunk", y='pval', hue="metric", capsize=.2, height=6, aspect=.75, kind="point", data=df_pre_allchunk)
    g.set(ylim=(0, 0.08))
    g.savefig(f'stats_pre_signi.jpeg')

    #plt.show()


    #### stats
    within = 'chunk'
    seuil = 0.05
    df_pre, pre_signif, df_post, post_signif, conclusions = smart_stats(df_allchunk.set_index(['chunk', 'sujet', 'cond', 'trial']), within, seuil, 'sujet')



