

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pandas as pd
import xarray as xr

import joblib 
import seaborn as sns
import pandas as pd

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n03_precompute_individual_model import *

debug = False








################################
######## COMPILATION ########
################################


def allsujet_hrv_tracker_mean():

    print('################')
    print(f'#### NO REF ####')
    print('################')

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    n_pnts_trim_resample = (len(conditions) + len(conditions) -1) * points_per_cond

    #hrv_tracker_mode = '2classes'
    for hrv_tracker_mode in ['4classes', '2classes']:

        xr_dict = {'sujet' : sujet_list, 'train' : train_percentage_values, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
        xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list.shape[0], 4, len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

        xr_dict = {'sujet' : sujet_list, 'train' : train_percentage_values, 'odor' : np.array(odor_list)}
        xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list.shape[0], 4 ,len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

        #### load results
        for sujet in sujet_list:

            os.chdir(os.path.join(path_precompute, 'predictions', 'subject_wise'))
            xr_hrv_tracker.loc[sujet,:,:,:] = np.squeeze(xr.load_dataarray(f'{hrv_tracker_mode}_no_ref_{sujet}_hrv_tracker_alltestsize.nc').values, 0)

        xr_hrv_tracker_mean = xr_hrv_tracker.mean(axis=0)
        xr_hrv_tracker_std = xr_hrv_tracker.std(axis=0)

        #### plot
        fig_trim, axs = plt.subplots(ncols=len(odor_list), nrows=len(train_percentage_values), figsize=(18, 9))
        for train_test_sel_i, train_test_sel in enumerate(train_percentage_values):
            for odor_i, odor in enumerate(odor_list):
                # std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
                # std_to_plot_down[std_to_plot_down < 0] = 0
                ax = axs[train_test_sel_i, odor_i]
                ax.plot(xr_hrv_tracker_mean.loc[train_test_sel, odor, 'prediction', :].values, color='y', label='prediction')
                ax.plot(xr_hrv_tracker_mean.loc[train_test_sel, odor, 'label', :].values, color='k', label='real', linewidth=1)
                ax.plot(xr_hrv_tracker_mean.loc[train_test_sel, odor, 'trig_odor', :].values, color='r', label='odor_trig')
                ax.plot(xr_hrv_tracker_mean.loc[train_test_sel, odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[train_test_sel, odor, 'prediction', :].values, 
                        color='r', label='odor_trig', linestyle=':', linewidth=0.5)
                # ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
                if train_test_sel_i == 0:
                    ax.set_title(f"odor : {odor}")
                if odor_i == 0:
                    ax.set_ylabel(f"train : {train_test_sel}")
        plt.suptitle(f'TRIMMED')
        plt.legend()
        # fig_trim.show()
        plt.close()

        ######## SAVE ########
        os.chdir(os.path.join(path_results, 'allsujet'))
        fig_trim.savefig(f'{hrv_tracker_mode}_allsujet_no_ref_hrv_tracker_whole.png')


    print('################')
    print(f'#### REF ####')
    print('################')

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    n_pnts_trim_resample = (len(conditions) + len(conditions) -1) * points_per_cond

    #hrv_tracker_mode = '4classes'
    for hrv_tracker_mode in ['4classes', '2classes']:

        xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
        xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

        xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list)}
        xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

        #### load results
        for sujet in sujet_list:

            os.chdir(os.path.join(path_precompute, 'predictions', 'subject_wise'))
            xr_hrv_tracker.loc[sujet,:,:,:] = np.squeeze(xr.load_dataarray(f'{hrv_tracker_mode}_o_ref_{sujet}_hrv_tracker.nc').values, 0)

        xr_hrv_tracker_mean = xr_hrv_tracker.mean(axis=0)
        xr_hrv_tracker_std = xr_hrv_tracker.std(axis=0)

        #### plot
        fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(13, 4))
        for odor_i, odor in enumerate(odor_list):
            # std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
            # std_to_plot_down[std_to_plot_down < 0] = 0
            ax = axs[odor_i]
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=1)
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[odor, 'prediction', :].values, 
                    color='r', label='odor_trig', linestyle=':', linewidth=0.5)
            # ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
            ax.set_title(f"odor : {odor}")
        plt.suptitle(f'TRIMMED')
        plt.legend()
        # fig_trim.show()
        plt.close()

        ######## SAVE ########
        os.chdir(os.path.join(path_results, 'allsujet'))
        fig_trim.savefig(f'{hrv_tracker_mode}_allsujet_o_ref_hrv_tracker_whole.png')









########################################################
######## COMPUTE ALLSUJET ONE CLASSIFIER ########
########################################################


def allsujet_hrv_tracker_one_classifier():

    print('################')
    print(f'#### NO REF ####')
    print('################')

    for hrv_tracker_mode in ['4classes', '2classes']:

        os.chdir(os.path.join(path_precompute, 'predictions', 'allsujet'))

        xr_allsujet = xr.load_dataarray(f'{hrv_tracker_mode}_o_ref_allsujettrain_hrv_tracker.nc')

        xr_hrv_tracker_mean = xr_allsujet.mean('sujet')
        xr_hrv_tracker_std = xr_allsujet.std('sujet')

        #### plot
        fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 7))
        for odor_i, odor in enumerate(odor_list):
            std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
            std_to_plot_down[std_to_plot_down < 0] = 0
            ax = axs[odor_i]
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=3)
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[odor, 'prediction', :].values, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
            ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
            ax.set_title(f"odor : {odor}")
        plt.suptitle(f'TRIMMED')
        plt.legend()
        # fig_trim.show()
        plt.close()

        ######## SAVE ########
        os.chdir(os.path.join(path_results, 'allsujet'))
        fig_trim.savefig(f'{hrv_tracker_mode}_one_classifier_allsujet_no_ref_hrv_tracker.png')









########################################################
######## RES FOR CLASSIFIER ALLSUJET ########
########################################################


def res_one_classifier_allsujet():

    os.chdir(os.path.join(path_precompute, 'predictions', 'allsujet'))

    hrv_tracker_mode = ['2classes', '4classes'] 

    for hrv_tracker_mode_sel in hrv_tracker_mode:

        xr_data = xr.open_dataarray(f'{hrv_tracker_mode_sel}_o_ref_allsujettrain_hrv_tracker.nc')
        xr_score = xr.open_dataarray(f'{hrv_tracker_mode_sel}_o_ref_allsujettrain_hrv_tracker_score.nc')
        xr_hrv_tracker_mean = xr_data.mean('sujet')
        xr_hrv_tracker_std = xr_data.std('sujet')

        #### plot & save
        fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 9))
        for odor_i, odor in enumerate(odor_list):
            std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
            std_to_plot_down[std_to_plot_down < 0] = 0
            ax = axs[odor_i]
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=3)
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
            ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[odor, 'prediction', :].values, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
            ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
            ax.set_title(f"odor : {odor}")
        plt.suptitle(f'TRIMMED')
        plt.legend()
        # fig_trim.show()
        plt.close()

        ######## SAVE ########
        os.chdir(os.path.join(path_results, 'allsujet'))
        fig_trim.savefig(f'{hrv_tracker_mode_sel}_one_classifier_allsujet_o_ref_hrv_tracker.png')
    


################################
######## SVM PARAMS ########
################################


def df_allsujet_SVM_params():

    df_allsujet_params = pd.DataFrame()

    for sujet in sujet_list:

        os.chdir(os.path.join(path_precompute, 'params'))

        if os.path.exists(f'{sujet}_hrv_tracker_SVM_test.xlsx'):

            df_sujet = pd.read_excel(f'{sujet}_hrv_tracker_SVM_test.xlsx')
            df_allsujet_params = pd.concat((df_allsujet_params, df_sujet))

        else:

            print(f"{sujet} not done")

    df_allsujet_params = df_allsujet_params.drop(columns=['Unnamed: 0'])

    os.chdir(os.path.join(path_results, 'params'))

    for hrv_tracker_mode_sel in ['2classes', '4classes']:

        df_plot = df_allsujet_params.query(f"hrv_tracker_mode == '{hrv_tracker_mode_sel}' and ref == 'no_ref'")

        fig, axs = plt.subplots(ncols=3, figsize=(17,6))
        for odor_i, odor in enumerate(odor_list):
            ax = axs[odor_i]
            sns.pointplot(data=df_plot.query(f"odor == '{odor}'"), x='train_percentage', y='score', hue='sujet', ax=ax)
            ax.set_title(odor)
            ax.legend_.remove()
            ax.set_ylim(0.5,1)
        plt.suptitle(f"{hrv_tracker_mode_sel} no_ref train percentage effect")
        plt.tight_layout()
        # plt.show()
        fig.savefig(f"{hrv_tracker_mode_sel}_noref_train_percentage_effect.png")
        plt.close('all')

        sns.catplot(kind='swarm', data=df_plot, x='C', y='score', col='odor')
        plt.suptitle(f"{hrv_tracker_mode_sel} SVM params C selection")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{hrv_tracker_mode_sel}_SVM_params_C_selection.png")
        plt.close('all')

        sns.catplot(kind='swarm', data=df_plot, x='gamma', y='score', col='odor')
        plt.suptitle(f"{hrv_tracker_mode_sel} SVM params gamma selection")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{hrv_tracker_mode_sel}_SVM_params_gamma_selection.png")
        plt.close('all')

        fig, axs = plt.subplots(ncols=2, figsize=(17,6))
        for odor_i, odor in enumerate(['+', '-']):
            ax = axs[odor_i]
            sns.pointplot(data=df_plot.query(f"odor == '{odor}'"), x='train_percentage', y='score', hue='sujet', ax=ax)
            ax.set_title(odor)
            ax.legend_.remove()
            ax.set_ylim(0.4,1)
        plt.suptitle(f"{hrv_tracker_mode_sel} o_ref train percentage effect")
        # plt.show()
        fig.savefig(f"{hrv_tracker_mode_sel}_o_ref_train_percentage_effect.png")








################################
######## FEATURES IMPORTANCE ########
################################


def feature_importance_res():

    for hrv_tracker_mode_sel in ['2classes', '4classes']:

        os.chdir(os.path.join(path_precompute, 'predictions', 'allsujet'))
        df_features_importance = pd.read_excel(f"{hrv_tracker_mode_sel}_features_importance.xlsx")

        fig, ax = plt.subplots(figsize=(9,6))
        sns.pointplot(data=df_features_importance, x='importance_mean', y='feature', hue='odor', dodge=0.2, linestyles='', ax=ax)
        plt.title(f"{hrv_tracker_mode_sel} features importance")
        # plt.show()

        os.chdir(os.path.join(path_results, 'params'))
        fig.savefig(f"{hrv_tracker_mode_sel}_features_importance.png")
        plt.close('all')











################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':

    allsujet_hrv_tracker_mean()
    allsujet_hrv_tracker_one_classifier()
    df_allsujet_SVM_params()
    res_one_classifier_allsujet()
    feature_importance_res()



