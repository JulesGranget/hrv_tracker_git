

import numpy as np
import scipy.signal

################################
######## MODULES ########
################################

# anaconda (numpy, scipy, pandas, matplotlib, glob2, joblib, xlrd)
# neurokit2 as nk
# respirationtools
# mne
# neo
# bycycle
# pingouin

################################
######## GENERAL PARAMS ######## 
################################

srate = 500


teleworking = False

enable_big_execute = False
perso_repo_computation = False

#sujet = 'DEBUG'

conditions = ['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']

sujet_list =                    np.array(['01PD','03VN','06EF','07PB','08DM','09TA',
                            '11FA','12BD','13FP','14MD','15LG','16GM','17JR','18SE','19TM','20TY','21ZV',
                            '23LF','24TJ','26MN','28NT','29SC','30AR','32CM','33MA'])

sujet_list_rev =                np.array(['PD01','VN03','EF06','PB07','DM08','TA09',
                            'FA11','BD12','FP13','MD14','LG15','GM16','JR17','SE18','TM19','TY20','ZV21',
                            'LF23','TJ24','MN26','NT28','SC29','AR30','CM32','MA33'])

# ['02MJ','27BD','10BH'] signal problems
# ['04GB', '25DF'] dypnea induction failed

sujet_best_list =               np.array(['BD12','CM32','FA11','GM16','JR17','MA33','MN26',
                            'PD01','SC29','TA09','TJ24','TM19','VN03','ZV21'])
sujet_best_list_rev =           np.array(['12BD','32CM','11FA','16GM','17JR','33MA','26MN',
                            '01PD','29SC','09TA','24TJ','19TM','03VN','21ZV'])

sujet_no_respond =              np.array(['EF06','PB07','DM08','FP13','MD14','LG15',
                            'TY20','LF23','NT28','AR30','SE18'])
sujet_no_respond_rev =          np.array(['06EF','07PB','08DM','13FP','14MD','15LG',
                            '20TY','23LF','28NT','30AR','18SE'])


odor_list = ['o', '+', '-']



################################
######## ODOR ORDER ########
################################

odor_order = {

'01PD' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   '02MJ' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '03VN' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   
'04GB' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '05LV' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '06EF' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   
'07PB' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   '08DM' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   '09TA' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'10BH' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '11FA' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   '12BD' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   
'13FP' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '14MD' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '15LG' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},
'16GM' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '17JR' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '18SE' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'19TM' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '20TY' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   '21ZV' : {'ses02' : 'o', 'ses03' : '+', 'ses04' : '-'},   
'22DI' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   '23LF' : {'ses02' : '+', 'ses03' : '-', 'ses04' : 'o'},   '24TJ' : {'ses02' : '-', 'ses03' : '+', 'ses04' : 'o'},   
'25DF' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '26MN' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '27BD' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   
'28NT' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},   '29SC' : {'ses02' : 'o', 'ses03' : '-', 'ses04' : '+'},   '30AR' : {'ses02' : '+', 'ses03' : 'o', 'ses04' : '-'},
'31HJ' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '32CM' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'},   '33MA' : {'ses02' : '-', 'ses03' : 'o', 'ses04' : '+'}
}





########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()
init_workdir = os.getcwd()

if PC_ID == 'LAPTOP-EI7OSP7K':

    try: 
        os.chdir('N:\\')
        teleworking = False
    except:
        teleworking = True 

    if teleworking:

        PC_working = 'Jules_VPN'
        path_main_workdir = 'Z:\\Projets\\HRV_Tracker\\Scripts'
        path_general = 'Z:\\Projets\\HRV_Tracker'
        path_memmap = 'Z:\\Projets\\HRV_Tracker\\memmap'
        n_core = 4

    else:

        PC_working = 'Jules_VPN'
        path_main_workdir = 'N:\\Projets\\HRV_Tracker\\Scripts'
        path_general = 'N:\\Projets\\HRV_Tracker'
        path_memmap = 'N:\\Projets\\HRV_Tracker\\memmap'
        n_core = 4


elif PC_ID == 'jules-precisiont1700':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker/Scripts'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker/memmap'
    n_core = 5

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\memmap'
    n_core = 2

elif PC_ID == 'pc-jules' or PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker/Scripts'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker/memmap'
    n_core = 4

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/memmap'
    n_core = 6

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget/EEG_Paris_J/memmap'
    n_core = 15

#### interactif node from cluster
elif PC_ID == 'node14':

    PC_working = 'node14'
    path_main_workdir = '/crnldata/cmo/Projets/HRV_Tracker/Scripts'
    path_general = '/crnldata/cmo/Projets/HRV_Tracker'
    path_memmap = '/crnldata/cmo/Projets/HRV_Tracker/memmap'
    n_core = 15

#### non interactif node from cluster
elif PC_ID == 'node13':

    PC_working = 'node13'
    path_main_workdir = '/mnt/data/julesgranget/HRV_Tracker/Scripts'
    path_general = '/mnt/data/julesgranget/HRV_Tracker'
    path_memmap = '/mnt/data/julesgranget/HRV_Tracker/memmap'
    n_core = 15

else:

    PC_working = 'node13'
    path_main_workdir = '/mnt/data/julesgranget/HRV_Tracker/Scripts'
    path_general = '/mnt/data/julesgranget/HRV_Tracker'
    path_memmap = '/mnt/data/julesgranget/HRV_Tracker/memmap'
    n_core = 15
    
path_mntdata = '/mnt/data/julesgranget/HRV_Tracker'
path_data = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_slurm = os.path.join(path_general, 'Scripts_slurm')

os.chdir(init_workdir)

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10








################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)





################################
######## HRV TRACKER ########
################################

cond_label_tracker = {'FR_CV_1' : 1, 'MECA' : 2, 'CO2' : 3, 'FR_CV_2' : 1}

train_percentage_values = [0.5, 0.6, 0.7, 0.8]


prms_tracker = {
'metric_list' : ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_S', 'HRV_COV', 'HRV_MAD', 'HRV_MEDIAN'],
'win_size_sec' : 30,
'odor_trig_n_bpm' : 75,
'jitter' : 0,
'srate' : srate
}

points_per_cond = 1000
trim_between = 100 #sec



