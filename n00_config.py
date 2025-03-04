

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




########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'jules-precisiont1700':

    PC_working = 'jules-precisiont1700'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker/Scripts'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/HRV_Tracker/Mmap'
    n_core = 6

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/Projets/HRV_Tracker/Scripts'
    path_general = '/crnldata/cmo/Projets/HRV_Tracker'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 15

path_data = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_respfeatures = os.path.join(path_general, 'Analyses', 'results') 





################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)




