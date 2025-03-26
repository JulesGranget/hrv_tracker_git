import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter
from bycycle.cyclepoints import find_extrema
from HRV_TRACKER_stage_ecg_clean import *

ppg_datas = []
ecg_datas = []

ppgtest = 'MAX86178_20230613_151101.ppg.csv'   
#1 :MAX86178_20230613_144835.ppg.csv bouge un peu
#2 :MAX86178_20230613_150706.ppg.csv bouge un peu
#3 :MAX86178_20230613_151101.ppg.csv assez clean
#4 :MAX86178_20230613_151156.ppg.csv assez clean
#5 :MAX86178_20230613_151449.ppg.csv assez clean

ecgtest = 'MAX86178_20230616_114919.ecg.csv'   
#1 :MAX86178_20230616_114919.ecg.csv
#2 :MAX86178_20230616_115550.ecg.csv

with open(ppgtest, 'r') as file:
    reader = csv.reader(file)
    debut = False
    for row in reader:

        if row[0]=='stop time': #mettre 0 si on prend directement le csv généré sans le mettre dans le kst ou 1 si on le met dans le kst
            break

        if debut == True:
            ppg_datas.append(float(row[7])) # pour le 1er PPG et aussi faut récupérer le timestamp pour le temps
        
        elif row[0] == 'timestamp': #mettre 0 si on prend directement le csv généré sans le mettre dans le kst ou 1 si on le met dans le kst
            debut = True

with open(ecgtest, 'r') as file1:
    reader1 = csv.reader(file1)
    debut = False
    for row1 in reader1:

        if row1[0]=='stop time': #mettre 0 si on prend directement le csv généré sans le mettre dans le kst ou 1 si on le met dans le kst
            break

        if debut == True:
            ecg_datas.append(float(row1[2])) # pour le 1er PPG et aussi faut récupérer le timestamp pour le temps
        
        elif row1[0] == 'timestamp': #mettre 0 si on prend directement le csv généré sans le mettre dans le kst ou 1 si on le met dans le kst
            debut = True

ecg_datas = -np.array(ecg_datas)
np.save('ecg_datas.npy', ecg_datas)
temps_ppg=[]
temps_ecg=[]
for i in range(len(ppg_datas)):
    temps_ppg.append(i/25) #25 car 25Hz pour mesure ppg

for i in range(len(ecg_datas)):
    temps_ecg.append(i/125) #125 car 125Hz pour mesure ecg et mettre 500Hz pour 15H test si je le fais 

'''plt.plot(temps_ppg,ppg_datas)

plt.show()'''

'''plt.plot(temps_ecg,ecg_datas)

plt.show()'''

def get_peaks_from_ppg(sig, srate, btype='bandpass', ftype='bessel', order=5, debug=False):

    """"
    sig : np.array
    """
    sig=np.array(sig)

    #### zscore
    sig = (sig - sig.mean()) / sig.std()

    #### identify heart rate for filtering
    nwind = int(5*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx = scipy.signal.welch(sig, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)

    
    '''plt.plot(hzPxx, Pxx)
    plt.show()'''

    hzPxx_i=0
    hzPxx_valeur=0
    while hzPxx_valeur<0.5:
        hzPxx_valeur=hzPxx[hzPxx_i]
        hzPxx_i=hzPxx_i+1

    #low_cut_off= np.where(hzPxx<0.5)[0]
    low_cut_off = Pxx[hzPxx_i:].argmax() + hzPxx_i
    #print(low_cut_off)

    band = [hzPxx[low_cut_off] - 0.5, hzPxx[low_cut_off] + 0.5] 
    #print(band)
    #print(Pxx[hzPxx_i:].argmax())

    #### filter signal
    if np.isscalar(band):
        Wn = band / srate * 2
    else:
        Wn = [e / srate * 2 for e in band]

    #print(Wn)
    filter_coeff = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output='sos')

    sig_clean = scipy.signal.sosfiltfilt(filter_coeff, sig, axis=0)

    #### get peaks
    peaks, troughs = find_extrema(sig_clean, srate, band)

    
    '''plt.plot(sig, label='raw')
    plt.plot(sig_clean, label='clean')
    plt.vlines(peaks, ymin=sig_clean.min(), ymax=sig_clean.max(), colors='r')
    plt.legend()
    plt.show()'''

    peaks_sec = peaks / srate
    RRI = np.diff(peaks_sec)

    return RRI

def ecg_peaks(sig_physio,srate):

    sig_physio, ecg_peaks = compute_ecg(sig_physio, srate)
    cR_val = ecg_peaks/srate

    RRI = np.diff(cR_val)

    return RRI

def label_vec(signal,srate) :
    temps=[]
    label_vec=[]
    for i in range(len(signal)): # len(signal) = nombre de points = 25*temps_en_s
        temps.append(i/srate)
    
    temps_total=len(signal)/srate # temps total en s du signal

    for i in range(temps_total):
        if 0<i<60:
            label_vec.append(0)
        elif 60<=i<120*5:
            label_vec.append(0)
        elif 120*5<=i<120*6:
            label_vec.append(0)
        elif 120*6<=i<120*11:
            label_vec.append(1)


#label_vec_ppg = get_peaks_from_ppg(ppg_datas,25) # attention mesure à 25Hz
#label_vec_ecg = ecg_peaks(ecg_datas,125) # attention mesure à 125Hz
ppg_sec_diff = get_peaks_from_ppg(ppg_datas,25)
print(ppg_sec_diff)
ecg_sec_diff = ecg_peaks(ecg_datas,125)
print(ecg_sec_diff)