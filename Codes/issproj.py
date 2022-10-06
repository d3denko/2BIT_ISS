import numpy as np
import matplotlib.pyplot as plt
from numpy.matrixlib import matrix
import scipy
from scipy.signal import spectrogram, find_peaks, buttord, butter, lfilter, tf2zpk, freqz
import soundfile as sf

def custom_DFT(frame):
    N_size = 1024
    range_1024 = np.arange(0, N_size)
    const = -1 * 1j * (2*np.pi/N_size)

    b_matrix = []
    b_matrix = np.e ** (const * range_1024) # e ** -j* 2pi/1024 * k * n
    b_matrix = np.array([b_matrix])

    b_matrix = b_matrix.T ** range_1024

    dft = b_matrix.dot(frame.T)
    return dft


#1 - Nacitanie vstupneho signalu, jeho dlzka vo vzorkoch a sekundach,
#    maximalna a minimalna hodnota

filename = "/home/d3denk0/Desktop/ISS/xhoril01.wav"
y, fs = sf.read(filename)


filename = "/home/d3denk0/Desktop/ISS/useful_info.txt"
file = open(filename,"w+")

# Dlzka nahravky v s
t = np.arange(y.size)/fs

file.write("~~~~~~~~~~~~~ ZAKLADNE INFO O SIGNALE ~~~~~~~~~~~~~\n")
file.write("- Dlzka signalu vo vzorkoch: " +str(len(y)) + "\n")
file.write("- Dlzka signalu v sekundach: " +str(len(y)/fs)+ " s\n")
file.write("- Maximalna hodnota signalu: "+ str(np.max(y)) + "\n")
file.write("- Minimalna hodnota signalu: "+ str(np.min(y)) + "\n")

plt.plot(t,y)
plt.title("Povodny signal")
plt.xlabel("t[s]")
plt.show()

# Upravovany signal
s = [] 

#2 - Ustrednenie signalu, normalizovanie do dynamickeho 
#    rozsahu, signal rozdelte na ramce a ulozte do matice,
#    vyberte "pekny" znely ramec a zobrazte ho

#Ustrednenie
for val in y:
    s.append(val - np.mean(y))

#Normalizovanie
abs_koef = np.max(np.abs(s))
for i in range(len(s)):
    s[i] = s[i]/abs_koef

#Ramce
lower_border = 0
higher_border = 1024
tmp_arr = []

while(higher_border < len(s)):
    tmp_arr = np.append(tmp_arr,s[lower_border:higher_border])
    higher_border+=512
    lower_border+= 512

last_frame = []
last_frame = np.append(last_frame, s[lower_border: len(s)])
for i in range(len(last_frame), 1024):
    last_frame = np.append(last_frame, [0])

fin_arr = np.concatenate((tmp_arr, last_frame))

num_column = int (len(fin_arr)/1024)
matrix = np.reshape(fin_arr, (num_column, 1024))

matrix=matrix.transpose()

_,ax = plt.subplots(2,1,figsize=(15,7.5))

chosen_frame = 33
frame_t= np.arange(512*chosen_frame, 512*(chosen_frame+2))/fs
ax[0].plot(frame_t, matrix.T[chosen_frame])
label = 'Ramec ' + str(chosen_frame)
ax[0].set_title(label)
ax[0].set_xlabel('t [s]')


#3 - Vypocet DFT pre 1024 vzorkov

#for i in range(num_column):
dft = custom_DFT(matrix.T[chosen_frame])
dft = np.abs(dft)

compare = np.fft.fft(matrix.T[chosen_frame])
compare = np.abs(compare)

file.write("\n~~~~~~~~~~~~~ POROVNANIE DFT ~~~~~~~~~~~~~ \n")
file.write("DFT su rovnake: "+str(np.allclose(compare, dft)) +"\n")

label = 'DFT '+str(chosen_frame)
ax[1].set_title(label)
ax[1].set_xlabel('f [HZ]')
ax[1].plot(((np.arange(1024 / 2)/1024) * fs), dft[np.arange(len(dft)//2)])

plt.tight_layout()
plt.show()


#4 - Spektrogram
s = np.array(s)
freq, time, spd = spectrogram(s, fs, nperseg=1024, noverlap=512)
spd = 10 * np.log10(spd + 1e-20)
plt.figure(figsize=(15,5))
plt.pcolormesh(time,freq, spd)
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.title('Spektrogram')
colorbar = plt.colorbar()
colorbar.set_label('Spektralna hustota vykonu [dB]', rotation = 270, labelpad = 15)
plt.show()

#5 - Urcenie rusivych frekvencii
noises_dft = custom_DFT(matrix.T[0])
noises_dft = np.abs(noises_dft)

interval = np.array(noises_dft[0:(noises_dft.size)//2])
interval = interval*1024

freq_indexes,_ = find_peaks(interval, height = 16000)

freq_array = freq_indexes/1024*fs

f4 = sorted(freq_array,reverse=True)[0]
f3 = sorted(freq_array,reverse=True)[1]
f2 = sorted(freq_array,reverse=True)[2]
f1 = sorted(freq_array,reverse=True)[3]

koef1 = f2/f1
koef2 = f3/f1
koef3 = f4/f1
approx = 15.625/f1

file.write("\n~~~~~~~~~~~~~ ZAVISLOST RUSIVYCH FREKVENCII ~~~~~~~~~~~~~ \n")
if ((koef2-koef1) > 1-approx) and ((koef2-koef1) < 1+approx):
    if ((koef3-koef2) > 1-approx) and ((koef3-koef2) < 1+approx):
        file.write("Frekvencie su nasobkami najmensej frekvencie\n")
else:
    file.write("Frekvencie nie su nasobkami najmensej frekvencie\n")

#6 - Generovanie signalu - 4 cosinusovky
c1 = np.cos(2*np.pi* f1 * (np.arange(s.size) / fs))
c2 = np.cos(2*np.pi* f2 * (np.arange(s.size) / fs))
c3 = np.cos(2*np.pi* f3 * (np.arange(s.size) / fs))
c4 = np.cos(2*np.pi* f4 * (np.arange(s.size) / fs))

rslt_cos = c1 +c2 +c3 + c4

filename = "/home/d3denk0/Desktop/ISS/4cos.wav"
sf.write(filename,rslt_cos.astype(np.float32),fs)

freq, time, spd = spectrogram(rslt_cos, fs)
spd = 10 * np.log10(spd + 1e-20)

plt.figure(figsize=(15,5))
plt.pcolormesh(time,freq, spd)
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.title('Spektrogram cosinusoviek')
colorbar = plt.colorbar()
colorbar.set_label('Spektralna hustota vykonu [dB]', rotation = 270, labelpad = 15)
plt.show()

#7 - Cistiace filtry
nyquist_freq = fs/2
filter_num = 1
file.write("\n~~~~~~~~~~~~~ FILTRE ~~~~~~~~~~~~~ \n")

for freq in freq_array:
    ord, wn = buttord([(freq-50)/nyquist_freq, (freq+50)/nyquist_freq], [(freq-15)/nyquist_freq, (freq+15)/nyquist_freq],3,40)
    numerator,denominator = butter(ord,wn,btype='bandstop')

    # impulzna odozva
    n_imp = 32
    imp = [1, *np.zeros(n_imp-1)]
    dig_filter = lfilter(numerator, denominator, imp)

    _,ax = plt.subplots(2,2,figsize=(15,10))

    ax[0][0].stem(np.arange(n_imp), dig_filter, basefmt=' ')
    ax[0][0].set_xlabel('n')
    ax[0][0].set_title('Impulzna odozva h[n] '+ str(filter_num) + '. filtru')

    #8 - Nuly a poly
    zeros, poles, k = tf2zpk(numerator, denominator)

    # stabilita filtru
    is_stable = (poles.size == 0) or np.all(np.abs(poles) < 1)
    file.write("Filter " + str(filter_num) + " je stabilny:" +str(is_stable)+ "\n")

    # jednotkova kruznica
    circle = np.linspace(0, 2*np.pi,100)
    ax[0][1].plot(np.cos(circle), np.sin(circle))

    ax[0][1].scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='r', label='nuly')
    ax[0][1].scatter(np.real(poles), np.imag(poles), marker='x', color='g', label='poly')

    ax[0][1].set_title('Nuly a poly '+ str(filter_num) + '. filtru')
    ax[0][1].set_xlabel('Realna zlozka {R}')
    ax[0][1].set_ylabel('Imaginarna zlozka {I}')
    ax[0][1].legend(loc='lower left')

    #9 - Frekvencna charakteristika
    response_freqz, freq_response = freqz(numerator,denominator)

    ax[1][0].plot(response_freqz/ 2 /np.pi * fs, np.abs(freq_response))
    ax[1][0].set_xlabel('f [Hz]')
    ax[1][0].set_title('Modul frekvencnej charakteristiky $|H(e^{j\omega})|$ ' + str(filter_num) + '. filtru')

    ax[1][1].plot(response_freqz/ 2 /np.pi * fs, np.angle(freq_response))
    ax[1][1].set_xlabel('f [Hz]')
    ax[1][1].set_title('Argument frekvencnej charakteristiky $\mathrm{arg}\ H(e^{j\omega})$ ' + str(filter_num) + '. filtru')

    for i,j in ax:
        i.grid(alpha=0.5, linestyle='--')
        j.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    #10 - Filtracia 
    s = lfilter(numerator, denominator, s)

    filter_num+=1


#10 Filtracia - normalizovanie
abs_koef = np.max(np.abs(s))
final_signal = []
for val in s:
    final_signal.append(val/abs_koef)

final_signal = np.array(final_signal)
freq, time, spd = spectrogram(final_signal, fs)
spd = 10 * np.log10(spd + 1e-20)

plt.figure(figsize=(15,5))
plt.pcolormesh(time,freq, spd)
plt.ylabel('f [Hz]')
plt.xlabel('t [s]')
plt.title('Vysledny spektrogram')
colorbar = plt.colorbar()
colorbar.set_label('Spektralna hustota vykonu [dB]', rotation = 270, labelpad = 15)
plt.show()

filename = "/home/d3denk0/Desktop/ISS/clean_bandstop.wav"
sf.write(filename, final_signal, fs)
file.close()