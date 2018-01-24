#import necessary libraries
from scipy.io import wavfile #waveform reading
import matplotlib.pyplot as plt #plotting

#read in wav files
sf_44k, sound_44k = wavfile.read('010_cause_1_tk2_.wav')
sf_7k, sound_7k = wavfile.read('010_cause_1_tk2_50_7k.wav') 

#plot each sound on the same pltting area
plt.subplot(211) #sound 1
plt.title('Spectrogram of an uncompressed sound (CD - 44.1 kHz)')
plt.specgram(sound_44k,Fs=sf_44k) 
#plt.xlabel('Time')
plt.ylabel('Frequency')
plt.xticks([], [])

plt.subplot(212) #sound 2
plt.title('Spectrogram of a compressed sound (VoIP - 16 kHz)')
plt.specgram(sound_7k,Fs=sf_7k) 
plt.xlabel('Time')
plt.ylabel('Frequency')

plt.show() #plot all
