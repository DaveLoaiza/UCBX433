# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:11:21 2017

Homework # 4

David Loaiza -- dave.loaiza@gmail.com

Python For Data Analysis And Scientific Computing
UC Berkeley Extension

Generate the English phonemes corresponding to the vowels: a, e, i, o, u





6. Organize your code: use each line from this HW as a comment line before coding each step
7. Save these steps in a .py file and email it to me before next class. I will run it!
"""

from scipy.io.wavfile import read, write
from scipy import arange, sin, pi
import sounddevice as sd

#1. Synthesize each formant tone in preparation for creating each vowel
#We will simplify the synthesis of each vowel: a, e, i, o, u accepting the following constraints:
#   1. you will use pure tones only (serving as center frequencies) using the sin function
#   2. you will manually adjust the amplitude of each formant frequency you generate
    #3. you will use the first three formants for the average male speaker
#2. Create a function that generates each simple formant tone with sampling frequency Fs=8kHz,
#duration 2 seconds, and different amplitudes based on the spectrograms of the American
#English Vowels provided by Ladeforged 2006:185-187. Refer to the spectrogram provided in the
#lectures and use your judgment. Obtain best results by testing and repetition!
#3. Use the formant frequencies given by J.C. Wells - refer to the table provided in the lectures

Fs = 8000
duration = 2
x = arange(Fs*duration)

#a_f1 = 7*sin(2*pi*x*(740/Fs))
#a_f2 = 4*sin(2*pi*x*(1180/Fs))
#a_f3 = 2*sin(2*pi*x*(2640/Fs))
#
#e_f1 = 6*sin(2*pi*x*(360/Fs))
#e_f2 = 2*sin(2*pi*x*(2220/Fs))
#e_f3 = 2*sin(2*pi*x*(2960/Fs))
#
#i_f1 = 4*sin(2*pi*x*(600/Fs))
#i_f2 = 2*sin(2*pi*x*(2060/Fs))
#i_f3 = 1*sin(2*pi*x*(2840/Fs))
#
#o_f1 = 5*sin(2*pi*x*(380/Fs))
#o_f2 = 3*sin(2*pi*x*(940/Fs))
#o_f3 = 0.6*sin(2*pi*x*(2300/Fs))
#
#u_f1 = 4*sin(2*pi*x*(320/Fs))
#u_f2 = 2*sin(2*pi*x*(920/Fs))
#u_f3 = 1*sin(2*pi*x*(2200/Fs))
#

freq = [(740,1180,2640),(360,2220,2960),(600,2060,2840),(380,940,2300),(320,920,2200)]

def formants(f):
    f1, f2, f3 = f
    signal1 = 10*sin(2*pi*x*(f1/Fs))
    signal2 = 5*sin(2*pi*x*(f2/Fs))
    signal3 = 1*sin(2*pi*x*(f3/Fs))
    signal = signal1 + signal2 + signal3
    return signal


#4. Simply add all three formants for each vowel and save the files in .wav format
vowel_1 = formants(freq[0])
vowel_2 = formants(freq[1])
vowel_3 = formants(freq[2])
vowel_4 = formants(freq[3])
vowel_5 = formants(freq[4])

write('vowel1.wav', Fs, vowel_1)
write('vowel2.wav', Fs, vowel_2)
write('vowel3.wav', Fs, vowel_3)
write('vowel4.wav', Fs, vowel_4)
write('vowel5.wav', Fs, vowel_5)

#5. Extra point (optional): you can incorporate the code I provided in the previous slide as a
#module.py to plot each vowel sound. In this case send me both files.
