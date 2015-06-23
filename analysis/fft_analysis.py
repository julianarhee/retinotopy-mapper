
import os
import matplotlib.pyplot as plt
import cPickle as pkl
import numpy as np
#import pandas as pd
import time
from PIL import Image
from multiprocessing import Pool 
import glob

import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


# def get_fft(imarray):
# 	FFT = 
# 	return FFT

# Your code, but wrapped up in a function       
def convert(filename):  
    im = Image.open(filename)
    w,h = im.size
    imc = im.crop((75,0,w,h)) # left-upper, right-bottom

    return np.array(imc)


def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

pdp = pd.rolling_mean(pix, window=ncycles*fr_per_cyc)





#### SANITY CHECK ##########
'''

import numpy as np
import numpy.fft as fft
from numpy import pi
import pylab as plt

signal_frequency = 1 / 10.
sampling_rate = 60.0
duration = 80.0
amplitude = 1
noise_amplitude = 0.1
offset = 128

t = np.linspace(0, duration, int(sampling_rate * duration))
sig = offset + amplitude * np.sin(2*pi*signal_frequency*t) + noise_amplitude * np.random.randn(len(t))


mag = abs(fft.fft(sig))
print t
print 1 / sampling_rate
freqs = fft.fftfreq(len(t), 1 / sampling_rate)

plt.subplot(2, 1, 1)
plt.plot(t, sig)
plt.subplot(2, 1, 2)
plt.plot(freqs, 20 * np.log10(mag), '-')
plt.show()




l = np.where(freqs==-0.5)[0]
u = np.where(freqs==0.5)[0]

plt.subplot(3,1,3)
plt.plot(freqs[l:u], 20 * np.log10(mag)[l:u], '-')
'''


########




base_dir = '/Volumes/MAC/data/middle_depth/'
sessions = os.listdir(base_dir)

# for s in sessions:
# 	imlist = sorted(glob.glob(os.path.join(base_dir, s, '*.png')), key=natural_keys)
# 	# i = 0
# 	poolfirst = Pool()
# 	results1 = poolfirst.map(convert, imlist[0:int(len(imlist)*0.5)])
# 	stack1 = np.dstack(results1)

# 	poolsecond = Pool()
# 	results2 = poolsecond.map(convert, imlist[int(len(imlist)*0.5):len(imlist)])
# 	stack2 = np.dstack(results2)

#     stack = np.dstack([stack1, stack2])
#     del results1, results2, stack1, stack2

	# if '001' in s:
	# 	Nstack = np.dstack([stack1, stack2])
	# elif '002' in s:
	# 	Tstack = np.dstack([stack1, stack2])
	# elif '003' in s:
	# 	Ustack = np.dstack([stack1, stack2])
	# elif '004' in s:
	# 	Dstack = np.dstack([stack1, stack2])


# ncycles = 2 # n cycles to use for moving average to remove slow varying signal
# fr_per_cycle = 600 # 10 sec to run 1 cycle * frame-rate, 60 hz
# stack = Nstack

# fps = 60.
# nstim_cycles = 8.
# dur_per_cycle = 10.
# fr_per_cycle = fps*dur_per_cycle
# stim_freq = nstim_cycles/(dur_per_cycle*nstim_cycles)


signal_frequency = 1 / 10.
sampling_rate = 60.0
duration = 80.0
amplitude = 1
noise_amplitude = 10
offset = 128
nsamples = int(sampling_rate * duration)

binwidth = sampling_rate / nsamples
signal_bin = int(signal_frequency / binwidth)

t = np.linspace(0, duration, nsamples)


for s in sessions:
    imlist = sorted(glob.glob(os.path.join(base_dir, s, '*.png')), key=natural_keys)
    # i = 0
    poolfirst = Pool()
    results1 = poolfirst.map(convert, imlist[0:int(len(imlist)*0.5)])
    stack1 = np.dstack(results1)

    poolsecond = Pool()
    results2 = poolsecond.map(convert, imlist[int(len(imlist)*0.5):len(imlist)])
    stack2 = np.dstack(results2)

    stack = np.dstack([stack1, stack2])
    del results1, results2, stack1, stack2


magPix = np.zeros((stack.shape[0], stack.shape[1]))
phasePix = np.zeros((stack.shape[0], stack.shape[1]))
for i in range(0, stack.shape[0]):
    for j in range(0, stack.shape[1]):

        # # MEAN SUBTRACT?
        # pRaw = stack[i,j,:] #/255. # normalize?
        # pix = pRaw - np.mean(pRaw)
        # # pMA = movingaverage(pix, ncycles*fr_per_cyc)?
        # # pix = pRaw - pMA

        sig = stack[i,j,:]
        #sig = scipy.signal.detrend(sigr)

        # plt.subplot(211)
        # plt.plot(t[0:len(sig)], sig)
        # plt.subplot(212)
        # plt.plot(t[0:len(sig)], sigd)

        ft = np.fft.fft(sig)
        mag = np.abs(ft)
        phase = np.angle(ft)

        magPix[i,j] = 20 * np.log10(mag[signal_bin])
        phasePix[i,j] = phase[signal_bin]
plt.imshow(phasePix)

plt.colorbar()



        # print t
        # print 1 / sampling_rate
        # freqs = fft.fftfreq(len(t), 1 / sampling_rate)

        # print('At signal frequency %f: amplitude = %f dB, phase = %f radians' % (signal_frequency, 20 * np.log10(mag[signal_bin]), phase[signal_bin])) 





        y = np.fft.fft(pix) # FFT
        ps = np.abs(y)**2 # POWER
        mag = np.abs(y) # MAGNITUDE

        del freqs
        freqs = np.fft.fftfreq(len(pix), 1/60.)

        plt.plot(freqs, 20*np.log10(mag))

        # idx = np.argsort(freqs)
        # f = freqs[idx]

        # p = ps[idx]
        # log10mag = np.log10(np.abs(y))[idx]

        plt.figure()
        plt.subplot(211)
        plt.plot(f, p)
        plt.subplot(212)
        plt.plot(f, log10mag)

        # ZOOM IN:
        upper=np.where(f==0.5)[0]
        lower=np.where(f==-0.5)[0]

        plt.figure()
        plt.subplot(211)
        plt.plot(f[lower:upper], p[lower:upper]) # Zoom in POWER
        plt.subplot(212)
        plt.plot(f[lower:upper], 20*log10mag[lower:upper]) # Zoom in MAG





        maxY = np.argmax(np.abs(y))
        maxFreq = freqs[maxY]

        idx = np.argsort(freqs)
        plt.figure()
        plt.plot(freqs[idx], ps[idx])
        plt.xlabel('frequency (Hz)')
        plt.ylabel('power')

        # PLOT HALF:
        yF = np.fft.fft(pix)
        N = len(yF)
        yF = yF[0:N/2]
        fr = np.linspace(0,fs/2.,N/2.)
        plt.plot(fr,np.abs(yF)**2)
        pwr = np.abs(yF)**2
        plt.title('max power at freq ' + str(fr[np.where(pwr==pwr.max())]))

        Y = np.fft.fft(pix)
        n = len(Y);
        power = np.abs(Y[1:np.floor(n/2)])**2;
        nyquist = 1/2.;
        f = np.array(range(1,n/2))
        ff = f/(n/2.)*nyquist
        plt.figure()
        plt.plot(ff, power)
        freqidx = np.argmax(power)
        plt.title('Max Power at freq: ' + str(ff[freqidx]))


        FFT[i,j,:] = scipy.fftpack.fft(pix, n)
        plt.plot(ff, power)



    for i in range(0, len(imlist), int(chunk_size)):
		print i
		results = pool.map(convert, imlist[i:i+int(chunk_size)])
		results = pool.map(convert, imlist[i:i+int(len(imlist)*0.5)])
    	stack = np.dstack(results)

    	amplitude = np.zeros(stack.shape)
    	power = np.zeros(stack.shape)
    	phase = np.zeros(stack.shape)
    	for r in range(0, stack.shape[0]):
    		for c in range(0, stack.shape[1]):
    			pix = stack[r,c]


                mpix = pix - np.mean(pix)

    			n = len(pix)
    			y = scipy.fftpack.fft(mpix, n)

                F2 =  scipy.fftpack.fftshift(pix)
                ps2d = np.abs(F2)**2 # Calculate power spectrum
                freqs = scipy.fftpack.fftfreq(pix.size, 1./60)
                #phases = np.angle(F2)
                magnitude_spectrum = scipy.log10(ps2d)
                plt.plot(freqs, magnitude_spectrum)


                
                ps = np.abs(np.fft.fft(mpix))**2
                s = np.abs(scipy.fftpack.rfft(mpix))**2

                fs = 60.
                time_step = 1 / fs.
                freqs = np.fft.fftfreq(pix.size, time_step)
                idx = np.argsort(freqs)

                plt.plot(freqs[idx], ps[idx])


    			# tmp_phase = np.unwrap(np.abs(y)) # get phases
    			# tmp_phase = np.abs(y) # get phase
    			# phase_deg = tmp_phase*(180./np.pi) # turn into degrees...

    			amplitude[r,c,:] = np.abs(y) / n # amplitude (amp of freqs is the spectrum of wave form)
    			power[r,c,:] = (np.abs(y)**2) / n # power of the DFT
    			phase[r,c,:] = np.angle(y) * (180./np.pi) # phase in deg

    			A = A[1:-1] # Why is the first value always so big?!
    			np.where(A==A.max())


    			amplitude = np.abs(y) / n # amplitude of DFT
    			power = (np.abs(y)**2) / n # power of the DFT
				logmagnitudes = scipy.log10(power)


    			fs = 60.
    			# freqs = scipy.fftpack.fftfreq(len(pix), 1./fs)
    			rate = 60.
    			n = len(y)
    			freqs = np.linspace(0, rate/2, n)

    			plt.subplot(311)
    			plt.plot(pix)

    			plt.subplot(312)
				plt.plot(freqs, power)

				plt.subplot(313)
				plt.plot(freqs, amplitude)

    			phase = np.unwrap(np.abs(y))
    			phase_deg = phase*(180./np.pi)


    			freqs = scipy.fftpack.fftfreq(len(pix), 1./60)
    			plt.plot(freqs, logmagnitudes)


				# f = range(int(n))
				# f = [i*(60/n) for i in f] # frequency range
				nyqF = fs/2 # Nyquist frequency

				plt.subplot(211)
				plt.plot(f, ampl)
				plt.subplot(212)
				plt.plot(pix1)

				plt.plot(freqs[0:int(nyqF)], phase_deg[0:int(nyqF)])



plt.subplot(211)
plt.plot(freqs, magnitude_spectrum, 'x')




