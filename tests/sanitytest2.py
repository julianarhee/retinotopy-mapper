import numpy as np
import numpy.fft as fft
from numpy import pi
import pylab as plt

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


sig = offset + amplitude * np.sin(2*pi*signal_frequency*t) + noise_amplitude * np.random.randn(len(t))
sig = sig.astype(np.int)

ft = fft.fft(sig)
mag = abs(ft)
phase = np.angle(ft)

print t
print 1 / sampling_rate
freqs = fft.fftfreq(len(t), 1 / sampling_rate)


print('At signal frequency %f: amplitude = %f dB, phase = %f radians' % (signal_frequency, 20 * np.log10(mag[signal_bin]), phase[signal_bin])) 

plt.subplot(3, 1, 1)
plt.plot(t, sig)
plt.subplot(3, 1, 2)
plt.plot(freqs, 20 * np.log10(mag), '-')
plt.subplot(3, 1, 3)
plt.plot(freqs, phase)
plt.show()