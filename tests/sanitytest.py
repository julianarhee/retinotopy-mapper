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

t = np.linspace(0, duration, int(sampling_rate * duration))
sig = offset + amplitude * np.sin(2*pi*signal_frequency*t) + noise_amplitude * np.random.randn(len(t))
sig = sig.astype(np.int)



mag = abs(fft.fft(sig))
print t
print 1 / sampling_rate
freqs = fft.fftfreq(len(t), 1 / sampling_rate)

plt.subplot(2, 1, 1)
plt.plot(t, sig)
plt.subplot(2, 1, 2)
plt.plot(freqs, 20 * np.log10(mag), '*')
plt.show()