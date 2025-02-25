import numpy as np
import os
# import skimage
# from skimage.measure import block_reduce
from scipy.misc import imread
import matplotlib.pylab as plt
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys

sampling_rate = 60.0
reduce_factor = (4, 4)
cache_file = True
target_freq = 0.1
binspread = 0

imdir = sys.argv[1]

files = os.listdir(imdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']

# sample = imread(os.path.join(imdir, files[0]))
fn = os.path.join(imdir, files[0])
with open(fn, 'rb') as f:
    F = pkl.load(f)
sample = F['im']
f.close()

# sample = block_reduce(sample, reduce_factor)

# plt.imshow(sample)
# plt.show()

stack = np.empty((sample.shape[0], sample.shape[1], len(files)))

print('copying files')


for i, f in enumerate(files):

    # if i > 20:
    #   break

    if i % 100 == 0:
        print('%d images processed...' % i)

    # im = imread(os.path.join(imdir, f)).astype('float')
    fn = os.path.join(imdir, f)
    with open(fn, 'rb') as fp:
        F = pkl.load(fp)
    im = F['im']
    fp.close()

    # im_reduced = block_reduce(im, reduce_factor)
    im_reduced = im

    stack[:,:,i] = im_reduced


mag_map = np.empty(sample.shape)
phase_map = np.empty(sample.shape)

for x in range(sample.shape[0]):
    print x
    
    for y in range(sample.shape[1]):

        sig = stack[x, y, :]

        sig = scipy.signal.detrend(sig)

        ft = fft.fft(sig)
        mag = abs(ft)
        phase = np.angle(ft)


        freqs = fft.fftfreq(len(sig), 1 / sampling_rate)
        binwidth = freqs[1] - freqs[0]

        target_bin = int(target_freq / binwidth)

        # if x % int(sample.shape[0] / 4) == 0 or y % int(sample.shape[1] / 4) == 0:
        #   plt.subplot(2,1,1)
        #   plt.plot(freqs, mag, '*')
        #   plt.subplot(2,1,2)
        #   plt.plot(freqs, phase, '*')
        #   plt.show()

        if binspread != 0:
            mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
            phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
        else:
            mag_map[x, y] = 20*np.log10(mag[target_bin])
            phase_map[x, y] = phase[target_bin]


plt.subplot(2, 1, 1)
plt.imshow(mag_map)
plt.colorbar()
plt.subplot(2, 1, 2)
fig = plt.imshow(phase_map)
plt.colorbar()
plt.show()


