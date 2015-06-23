import numpy as np, h5py 

import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import matplotlib.pylab as plt
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys

sampling_rate = 60.0
reduce_factor = (1, 1)
cache_file = True
target_freq = 1/24. #0.1
binspread = 0

# imdir = sys.argv[1]

# files = os.listdir(imdir)
# files = [f for f in files if os.path.splitext(f)[1] == '.png']

# files = files[0:int(round(len(files)*0.5))]

# sample = imread(os.path.join(imdir, files[0]))
# sample = sample[30:-1, 50:310]
# print "FIRST", sample.dtype
# sample = block_reduce(sample, reduce_factor)

# plt.figure()
# plt.imshow(sample)
# plt.show()

# datadir = '/Volumes/MAC/Julie'
f = h5py.File('/share/scratch2/julianarhee/brains1.mat','r') 
data = f.get('brains')
# data = np.array(data)
sample = data[0]



# stack = np.empty((data.shape[1], data.shape[2], data.shape[0]))
# print len(files)

# print('copying files')


# for i, f in enumerate(files):

# 	# if i > 20:
# 	# 	break

# 	if i % 100 == 0:
# 		print('%d images processed...' % i)
# 	im = imread(os.path.join(imdir, f)).astype('float')
# 	# print im.shape
# 	im = im[30:-1, 50:310]
# 	# print im.shape
# 	im_reduced = block_reduce(im, reduce_factor)
# 	# im_reduced = im
# 	stack[:,:,i] = im_reduced

s = data[0].ravel()

mag_map = np.empty((data.shape[0]))
phase_map = np.empty((data.shape[0]))

flat = np.empty((len(s), data.shape[0]))
for i in range(data.shape[0]):
	print i
	sig = data[i].ravel()
	flat[:,i] = sig

	# ft = scipy.fftpack.fft(sig, axis=1)
	ft = fft.fft(sig)
	mag = abs(ft)
	phase = np.angle(ft)
	freqs = fft.fftfreq(len(sig), 1 / sampling_rate)
	binwidth = freqs[1] - freqs[0]
	target_bin = int(target_freq / binwidth)

	if binspread != 0:
		mag_map[i] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
		phase_map[i] = np.mean(phase[target_bin-binspread:target_bin+binspread])
	else:
		mag_map[i] = 20*np.log10(mag[target_bin])
		phase_map[i] = phase[target_bin]

mag_map.reshape()


mag_map = np.empty(sample.shape)
phase_map = np.empty(sample.shape)

for x in range(sample.shape[0]):
	print x
	for y in range(sample.shape[1]):

		# sig = stack[x, y, :]
		sig = data[i].ravel()

		# sig = scipy.signal.detrend(sig)

		ft = fft.fft(sig)
		mag = abs(ft)
		phase = np.angle(ft)


		freqs = fft.fftfreq(len(sig), 1 / sampling_rate)
		binwidth = freqs[1] - freqs[0]

		target_bin = int(target_freq / binwidth)

		if binspread != 0:
			mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
			phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
		else:
			mag_map[x, y] = 20*np.log10(mag[target_bin])
			phase_map[x, y] = phase[target_bin]


# plt.subplot(2, 1, 1)
# plt.imshow(mag_map)
# plt.colorbar()
# plt.subplot(2, 1, 2)
fn = 'maps.pkl'
with open(fn, 'wb') as f:
	pkl.dump(phase_map, fn)

plt.figure()
fig = plt.imshow(phase_map)
plt.colorbar()
plt.show() 
savefig('phase.png')
# fig.set_cmap("spectral")

