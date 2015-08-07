import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import matplotlib.pylab as plt
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
from libtiff import TIFF

sampling_rate = 60.0
reduce_factor = (4, 4)
cache_file = True
target_freq = 0.08

imdir = sys.argv[1]

files = os.listdir(imdir)
files = sorted([f for f in files if os.path.splitext(f)[1] == '.tif'])

#files = sorted([f for f in files if os.path.isfiles(os.path.join(imdir, f))])
#files = files[11981:-1]
# files = files[0:100]
print files[0]
#sample = imread(os.path.join(imdir, files[0]))
tiff = TIFF.open((os.path.join(imdir, files[0])), mode='r')
sample = tiff.read_image().astype('float')
print sample.dtype, [sample.max(), sample.min()]
tiff.close()
sample = block_reduce(sample, reduce_factor)
# print sample.dtype
# plt.imshow(sample)
# plt.show()

cos_im = np.zeros((sample.shape[0], sample.shape[1]), dtype='float')
sin_im = np.zeros((sample.shape[0], sample.shape[1]), dtype='float')
len_im = np.zeros((sample.shape[0], sample.shape[1]), dtype='float')

n_images = len(files)

t = np.arange(0, n_images/sampling_rate, 1.0/sampling_rate)

sin_ref = np.sin(2 * np.pi * t * target_freq )
cos_ref = np.cos(2 * np.pi * t * target_freq )

# plt.hold(True)
# plt.plot(t, sin_ref)
# plt.plot(t, cos_ref)
# plt.show()

# sin_ref /= np.sqrt(np.sum(sin_ref**2))
# cos_ref /= np.sqrt(np.sum(cos_ref**2))

# plt.hold(True)
# plt.plot(t, sin_ref)
# plt.plot(t, cos_ref)
# plt.show()

print('copying files')

# tiff = TIFF.open((os.path.join(imdir, files[0])), mode='r')
# sample = tiff.read_image().astype('float')

#ref_im = imread(os.path.join(imdir, files[n_images/2])).astype('float')
tiff = TIFF.open(os.path.join(imdir, files[n_images/2]), mode='r')
ref_im = tiff.read_image().astype('float')
tiff.close()

ref_im_reduced = block_reduce(ref_im, reduce_factor)

ref_im_reduced -= np.mean(ref_im_reduced.ravel())

for i, f in enumerate(sorted(files)):

	# if i > 20:
	# 	break

	if i % 100 == 0:
		print('%d images processed...' % i)
	#im = imread(os.path.join(imdir, f)).astype('float')
	tiff = TIFF.open(os.path.join(imdir, f), mode='r')
	im = tiff.read_image().astype('float')
	tiff.close()

	im_reduced = block_reduce(im, reduce_factor)

	im_reduced -= np.mean(im_reduced.ravel())

	im_reduced -= ref_im_reduced


	len_im += im_reduced**2
	cos_im += cos_ref[i] * im_reduced
	sin_im += sin_ref[i] * im_reduced



norm_im = np.sqrt(len_im)

# cos_im /= norm_im
# sin_im /= norm_im

print cos_im

mag_map = np.sqrt(cos_im**2 + sin_im**2)
phase_map = np.arctan2(cos_im, sin_im)


plt.subplot(2, 2, 1)
plot = plt.imshow(cos_im)
plt.colorbar()

plt.subplot(2, 2, 2)
plot = plt.imshow(sin_im)
plt.colorbar()


plt.subplot(2, 2, 3)
plot = plt.imshow(mag_map)
plot.set_cmap('hot')
plt.colorbar()

plt.subplot(2, 2, 4)
plot = plt.imshow(phase_map)
plot.set_cmap('spectral')
plt.colorbar()


figdir = os.path.join(os.path.split(imdir)[0], 'figures')
if not os.path.exists(figdir):
	os.makedirs(figdir)
imname = '/' + os.path.split(imdir)[1] + '_' + str(target_freq) + '.png'
plt.savefig(figdir + imname)

plt.show()
