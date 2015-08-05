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
target_freq = 0.1

imdir = sys.argv[1]

files = os.listdir(imdir)
files = sorted([f for f in files if os.path.splitext(f)[1] == '.png'])

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

# FIND CHUNKS:
strtidxs = []
strts = [i for i,f in enumerate(files) if '_0_' in f]


condition = os.path.split(imdir)[1]
positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
plist = list(itertools.chain.from_iterable(positions))
positions = [map(float, i.split(',')) for i in plist]

strt_idx = 0
while strt_idx < len(files):
	print strt_idx
	triple = []
	#while len(triple) < 3 and len(set(triple)) != 1:
	for i, f in enumerate(files[strt_idx:]):
		if len(tr)
		triple.append(f)
	print len(triple)
	strt_idx = i+1

if 'H-Up' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) < 0)))
if 'H-Down' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) > 0)))
if 'V-Left' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[0] for p in positions]) < 0)))
if 'V-Right' in condition:
	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[0] for p in positions]) > 0)))
idxs = [i+1 for i in find_cycs]




for fidx in range(0, len(files), 3): #[files[i] for i in strts]:
	trip = [f.split('_')[2] for f in files[fidx:fidx+3]]
	print trip
	if not len(set(trip)) == 1:
		print "FRAME COUNTS ARE OFF, starting from: ", f
		break
	else:
		for i, f in enumerate(files[idx:idx+3]):
			tiff = TIFF.open(os.path.join(imdir, f), mode='r')
			im = tiff.read_image().astype('float')
			tiff.close()
			T[:,:,i] = im

	strtidxs.append(int(idx))


print('copying files')

for i, f in enumerate(files):

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
