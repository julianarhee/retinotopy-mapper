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
from PIL import Image

import warnings


imdir = sys.argv[1]
outdir = os.path.join(os.path.split(imdir)[0], 'output', os.path.split(imdir)[1])
if not os.path.exists(outdir):
	try:
		os.mkdir(outdir)
	except OSError:
		os.mkdir(os.path.split(outdir)[0])
		os.mkdir(outdir)

files = os.listdir(imdir)
files = sorted([f for f in files if os.path.splitext(f)[1] == '.tif'])
print len(files)

#files = sorted([f for f in files if os.path.isfiles(os.path.join(imdir, f))])
#files = files[11981:-1]
# files = files[0:100]
print files[0]
#sample = imread(os.path.join(imdir, files[0]))
tiff = TIFF.open((os.path.join(imdir, files[0])), mode='r')
sample = tiff.read_image().astype('float')
print sample.dtype, [sample.max(), sample.min()]
tiff.close()
#sample = block_reduce(sample, reduce_factor)

# FIND cycle chunks:
strts = [(i,f) for i,f in enumerate(files) if f.split('_')[4]=='0']
cycidxs = [s[0] for s in strts]

curridx = 0
nframes = 0
nfiles = []
for i, idx in enumerate(sorted(cycidxs)):
	print idx

	if i == len(cycidxs)-1:
		print "hi"
		frames = [int(f.split('_')[2]) for f in files[idx:]]
		curr_files = files[idx:]
	else:
		frames = [int(f.split('_')[2]) for f in files[idx:cycidxs[i+1]]]
		curr_files = files[idx:cycidxs[i+1]]
	#print frames

	find_consecs = np.where(np.diff(frames)==1)[0] + 1
	#print len(find_consecs)

	curridx = 0
	for ci, cidx in enumerate(sorted(find_consecs)):
		#print "[", curridx, ',', cidx, ']'
		if ci % 100 == 0:
			print('%d images processed...' % ci)

		fns = curr_files[curridx:cidx]
		nfiles.append(len(fns))
		#print len(fns)

		T = np.empty((sample.shape[0], sample.shape[1], len(fns)))
		for fi, fn in enumerate(fns):
			#print fi
			tiff = TIFF.open(os.path.join(imdir, fn), mode='r')
			im = tiff.read_image().astype('float')
			tiff.close()
			T[:,:,fi] = im

		#lastaxis = int(len(fns)-1)
		#print T.shape[2]

		avg_frame = np.mean(T, axis=2, dtype=np.float64)

		# with warnings.catch_warnings():
		# 	warnings.filterwarnings('error')
		# 	try:
		# 		avg_frame = np.mean(T, axis=2, dtype=np.float64)
		# 	except RuntimeWarning:
		# 		print T.shape
		# 		print len(fns), cidx
		# 		avg_frame = T

		fname = os.path.join(outdir, fn)
		tiff = TIFF.open(fname, mode='w') # use last fn in triplet to save
		tiff.write_image(avg_frame)
		tiff.close()
		#print curridx
		curridx = cidx
		nframes += 1
		#print fname