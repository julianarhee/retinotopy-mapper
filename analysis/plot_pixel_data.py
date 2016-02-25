#!/usr/bin/env python2

# From moving-bar protocol, look at individual conditions and pixels.
# Looking for peak in magnitude at the stimulation frequency.


import numpy as np
import os
# from skimage.measure import block_reduce
# from scipy.misc import imread
import cPickle as pkl
import scipy.signal
# import numpy.fft as fft
# import sys
# import optparse
from libtiff import TIFF
# from PIL import Image
import re
# import itertools
# from scipy import ndimage

import math
import matplotlib as mpl
import matplotlib.pylab as plt
# import matplotlib.cm as cm
import pandas as pd

outdir = sys.argv[1]
files = os.listdir(outdir)


files = sorted([f for f in files if '_fft' in f]) # get giant FFT file for all runs 
fname = os.path.join(outdir, files[0]) # choose particular run condition
with open(fname, 'rb') as f:
	F = pkl.load(f)

curr_file = os.path.split(fname)[1] # get name of .pkl file for particular condition
curr_cond = str.split(curr_file, '_')[2] # get specific run condition
curr_run = str.split(curr_file, '_')[3] # get run number

parenth = re.compile("\((.+)\)") # find pattern of "( xxxx )"
m = parenth.search(curr_file).group(1)
reduce_value = [int(i) for i in m[0]][0] # just use the first num 
n_x = len(set(F['ft'][0])) # number of pixels HEIGHT (164, if bin 3 and reduce=1)
n_y = len(set(F['ft'][1])) # number of pixels WIDTH (218, if bin 3 and reduce=1)
n_pixels = n_x * n_y

files = os.listdir(outdir)
files = sorted([f for f in files if '_target' in f and curr_cond in f]) # get matching run info
fname = os.path.join(outdir, files[0]) # look at first matching run condition
with open(fname, 'rb') as f:
	D = pkl.load(f)


magnitudes = [np.abs(F['ft'][2][p]) for p in range(n_pixels)]
freqs = D['freqs']


fig = plt.figure()

# for x in range(n_x):
# 	for y in range(n_y):

# GOOD: 35722 -- F['ft'][0][35722] = 163 ("x") | F['ft'][1][35722] = 188 ("y")

pidx = 0
for i in xrange(35650, 35700, 1):#range(n_pixels):

	fig.add_subplot(5, 10, pidx) 
	plt.plot(freqs[0:int(len(freqs)*.25)], magnitudes[i][0:int(len(freqs)*.25)])
	plt.plot(freqs[D['target_bin']], magnitudes[i][D['target_bin']], 'r*')
	pidx += 1