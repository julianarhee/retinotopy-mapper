
import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys
import optparse
from libtiff import TIFF
from PIL import Image
import re
import itertools
from scipy import ndimage

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--reduce', action="store", dest="reduce_val", default="4", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="4", help="size of Gaussian kernel for smoothing")
(options, args) = parser.parse_args()

headless = options.headless
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
gsigma = int(options.gauss_kernel)
if headless:
	import matplotlib as mpl
	mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm

outdir = sys.argv[1]

###########################
files = os.listdir(outdir)

# GET BLOOD VESSEL IMAGE:
ims = [f for f in files if os.path.splitext(f)[1] == '.png']
print ims
impath = os.path.join(outdir, ims[0])
image = Image.open(impath).convert('L')
imarray = np.asarray(image)

# GET DATA STRUCT FILES:
# sessions = [f for f in flist if os.path.splitext(f)[1] != '.png']
# session_path = os.path.join(outdir, sessions[int(0)]) ## LOOP THIS

##########################

#files = os.listdir(outdir)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']
dstructs = [f for f in files if 'D_' in f and str(reduce_factor) in f]
print dstructs

D = dict()
for f in dstructs:
	outfile = os.path.join(outdir, f)
	with open(outfile,'rb') as fp:
		D[f] = pkl.load(fp)
# close


# MATCH ELEV vs. AZIM conditions:
ftmap = dict()
outshape = D[D.keys()[0]]['ft_real'].shape
for curr_key in D.keys():
	reals = D[curr_key]['ft_real'].ravel()
	imags = D[curr_key]['ft_imag'].ravel()
	ftmap[curr_key] = [complex(x[0], x[1]) for x in zip(reals, imags)]
	ftmap[curr_key] = np.reshape(np.array(ftmap[curr_key]), outshape)

V_keys = [k for k in ftmap.keys() if 'V' in k]
H_keys = [k for k in ftmap.keys() if 'H' in k]


azimuth_phase = np.angle(ftmap[V_keys[0]] / ftmap[V_keys[1]])
elevation_phase = np.angle(ftmap[H_keys[0]] / ftmap[H_keys[1]])







# freqs = D[V_keys[0]]['freqs']
# target_freq = D[V_keys[0]]['target_freq']
# target_bin = D[V_keys[0]]['target_bin']


# PLOT IT ALL:

plt.subplot(3,4,1) # GREEN LED image
plt.imshow(imarray,cmap=cm.Greys_r)

plt.subplot(3,4,2) # ABS PHASE -- elevation
fig = plt.imshow(elevation_phase, cmap="spectral")
plt.colorbar()
plt.title("elevation")

plt.subplot(3, 4, 3) # ABS PHASE -- azimuth
fig = plt.imshow(azimuth_phase, cmap="spectral")
plt.colorbar()
plt.title("azimuth")


# GET ALL RELATIVE CONDITIONS:

# PHASE:
for i,k in enumerate(H_keys): #enumerate(ftmap.keys()):
	plt.subplot(3,4,i+5)
	phase_map = np.angle(ftmap[k]) #np.angle(complex(D[k]['ft_real'], D[k]['ft_imag']))
	#plt.figure()
	fig = plt.imshow(phase_map, cmap=cm.spectral)
	plt.title(k)
	plt.colorbar()

for i,k in enumerate(V_keys): #enumerate(ftmap.keys()):
	plt.subplot(3,4,i+7)
	phase_map = np.angle(ftmap[k]) #np.angle(complex(D[k]['ft_real'], D[k]['ft_imag']))
	#plt.figure()
	fig = plt.imshow(phase_map, cmap=cm.spectral)
	plt.title(k)
	plt.colorbar()

# MAG:
for i,k in enumerate(H_keys): #enumerate(D.keys()):
	plt.subplot(3,4,i+9)
	mag_map = D[k]['mag_map']
	fig = plt.imshow(phase_map, cmap=cm.Greys_r)
	plt.title(k)
	plt.colorbar()

for i,k in enumerate(V_keys): #enumerate(D.keys()):
	plt.subplot(3,4,i+11)
	mag_map = D[k]['mag_map']
	fig = plt.imshow(phase_map, cmap=cm.Greys_r)
	plt.title(k)
	plt.colorbar()

#plt.suptitle(session_path)
sessionpath = os.path.split(outdir)[0]
plt.suptitle(sessionpath)


# SAVE FIG
outdirs = os.path.join(sessionpath, 'figures')
which_sesh = os.path.split(sessionpath)[1]
print outdirs
if not os.path.exists(outdirs):
	os.makedirs(outdirs)
imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '.svg'
plt.savefig(outdirs + '/' + imname, format='svg', dpi=1200)
print outdirs + '/' + imname
plt.show()


