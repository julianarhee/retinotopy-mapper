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
import pandas as pd


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

parser = optparse.OptionParser()
parser.add_option('--headless', action="store_true", dest="headless", default=False, help="run in headless mode, no figs")
parser.add_option('--freq', action="store", dest="target_freq", default="0.05", help="stimulation frequency")
parser.add_option('--reduce', action="store", dest="reduce_val", default="2", help="block_reduce value")
parser.add_option('--sigma', action="store", dest="gauss_kernel", default="0", help="size of Gaussian kernel for smoothing")
parser.add_option('--format', action="store", dest="im_format", default="tif", help="saved image format")
parser.add_option('--fps', action="store", dest="sampling_rate", default="60.", help="camera acquisition rate (fps)")

(options, args) = parser.parse_args()

imdir = sys.argv[1]
#imdirs = [sys.argv[1], sys.argv[2]]

im_format = '.'+options.im_format
target_freq = float(options.target_freq)
reduce_factor = (int(options.reduce_val), int(options.reduce_val))
if reduce_factor[0] > 0:
	reduceit=1
else:
	reduceit=0
gsigma = int(options.gauss_kernel)
headless = options.headless
if headless:
	import matplotlib as mpl
	mpl.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm

sampling_rate = float(options.sampling_rate) #60. #np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
cache_file = True
cycle_dur = 1./target_freq #10.
binspread = 0

#stacks = dict()
#for imdir in imdirs:
basepath = os.path.split(os.path.split(imdir)[0])[0]
session = os.path.split(os.path.split(imdir)[0])[1]
cond = os.path.split(imdir)[1]

files = os.listdir(imdir)
print len(files)
files = sorted([f for f in files if os.path.splitext(f)[1] == str(im_format)])
print len(files)

tiff = TIFF.open(os.path.join(imdir, files[0]), mode='r')
sample = tiff.read_image().astype('float')
print "sample type: %s, range: %s" % (sample.dtype, str([sample.max(), sample.min()]))
print "sample shape: %s" % str(sample.shape)
tiff.close()

# FIND CYCLE STARTS:
# positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
# plist = list(itertools.chain.from_iterable(positions))
# positions = [map(float, i.split(',')) for i in plist]
# if 'H-Up' in cond:
# 	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) < 0)))
# if 'H-Down' in cond:
# 	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) > 0)))
# if 'V-Left' in cond:
# 	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[0] for p in positions]) < 0)))
# if 'V-Right' in cond:
# 	find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[0] for p in positions]) > 0)))
# idxs = [i+1 for i in find_cycs]
# idxs.append(0); idxs.append(len(positions))
# idxs = sorted(idxs)
# nframes_per_cycle = [idxs[i] - idxs[i-1] for i in range(1, len(idxs))]

if reduceit:
	sample = block_reduce(sample, reduce_factor, func=np.mean)

# READ IN THE FRAMES:
stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
print len(files)

print('copying files')

for i, f in enumerate(files):

	if i % 100 == 0:
		print('%d images processed...' % i)
	tiff = TIFF.open(os.path.join(imdir, f), mode='r')
	im = tiff.read_image().astype('float')
	tiff.close()

	if reduceit:
		im_reduced = block_reduce(im, reduce_factor, func=np.mean)
		stack[:,:,i] = im_reduced #ndimage.gaussian_filter(im_reduced, sigma=gsigma)
	else:
		stack[:,:,i] = im

#stacks[session] = stack

# # SET FFT PARAMETERS:
freqs = fft.fftfreq(len(stack[0,0,:]), 1 / sampling_rate)
binwidth = freqs[1] - freqs[0]
#target_bin = int(target_freq / binwidth)
target_bin = np.where(freqs == min(freqs, key=lambda x: abs(float(x) - target_freq)))[0][0]
print "TARGET: ", target_bin, freqs[target_bin]
print "FREQS: ", freqs

freqs_shift = fft.fftshift(freqs)
target_bin_shift = np.where(freqs_shift == min(freqs_shift, key=lambda x: abs(float(x) - target_freq)))[0][0]
print "TARGET-shift: ", target_bin_shift, freqs_shift[target_bin_shift]
print "FREQS-shift: ", freqs_shift


window = sampling_rate * cycle_dur * 2

# # FFT:
mag_map = np.empty(sample.shape)
# phase_map = np.empty(sample.shape)
ft_real = np.empty(sample.shape)
ft_imag = np.empty(sample.shape)

ft_real_shift = np.empty(sample.shape)
ft_imag_shift = np.empty(sample.shape)

dynrange = np.empty(sample.shape)

dlist = []
i = 0
for x in range(sample.shape[0]):
	for y in range(sample.shape[1]):

		pix = scipy.signal.detrend(stack[x, y, :], type='constant') # THIS IS BASICALLY MOVING AVG WINDOW...
		
		dynrange[x,y] = np.log2(pix.max() - pix.min())

		curr_ft = fft.fft(pix) # fft.fft(pix) / len(pix)])
		#curr_ft_shift = fft.fftshift(curr_ft)

		ft_real[x, y] = curr_ft[target_bin].real
		ft_imag[x, y] = curr_ft[target_bin].imag

		mag = np.abs(curr_ft)
		mag_map[x, y] = mag[target_bin]

		#flattend = [f for sublist in ((c.real, c.imag) for c in curr_ft) for f in sublist]
		dlist.append((x, y, curr_ft))
		
		i+=1

DF = pd.DataFrame.from_records(dlist)

		# dynrange[x,y] = np.log2(pix.max() - pix.min())

		# mag = np.abs(curr_ft)
		# mag_max = np.where(mag == mag.max())
		# mag_min = np.where(mag == mag.min())

		
		# ft_real[x, y] = curr_ft[target_bin].real
		# ft_imag[x, y] = curr_ft[target_bin].imag

		# ft_real_shift[x, y] = curr_ft_shift[target_bin_shift].real
		# ft_imag_shift[x, y] = curr_ft_shift[target_bin_shift].imag

		# # if i % 100 == 0:
		# # 	print ft_real[x, y], ft_imag[x,y]
		
		# #mag_map[x, y] = mag[target_bin]
		# i += 1


		# # try:
		# # 	dynrange[x,y] = np.log2(stack[x, y, :].max()/stack[x, y, :].min())
		# # except RunTimeWarning:
		# # 	print f, x, y, dynrange[x,y]

		# pix = scipy.signal.detrend(stack[x, y, :]) # THIS IS BASICALLY MOVING AVG WINDOW...
		# #pix = stack[x,y,:]
		
		# dynrange[x,y] = np.log2(pix.max() - pix.min())

		# #pix = scipy.signal.detrend(pix)

		# #sig = movingaverage(pix, window)
		# #mpix = (pix[0:len(sig)] - sig) / sig

		# # sig = scipy.signal.detrend(sig)

		# #ft = fft.fft(scipy.signal.detrend(stack[x, y, :]))
		# ft[x, y] = [fft.fft(pix) / len(pix)]
		# #phase = np.angle(ft)
		# #mag_tmp = np.abs(ft) #**2

D = dict()
#D['ft'] = DF

D['ft_real'] = ft_real #np.array(ft)
D['ft_imag'] = ft_imag
# D['ft_real_shift'] = ft_real_shift #np.array(ft)
# D['ft_imag_shift'] = ft_imag_shift

D['mag_map'] = mag_map
D['mean_intensity'] = np.mean(stack, axis=2)
#D['stack'] = stack
#del stack
D['dynrange'] = dynrange
D['target_freq'] = target_freq
D['fps'] = sampling_rate
D['freqs'] = freqs #fft.fftfreq(len(pix), 1 / sampling_rate)

#D['freqs_shift'] = freqs_shift #fft.fftfreq(len(pix), 1 / sampling_rate)

D['binsize'] = freqs[1] - freqs[0] 
D['target_bin'] = target_bin #np.where(freqs == min(freqs, key=lambda x: abs(float(x) - target_freq)))[0][0]
D['target_bin_shift'] = target_bin_shift
#D['nframes'] = nframes_per_cycle
D['reduce_factor'] = reduce_factor

# SAVE condition info:
sessionpath = os.path.split(imdir)[0]
outdir = os.path.join(sessionpath, 'structs')
if not os.path.exists(outdir):
	os.makedirs(outdir)
print outdir

fext = 'D_target_%s_%s.pkl' % (cond, str(reduce_factor))
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)


D = dict()
D['ft'] = DF
fext = 'D_fft_%s_%s.pkl' % (cond, str(reduce_factor))
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    # protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(D, f, protocol=pkl.HIGHEST_PROTOCOL)




# 		#print target_bin, DC_bin

# 		# if binspread != 0:
# 		# 	#mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
# 		# 	mag_map[x, y] = np.mean(mag[target_bin-binspread:target_bin+binspread+1] / mag[0])
# 		# 	phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
# 		# else:
# 		# 	#mag_map[x, y] = 20*np.log10(mag[target_bin])
# 		# 	mag_map[x,y] = mag[target_bin] # / mag[DC_bin]
# 		# 	#mag_map[x,y] = mag[target_bin]
# 		# 	phase_map[x, y] = phase[target_bin]


# 		# if x % int(sample.shape[0] / 4) == 0 or y % int(sample.shape[1] / 4) == 0:
# 		# 	plt.subplot(2,1,1)
# 		# 	plt.plot(freqs, mag, '*')
# 		# 	plt.subplot(2,1,2)
# 		# 	plt.plot(freqs, phase, '*')
# 		# 	plt.show()


# 		# if binspread != 0:
# 		# 	mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
# 		# 	phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
# 		# else:
# 		# 	#mag_map[x, y] = 20*np.log10(mag[target_bin])
# 		# 	mag_map[x,y] = mag[target_bin]/mag[0.]
# 		# 	#mag_map[x,y] = mag[target_bin]
# 		# 	phase_map[x, y] = phase[target_bin]

# # PLOT IT:

# # basepath = os.path.split(os.path.split(imdir)[0])[0]
# # session = os.path.split(os.path.split(imdir)[0])[1]
# # cond = os.path.split(imdir)[1]

# plt.subplot(2,2,1) # GREEN LED image
# outdir = os.path.join(basepath, 'output')
# if os.path.exists(outdir):
# 	flist = os.listdir(outdir)
# 	# GET BLOOD VESSEL IMAGE:
# 	ims = [f for f in flist if os.path.splitext(f)[1] == '.png']
# 	if ims:
# 		impath = os.path.join(outdir, ims[0])
# 		image = Image.open(impath).convert('L')
# 		imarray = np.asarray(image)

# 		plt.imshow(imarray,cmap=cm.Greys_r)
# 	else:
# 		print "*** Missing green-LED photo of cortex surface. ***"
# else:
# 	spnum = 2

# plt.subplot(2, 2, 2)
# fig =  plt.imshow(dynrange)
# plt.title('Dynamic range (bits)')
# plt.colorbar()


# plt.subplot(2, 2, 3)
# # mag_map = mag_map*1E4
# #fig =  plt.imshow(np.clip(mag_map, 0, mag_map.max()), cmap=cm.hot)
# # fig = plt.imshow(np.clip(mag_map, 0, mag_map.max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1.0)
# fig = plt.imshow(mag_map, cmap = plt.get_cmap('gray'))#mag_map.max())
# plt.title('Magnitude @ %0.3f' % (freqs[round(target_bin)]))
# #fig.set_cmap("hot")
# plt.colorbar()


# plt.subplot(2, 2, 4)
# fig = plt.imshow(phase_map)
# plt.title('Phase (rad) @ %0.3f' % freqs[round(target_bin)])
# fig.set_cmap("spectral")
# plt.colorbar()

# plt.suptitle(session + ': ' + cond)

# # plt.show()

# # SAVE FIG
# figdir = os.path.join(basepath, 'figures', session, 'fieldmap')
# print figdir
# if not os.path.exists(figdir):
# 	os.makedirs(figdir)
# imname = session + '_' + cond + '_fieldmap' + str(reduce_factor) + '.png'
# plt.savefig(figdir + '/' + imname)

# plt.show()


# # SAVE MAPS:
# outdir = os.path.join(basepath, 'output', session)
# if not os.path.exists(outdir):
# 	os.makedirs(outdir)

# fext = 'magnitude_%s_%s_%i.pkl' % (cond, str(reduce_factor), gsigma)
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
#     pkl.dump(mag_map, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)

# fext = 'phase_%s_%s_%i.pkl' % (cond, str(reduce_factor), gsigma)
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
#     pkl.dump(phase_map, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)


# fext = 'dynrange_%s_%s_%i.pkl' % (cond, str(reduce_factor), gsigma)
# fname = os.path.join(outdir, fext)
# with open(fname, 'wb') as f:
#     pkl.dump(dynrange, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)