import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import matplotlib.pylab as plt
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys

import matplotlib.cm as cm
import re
import itertools

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


imdirs = [sys.argv[1], sys.argv[2]]

# crop_fov = 0
# if len(sys.argv) > 2:
# 	cropped = sys.argv[2].split(',')
# 	[strtX, endX, strtY, endY] = [int(i) for i in cropped]
# 	crop_fov = 1

fdict = dict()
for imdir in imdirs:
	flist = os.listdir(imdir)
	flist = sorted([f for f in flist if os.path.splitext(f)[1] == '.png'])
	print flist[-1]
	fdict[imdir] = flist

if len(set([len(l) for l in fdict.values()])) > 1:
	cutoff = min(set([len(l) for l in fdict.values()]))
	chopthis = [f for f in fdict.keys() if len(fdict[f]) > cutoff]
	for key in chopthis:
		del fdict[key][cutoff:len(fdict[key])]

D = dict()
for fkey in fdict.keys():
	files = fdict[fkey]
	print fkey, len(files)
	sample = imread(os.path.join(fkey, files[0]))
	#sample = block_reduce(sample, reduce_factor, func = np.mean)
	plt.imshow(sample)
	plt.show()

	substack = np.empty((sample.shape[0], sample.shape[1], len(files)))
	print len(files)

	print('copying files')

	for i, f in enumerate(files):

		if i % 100 == 0:
			print('%d images processed...' % i)
		# print f
		im = imread(os.path.join(fkey, f)).astype('float')

		#im = im[strtX:endX,strtY:endY]
		# print im.shape

		#im_reduced = block_reduce(im, reduce_factor, func=np.mean)
		substack[:,:,i] = im #im_reduced

	D[fkey] = substack

stack = (D[D.keys()[0]] + D[D.keys()[1]] ) / 2.



# cycle_dur = 10.
# sampling_rate = 60. #np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
# reduce_factor = (4, 4)
# cache_file = True
# target_freq = 0.1
# binspread = 0

stack_sample = stack[:,:,0]
stack_sample = block_reduce(stack_sample, reduce_factor, func=np.mean)
plt.imshow(stack_sample)
plt.show()

stack = map(lambda x: block_reduce(x, reduce_factor, func=np.mean), stack)
print stack.shape

condition = os.path.split(imdir)[1]
session = os.path.split(os.path.split(imdir)[0])[1]
basepath = os.path.split(os.path.split(imdir)[0])[0]

stackdir = os.path.join(basepath, session, 'stacks')
if not os.path.exists(stackdir):
	os.makedirs(stackdir)

fext = 'stack_%s.pkl' % condition
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    pkl.dump(stack, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)







#files = sorted([f for f in files if os.path.isfile(os.path.join(imdir, f))])
#files = files[0:11982]
#files = files[11982:-1]
print files[-1]
# FIND CYCLE STARTS:
# This will depend on the file-saving name -- if 0'ed for frame # with each cycle, can just look for 0 in the fn
# Otherwise, prob better to just use position, which is also stored in brackets [x, y]

# strtidxs = []
# strts = [i for i,f in enumerate(files) if '_0_' in f]
# for f in [files[i] for i in strts]:
# 	idx = f.split('_')[3]
# 	strtidxs.append(int(idx))

# strtidxs[0:-3]

#files = files[0:strts[-1]]
# files = files[0:int(round(len(files)*0.5))]


positions = [re.findall("\[([^[\]]*)\]", f) for f in files]
plist = list(itertools.chain.from_iterable(positions))
positions = [map(float, i.split(',')) for i in plist]
find_cycs = list(itertools.chain.from_iterable(np.where(np.diff([p[1] for p in positions]) < 0)))
strt_idxs = [i+1 for i in find_cycs]
strt_idxs.append(0)
strt_idxs = sorted(strt_idxs)

sample = imread(os.path.join(imdir, files[0]))

# Divide into cycles:
chunks = []
step = 5
for i in range(0, len(strt_idxs)-1, step):
	print i
	chunks.append(files[strt_idxs[i]:strt_idxs[i+step]])


# If need to crop, can just replace strtX, strtY, etc.
# Otherwise, just use the whole thing.

# sample = sample[30:-1, 50:310]
# sample = sample[:,70:300]
#sample = sample[20:230,40:275]

if crop_fov:
	sample = sample[strtX:endX, strtY:endY]

	# strtX = 0 #35
	# endX = -1 #251
	# strtY = 0 #50
	# endY = -1 #275

# print "FIRST", sample.dtype
# sample = block_reduce(sample, reduce_factor)

# plt.figure()
# plt.imshow(sample)
# plt.show()


cycle_dur = 10.
sampling_rate = 60. #np.mean(np.diff(sorted(strt_idxs)))/cycle_dur #60.0
reduce_factor = (4, 4)
cache_file = True
target_freq = 0.1
binspread = 0

print "FIRST", sample.dtype
sample = block_reduce(sample, reduce_factor, func=np.mean)


# READ IN THE FRAMES:
stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
print len(files)

print('copying files')

for i, f in enumerate(files):

	if i % 100 == 0:
		print('%d images processed...' % i)
	# print f
	im = imread(os.path.join(imdir, f)).astype('float')

	#im = im[strtX:endX,strtY:endY]
	# print im.shape

	#im_reduced = block_reduce(im, reduce_factor, func=np.mean)
	stack[:,:,i] = im #im_reduced


# # SET FFT PARAMETERS:
# freqs = fft.fftfreq(len(stack[0,0,:]), 1 / sampling_rate)
# binwidth = freqs[1] - freqs[0]
# target_bin = int(target_freq / binwidth)
# window = sampling_rate * cycle_dur * 4
window = sampling_rate * cycle_dur * 2

# FFT:
mag_map = np.empty(sample.shape)
phase_map = np.empty(sample.shape)
dynrange = np.empty(sample.shape)
for x in range(sample.shape[0]):
	for y in range(sample.shape[1]):

		# try:
		# 	dynrange[x,y] = np.log2(stack[x, y, :].max()/stack[x, y, :].min())
		# except RunTimeWarning:
		# 	print f, x, y, dynrange[x,y]

		pix = stack[x, y, :]
		dynrange[x,y] = np.log2(pix.max() - pix.min())

		#pix = scipy.signal.detrend(pix)

		sig = movingaverage(pix, window)
		mpix = (pix[0:len(sig)] - sig) / sig

		# sig = scipy.signal.detrend(sig)

		#ft = fft.fft(scipy.signal.detrend(stack[x, y, :]))
		ft = fft.fft(mpix)
		phase = np.angle(ft)

		#ftraw = fft.fft(pix[0:len(mpix)])
		mag = np.abs(ft) #**2
		# if mag[target_bin]==0:
		# 	# mag[target_bin]=1E100
		# 	print x, y
		

		# SET FFT PARAMETERS:
		freqs = fft.fftfreq(len(mpix), 1 / sampling_rate) # sorted(fft.fftfreq(len(mpix), 1 / sampling_rate))
		binwidth = freqs[1] - freqs[0] 
		# np.where(freqs == min(freqs, key=lambda x: abs(float(x) - 0.1)))
		target_bin = round(target_freq/binwidth) #int(target_freq / binwidth)


		if binspread != 0:
			#mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
			mag_map[x, y] = np.mean(mag[target_bin-binspread:target_bin+binspread+1] / mag[0])
			phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
		else:
			#mag_map[x, y] = 20*np.log10(mag[target_bin])
			mag_map[x,y] = mag[target_bin] / mag[0.]
			#mag_map[x,y] = mag[target_bin]
			phase_map[x, y] = phase[target_bin]


		# if x % int(sample.shape[0] / 4) == 0 or y % int(sample.shape[1] / 4) == 0:
		# 	plt.subplot(2,1,1)
		# 	plt.plot(freqs, mag, '*')
		# 	plt.subplot(2,1,2)
		# 	plt.plot(freqs, phase, '*')
		# 	plt.show()


		# if binspread != 0:
		# 	mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
		# 	phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
		# else:
		# 	#mag_map[x, y] = 20*np.log10(mag[target_bin])
		# 	mag_map[x,y] = mag[target_bin]/mag[0.]
		# 	#mag_map[x,y] = mag[target_bin]
		# 	phase_map[x, y] = phase[target_bin]

# PLOT IT:
plt.subplot(1, 3, 1)
fig =  plt.imshow(dynrange)
plt.title('Dynamic range (bits)')
plt.colorbar()

plt.subplot(1, 3, 2)
# mag_map = mag_map*1E4
#fig =  plt.imshow(np.clip(mag_map, 0, mag_map.max()), cmap=cm.hot)
# fig = plt.imshow(np.clip(mag_map, 0, mag_map.max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1.0)
fig = plt.imshow(mag_map, cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1)#mag_map.max())

plt.title('Magnitude @ %0.3f' % (freqs[round(target_bin)]))
#fig.set_cmap("hot")
plt.colorbar()

plt.subplot(1, 3, 3)
fig = plt.imshow(phase_map)
plt.title('Phase (rad) @ %0.3f' % freqs[round(target_bin)])
fig.set_cmap("spectral")
plt.colorbar()

session = os.path.split(os.path.split(imdir)[0])[1]
cond = os.path.split(imdir)[1]
plt.suptitle(session + ': ' + cond)

# plt.show()

# SAVE FIG
basepath = os.path.split(os.path.split(imdir)[0])[0]
figdir = os.path.join(basepath, 'figures', session, 'fieldmap')
print figdir
if not os.path.exists(figdir):
	os.makedirs(figdir)
imname = session + '_' + cond + '_fieldmap' + str(reduce_factor) + '.png'
plt.savefig(figdir + '/' + imname)

plt.show()


# SAVE MAPS:
outdir = os.path.join(basepath, 'output', session)
if not os.path.exists(outdir):
	os.makedirs(outdir)

fext = 'magnitude_%s_%s.pkl' % (cond, str(reduce_factor))
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    pkl.dump(mag_map, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)

fext = 'phase_%s_%s.pkl' % (cond, str(reduce_factor))
fname = os.path.join(outdir, fext)
with open(fname, 'wb') as f:
    pkl.dump(phase_map, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)