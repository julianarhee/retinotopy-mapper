import numpy as np
import os
from skimage.measure import block_reduce
from scipy.misc import imread
import matplotlib.pylab as plt
import cPickle as pkl
import scipy.signal
import numpy.fft as fft
import sys

from PIL import Image
import matplotlib.cm as cm

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


#imname = '/media/juliana/DATA/widefield/JR009/20150618_JR009/snapshots/20150618_JR009_750um_below_stxSurface_Screen Shot 2015-06-18 at 10.11.59 AM.png'
# image = Image.open(imname).convert('L')
# imarray = np.asarray(image)


outdir = sys.argv[1] # PATH TO OUTDIR:  '/media/juliana/MAC/data/JR009/20150620/output'
which_sesh = sys.argv[2]
reduce_val = sys.argv[3]

flist = os.listdir(outdir)

# GET BLOOD VESSEL IMAGE:
ims = [f for f in flist if os.path.splitext(f)[1] == '.png']
impath = os.path.join(outdir, ims[0])
image = Image.open(impath).convert('L')
imarray = np.asarray(image)


# LOAD SESSIONS:
sessions = [f for f in flist if os.path.splitext(f)[1] != '.png']
session_path = os.path.join(outdir, sessions[int(which_sesh)]) ## LOOP THIS

files = os.listdir(session_path)
files = [f for f in files if os.path.splitext(f)[1] == '.pkl']

reduce_factor = (int(reduce_val), int(reduce_val))

# GET PHASE MAPS:
fn_phase = [f for f in files if 'phase' in f]
fn_vert = [f for f in fn_phase if 'V' in f and str(reduce_factor) in f]
fn_horiz = [f for f in fn_phase if 'H' in f and str(reduce_factor) in f]

H_phasemaps = dict()
V_phasemaps = dict()
for f in fn_horiz:
	outfile = os.path.join(session_path, f)
	with open(outfile,'rb') as fp:
		H_phasemaps[f] = pkl.load(fp)
for f in fn_vert:
	outfile = os.path.join(session_path, f)
	with open(outfile,'rb') as fp:
		V_phasemaps[f] = pkl.load(fp)

if not len(fn_vert) == 2:
	print "*****************************************"
	print "Missing reverse conditions, session: %s" % os.path.split(session_path)[1]
	print "-----------------------------------------"
	print "Only have: %s" % fn_vert[0]
	print "*****************************************"
	V_phase = 0.
	V_delay = 0.
else:
	V_phase = (V_phasemaps[V_phasemaps.keys()[0]] - V_phasemaps[V_phasemaps.keys()[1]]) / 2.
	V_delay = (V_phasemaps[V_phasemaps.keys()[0]] + V_phasemaps[V_phasemaps.keys()[1]]) / 2.

if not len(fn_horiz) == 2:
	print "*****************************************"
	print "Missing reverse conditions, session: %s" % os.path.split(session_path)[1]
	print "-----------------------------------------"
	print "Only have: %s" % fn_horiz[0]
	print "*****************************************"
	H_phase = 0.
	H_delay = 0.
else:
	H_phase = (H_phasemaps[H_phasemaps.keys()[0]] - H_phasemaps[H_phasemaps.keys()[1]]) / 2.
	H_delay = (H_phasemaps[H_phasemaps.keys()[0]] + H_phasemaps[H_phasemaps.keys()[1]]) / 2.




# GET MAG MAPS:
fn_mag = [f for f in files if 'mag' in f]
fn_vert = [f for f in fn_mag if 'V' in f and str(reduce_factor) in f]
fn_horiz = [f for f in fn_mag if 'H' in f and str(reduce_factor) in f]

H_magmaps = dict()
V_magmaps = dict()
for f in fn_horiz:
	outfile = os.path.join(session_path, f)
	with open(outfile,'rb') as fp:
		H_magmaps[f] = pkl.load(fp)
for f in fn_vert:
	outfile = os.path.join(session_path, f)
	with open(outfile,'rb') as fp:
		V_magmaps[f] = pkl.load(fp)

if not len(fn_vert) == 2:
	print "*****************************************"
	print "Missing reverse conditions, session: %s" % os.path.split(session_path)[1]
	print "-----------------------------------------"
	print "Only have: %s" % fn_vert[0]
	print "*****************************************"
	V_mag = 0.
	V_delaymag = 0.
else:
	V_mag = (V_magmaps[V_magmaps.keys()[0]] - V_magmaps[V_magmaps.keys()[1]]) / 2.
	V_delaymag = (V_magmaps[V_magmaps.keys()[0]] + V_magmaps[V_magmaps.keys()[1]]) / 2.

if not len(fn_horiz) == 2:
	print "*****************************************"
	print "Missing reverse conditions, session: %s" % os.path.split(session_path)[1]
	print "-----------------------------------------"
	print "Only have: %s" % fn_horiz[0]
	print "*****************************************"
	H_mag = 0.
	H_delaym = 0.
else:
	H_mag = (H_magmaps[H_magmaps.keys()[0]] - H_magmaps[H_magmaps.keys()[1]]) / 2.
	H_delayM = (H_magmaps[H_magmaps.keys()[0]] + H_magmaps[H_magmaps.keys()[1]]) / 2.





# PLOT IT:

# plt.subplot(1,3,1) # GREEN LED image
# plt.imshow(imarray,cmap=cm.Greys_r)


# plt.subplot(1, 3, 2) # ABS PHASE -- azimuth
# fig = plt.imshow(V_phase)
# fig.set_cmap("spectral")
# plt.colorbar()
# plt.title("azimuth")

# plt.subplot(1,3,3) # ABS PHASE -- elevation
# fig = plt.imshow(H_phase)
# fig.set_cmap("spectral")
# plt.colorbar()
# plt.title("elevation")

# plt.suptitle(session_path)


# plt.show()



# PLOT IT ALL:

plt.subplot(3,4,1) # GREEN LED image
plt.imshow(imarray,cmap=cm.Greys_r)


plt.subplot(3, 4, 3) # ABS PHASE -- azimuth
fig = plt.imshow(V_phase)
fig.set_cmap("spectral")
plt.colorbar()
plt.title("azimuth")

plt.subplot(3,4,2) # ABS PHASE -- elevation
fig = plt.imshow(H_phase)
fig.set_cmap("spectral")
plt.colorbar()
plt.title("elevation")


# plt.subplot(4, 4, 4)
# fig =  plt.imshow(dynrange)
# plt.title('Dynamic range (bits)')
# plt.colorbar()


# PHASE:
plt.subplot(3, 4, 5)
fig =  plt.imshow(H_phasemaps[H_phasemaps.keys()[0]], cmap=cm.spectral)
plt.title('Phase (rad): %s' % H_phasemaps.keys()[0])
plt.colorbar()

plt.subplot(3, 4, 6)
if H_phase.any():
	fig =  plt.imshow(H_phasemaps[H_phasemaps.keys()[1]], cmap=cm.spectral)
	plt.title('Phase (rad): %s' % H_phasemaps.keys()[1])
	plt.colorbar()

plt.subplot(3, 4, 7)
fig =  plt.imshow(V_phasemaps[V_phasemaps.keys()[0]], cmap=cm.spectral)
plt.title('Phase (rad): %s' % V_phasemaps.keys()[0])
plt.colorbar()

plt.subplot(3, 4, 8)
if V_phase.any():
	fig =  plt.imshow(V_phasemaps[V_phasemaps.keys()[1]], cmap=cm.spectral)
	plt.title('Phase (rad): %s' % V_phasemaps.keys()[1])
	plt.colorbar()


# MAGNITUDE:
scaleit = 1#1E4
plt.subplot(3, 4, 9)
#fig =  plt.imshow(np.clip(H_magmaps[H_magmaps.keys()[0]], 0, H_magmaps[H_magmaps.keys()[0]].max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 5)
fig =  plt.imshow(H_magmaps[H_magmaps.keys()[0]]*scaleit, cmap = plt.get_cmap('gray'))
plt.title('Mag: %s' % H_magmaps.keys()[0])
plt.colorbar()

plt.subplot(3, 4, 10)
if H_mag.any():
	# fig =  plt.imshow(np.clip(H_magmaps[H_magmaps.keys()[1]], 0, H_magmaps[H_magmaps.keys()[1]].max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 5)
	fig =  plt.imshow(H_magmaps[H_magmaps.keys()[1]]*scaleit, cmap = plt.get_cmap('gray'))
	plt.title('Mag: %s' % H_magmaps.keys()[1])
	plt.colorbar()

plt.subplot(3, 4, 11)
# fig =  plt.imshow(np.clip(V_magmaps[V_magmaps.keys()[0]], 0, V_magmaps[V_magmaps.keys()[0]].max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 5)
fig =  plt.imshow(V_magmaps[V_magmaps.keys()[0]]*scaleit, cmap = plt.get_cmap('gray'))
plt.title('Mag: %s' % V_magmaps.keys()[0])
plt.colorbar()

plt.subplot(3, 4, 12)
if V_mag.any():
	# fig =  plt.imshow(np.clip(V_magmaps[V_magmaps.keys()[1]], 0, V_magmaps[V_magmaps.keys()[1]].max()), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 5)
	fig =  plt.imshow(V_magmaps[V_magmaps.keys()[1]]*scaleit, cmap = plt.get_cmap('gray'))
	plt.title('Mag: %s' % V_magmaps.keys()[1])
	plt.colorbar()


# fig = plt.imshow(phase_map)
# plt.title('Phase (rad)')
# fig.set_cmap("spectral")
# plt.colorbar()

plt.suptitle(session_path)



# SAVE FIG
outdirs = os.path.join(outdir, 'figures')
print outdirs
if not os.path.exists(outdirs):
	os.makedirs(outdirs)
imname = which_sesh  + '_allmaps_' + str(reduce_factor) + '.png'
plt.savefig(outdirs + '/' + imname)

plt.show()




# # Only take first 15 cycles (to reduce memory load)
# # strts = [i for i,f in enumerate(files) if '_0_' in f]
# # files = files[0:strts[-1]]
# # files = files[0:int(round(len(files)*0.5))]

# sample = imread(os.path.join(imdir, files[0]))
# # sample = sample[30:-1, 50:310]
# # sample = sample[:,70:300]
# sample = sample[20:230,40:275]
# print "FIRST", sample.dtype
# sample = block_reduce(sample, reduce_factor)

# # plt.figure()
# # plt.imshow(sample)
# # plt.show()


# # LOAD IN THE ACQUIRED FRAMES:
# stack = np.empty((sample.shape[0], sample.shape[1], len(files)))
# print len(files)

# print('copying files')

# for i, f in enumerate(files):

# 	if i % 100 == 0:
# 		print('%d images processed...' % i)
# 	im = imread(os.path.join(imdir, f)).astype('float')
# 	# print im.shape
# 	# im = im[30:-1, 50:310]
# 	# im = im[:,70:300]
# 	im = im[20:230,40:275]
# 	# print im.shape
# 	im_reduced = block_reduce(im, reduce_factor)
# 	stack[:,:,i] = im_reduced


# # SET EXPERIMENT PARAMETERS FOR FFT:
# sampling_rate = 60.0
# reduce_factor = (2, 2)
# cache_file = True
# target_freq = 0.1
# binspread = 0

# freqs = fft.fftfreq(len(stack[0,0,:]), 1 / sampling_rate)
# binwidth = freqs[1] - freqs[0]
# target_bin = int(target_freq / binwidth)


# # FFT:
# mag_map = np.empty(sample.shape)
# phase_map = np.empty(sample.shape)
# dynrange = np.empty(sample.shape)
# for x in range(sample.shape[0]):
# 	for y in range(sample.shape[1]):

# 		dynrange[x,y] = np.log2(stack[x, y, :].max()/stack[x, y, :].min())

# 		ft = fft.fft(scipy.signal.detrend(stack[x, y, :]))
# 		mag = abs(ft)
# 		# if mag[target_bin]==0:
# 		# 	# mag[target_bin]=1E100
# 		# 	print x, y
# 		phase = np.angle(ft)

# 		if binspread != 0:
# 			mag_map[x, y] = 20*np.log10(np.mean(mag[target_bin-binspread:target_bin+binspread]))
# 			phase_map[x, y] = np.mean(phase[target_bin-binspread:target_bin+binspread])
# 		else:
# 			mag_map[x, y] = 20*np.log10(mag[target_bin])
# 			phase_map[x, y] = phase[target_bin]

# plt.subplot(1, 3, 1)
# fig =  plt.imshow(dynrange)
# plt.colorbar()

# plt.subplot(1, 3, 2)
# fig =  plt.imshow(np.clip(mag_map, 0, mag_map.max()))
# fig.set_cmap("hot")
# plt.colorbar()

# plt.subplot(1, 3, 3)
# fig = plt.imshow(phase_map)
# fig.set_cmap("spectral")
# plt.colorbar()

# # SAVE FIG
# basepath = os.path.split(os.path.split(imdir)[0])[0]
# session = os.path.split(os.path.split(imdir)[0])[1]
# figdir = os.path.join(basepath, 'figures', session, 'fieldmap')
# print figdir
# if not os.path.exists(figdir):
# 	os.makedirs(figdir)
# sess = os.path.split(os.path.split(imdir)[0])[1]
# cond = os.path.split(imdir)[1]
# imname = sess + '_' + cond + '_fieldmap' + str(reduce_factor) + '.png'
# plt.savefig(figdir + '/' + imname)

# plt.show()


# # SAVE MAPS:
# outdir = os.path.join(basepath, 'output', session)
# if not os.path.exists(outdir):
# 	os.makedirs(outdir)

# fext = 'magnitude_%s.pkl' % cond
# fname = os.path.join(outdir, fname)
# with open(fname, 'wb') as f:
#     pkl.dump(mag_map, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)

# fext = 'phase_%s.pkl' % cond
# fname = os.path.join(outdir, fname)
# with open(fname, 'wb') as f:
#     pkl.dump(phase_map, f, protocol=pkl.HIGHEST_PROTOCOL) #protocol=pkl.HIGHEST_PROTOCOL)