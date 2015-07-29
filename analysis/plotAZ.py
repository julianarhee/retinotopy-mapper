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

from libtiff import TIFF

imdir = sys.argv[1]
reduce_factor = (2,2)
reduceit = 1

crop_fov = 0
if len(sys.argv) > 2:
	cropped = sys.argv[2].split(',')
	[strtX, endX, strtY, endY] = [int(i) for i in cropped]
	crop_fov = 1

conditions = os.listdir(imdir)
conditions = [c for c in conditions if not 'png' in c]
D = dict()
for cond in conditions:
	print cond
	files = os.listdir(os.path.join(imdir, cond))
	files = sorted([f for f in files if os.path.splitext(f)[1] == '.png'])
	strtidxs = [i for i, item in enumerate(files) if re.search('_0_', item)]
	strtidxs.append(len(files))

	tiff = TIFF.open(os.path.join(imdir, cond, files[0]), mode='r')
	sample = tiff.read_image().astype('float')
	if reduceit:
		sample = block_reduce(sample, reduce_factor, func=np.mean)

	tiff.close()
	print sample.dtype

	# READ IN THE FRAMES:
	ST = []
	for iterx, idx in enumerate(strtidxs[0:-1]):
		tmp_files = files[idx:strtidxs[iterx+1]]

		stack = np.empty((sample.shape[0], sample.shape[1], len(tmp_files)))
		#print tmp_files[0], tmp_files[-1]

		print('copying files')

		for i, f in enumerate(tmp_files):

			if i % 100 == 0:
				print('%d images processed...' % i)

			#im = imread(os.path.join(imdir, cond, f)).astype('float')
			
			tiff = TIFF.open(os.path.join(imdir, cond, f), mode='r')
			im = tiff.read_image().astype('float')
			tiff.close()

			if reduceit:
				im = block_reduce(im, reduce_factor, func=np.mean)
			
			stack[:,:,i] = im

		ST.append(np.mean(stack, axis=2))

	ST = np.asarray(ST)
	print ST.shape
	D[cond] = np.mean(ST, axis=0)
	print D[cond].shape

blank = D['blank']
del D['blank']
print blank.shape

for dkey in D.keys():
	D[dkey+'_norm'] = (blank - D[dkey]) / blank


imdiff = D['gab-left'] - D['gab-right']

plt.subplot(2,3,1)
plt.imshow(D['gab-left'], cmap = plt.get_cmap('gray'))
plt.title('gab-left')
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(D['gab-right'], cmap = plt.get_cmap('gray'))
plt.title('gab-right')
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(D['gab-left_norm'], cmap = plt.get_cmap('gray'))
plt.title('gab-left_norm')
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(D['gab-right_norm'], cmap = plt.get_cmap('gray'))
plt.title('gab-right_norm')
plt.colorbar()

plt.subplot(2,3,6)
plt.imshow(D['gab-left_norm'] - D['gab-right_norm'], cmap = plt.get_cmap('gray'))
plt.title('norm diff')
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(imdiff, cmap = plt.get_cmap('gray'))
plt.title('plain diff')
plt.colorbar()


fname = '%s/flashbar.png' % os.path.split(imdir)[0]
plt.savefig(fname)

plt.show()

#img = img_as_uint(currdict['im'])
#io.imsave(fname, img)
#img = scipy.misc.toimage(currdict['im'], cmax=65535, cmin=0, mode='I')

#img = scipy.misc.toimage(currdict['im'], high=np.max(currdict['im']), low=np.min(currdict['im']), mode='I')
#img.save(fname)
# tiff = TIFF.open(fname, mode='w')
# img = Image.fromarray(imdiff)
# tiff.write_image(img)
# tiff.close()


# D = []
# prev = 0
# for i in idxs:
#     D.append(stack[:,:,prev:i])
#     prev = i
# D.append(stack[:,:,prev:])

# del stack

# nframes_per_cycle = [d.shape[2] for d in D]

# xtra = [i for i,d in enumerate(D) if d.shape[2]!=max(nframes_per_cycle)]
# print np.mean(nframes_per_cycle)

# # good = 0
# # for i,xt in enumerate(xtra):
# # 	plt.subplot(1, len(xtra)+1, i)
# # 	currbin = positions[idxs[xt]:idxs[xt+1]]
# # 	posx = [currbin[x][0] for x in range(len(currbin))]
# # 	posy = [currbin[x][1] for x in range(len(currbin))]
# # 	plt.plot(np.diff(posy))
# # 	plt.title('nframes: %i' % int(nframes_per_cycle[xt]))

# # while not good in xtra:
# # 	good += 1
# # plt.subplot(1,len(xtra)+1, i+1)
# # currbin = positions[idxs[good]:idxs[good+1]]
# # posx = [currbin[x][0] for x in range(len(currbin))]
# # posy = [currbin[x][1] for x in range(len(currbin))]
# # plt.plot(np.diff(posy))
# # plt.title('nframes: %i' % int(nframes_per_cycle[good]))


# # for x in xtra:
# # 	D[x] = D[x][:,:,0:595] 

# D = map(lambda x: x[:,:,0:min(nframes_per_cycle)], D)

# meanD = sum(D) / len(D)
# print meanD.shape

# S = np.empty((meanD.shape[0], meanD.shape[1], meanD.shape[2]), np.uint16)
# for i in range(1,meanD.shape[2]):
# 	S[:,:,i-1] = meanD[:,:,i] - meanD[:,:,0]

# del D

# # os.path.split(imdir)[0]
# condition = os.path.split(imdir)[1]
# framedir = os.path.join(os.path.split(imdir)[0], 'processed', condition)
# if not os.path.exists(framedir):
# 	os.makedirs(framedir)
# for i in range(S.shape[2]):
# 	fname = '%s/%0.4i.tif' % (framedir, i)
# 	imarray = S[:,:,i]
# 	#tiff = TIFF.open(fname, mode='w')
# 	#tiff.imsave(fname, imarray)
# 	#tiff.close()
# 	#plt.imshow(imarray)
# 	#plt.show()
# 	cv2.imwrite(fname, imarray)

	#img = scipy.misc.toimage(S[:,:,i], high=imarray.max(), low=imarray.min(), mode='I')
	#img = scipy.misc.toimage(S[:,:,i], high=65536, low=0, mode='I')
	#img.save(fname)

