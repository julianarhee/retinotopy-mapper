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

import PIL.Image as Image
import libtiff
import optparse

#imdir = sys.argv[1]

parser = optparse.OptionParser()
parser.add_option('--ext', action="store", dest="ext",
                  default="tif", help="frame image type (.tiff, .tif, .png)")
parser.add_option('--path', action="store",
                  dest="path", default="", help="input dir")


(options, args) = parser.parse_args()

ext = '.' + options.ext
imdir = options.path

outdir = os.path.join(os.path.split(imdir)[0], 'surface')
if not os.path.exists(outdir):
	os.makedirs(outdir)

cond = 'Surface'
# ext = '.tif'
files = os.listdir(os.path.join(imdir, cond))
files = sorted([f for f in files if os.path.splitext(f)[1] == ext])
print "FNs: ", files, cond


tiff = TIFF.open(os.path.join(imdir, cond, files[0]), mode='r')
im = tiff.read_image().astype('float')
# im = (im/2**12)*(2**16)
tiff.close()

fig = plt.figure(frameon=False)
# fig.set_size_inches(w,h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# ax.imshow(your_image, aspect='normal')
ax.imshow(im, cmap=plt.get_cmap('gray'))
# fig.savefig(fname, dpi)

# plt.imshow(im, cmap = plt.get_cmap('gray'))


# image = Image.open(os.path.join(imdir, cond, files[0]))
# image.show()

# fname = '%s/surface_%s.tif' % (outdir, os.path.split(os.path.split(imdir)[0])[1])
fname = '%s/%s_%s.tif' % (outdir, os.path.split(os.path.split(imdir)[0])[1], os.path.split(imdir)[1])
print outdir
# plt.savefig(fname)

tiff = TIFF.open(fname, mode='w')
tiff.write_image(im)
tiff.close()

print fname


fname = '%s/%s_%s_plot.png' % (outdir, os.path.split(os.path.split(imdir)[0])[1], os.path.split(imdir)[1])

plt.savefig(fname)

plt.show()


# import matplotlib.pyplot as pplt
# I = pplt.imread(os.path.join(imdir, cond, files[0]))
# pplt.imsave(fname, I, cmap=plt.get_cmap('gray'))
