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

imdir = sys.argv[1]

parser = optparse.OptionParser()
parser.add_option('--ext', action="store", dest="ext",
                  default="tif", help="frame image type (.tiff, .tif, .png)")

(options, args) = parser.parse_args()

ext = '.' + options.ext

outdir = os.path.join(os.path.split(imdir)[0], 'figures')
if not os.path.exists(outdir):
	os.makedirs(outdir)

cond = 'Surface'
# ext = '.tif'
files = os.listdir(os.path.join(imdir, cond))
files = sorted([f for f in files if os.path.splitext(f)[1] == ext])
print "FNs: ", files, cond


tiff = TIFF.open(os.path.join(imdir, cond, files[0]), mode='r')
im = tiff.read_image().astype('float')
tiff.close()

plt.imshow(im, cmap = plt.get_cmap('gray'))


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
# plt.savefig(fname)

plt.show()


import matplotlib.pyplot as pplt
I = pplt.imread(os.path.join(imdir, cond, files[0]))
pplt.imsave(fname, I, cmap=plt.get_cmap('gray'))