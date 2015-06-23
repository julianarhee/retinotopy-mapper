#!/usr/bin/env python2
# make directory of images (PIL) to present w/ psychopy.visual

from PIL import Image, ImageSequence
import h5py
f = h5py.File('stim.mat')
circstim = f['stim']
from scipy.misc import toimage

n_images = circstim.shape[0]
n=0
for i in range(n_images):
    print i
    im = Image.fromarray(circstim[i])
    im.save("./imframes/im%s.png" % str(n))
    n+=1

