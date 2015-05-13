
import os
import matplotlib.pyplot as plt
import cPickle as pkl
import numpy as np
#import pandas as pd
import time
from PIL import Image
from multiprocessing import Pool 
import glob

import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    

# Your code, but wrapped up in a function       
def convert(filename):  
    im = Image.open(filename)
    w,h = im.size
    imc = im.crop((75,0,w,h)) # left-upper, right-bottom

    return np.array(imc)


base_dir = '/Volumes/MAC/data/middle_depth/'
sessions = os.listdir(base_dir)

for s in sessions:
	imlist = sorted(glob.glob(os.path.join(base_dir, s, '*.png')), key=natural_keys)
	# i = 0
	poolfirst = Pool()
	results1 = poolfirst.map(convert, imlist[0:int(len(imlist)*0.5)])
	stack1 = np.dstack(results1)

	poolsecond = Pool()
	results2 = poolsecond.map(convert, imlist[int(len(imlist)*0.5):len(imlist)])
	stack2 = np.dstack(results2)

	if '001' in s:
		Nstack = np.dstack([stack1, stack2])
	elif '002' in s:
		Tstack = np.dstack([stack1, stack2])
	elif '003' in s:
		Ustack = np.dstack([stack1, stack2])
	elif '004' in s:
		Dstack = np.dstack([stack1, stack2])

del stack1, stack2, results1, results2

# FIRST LOOK AT AZ MAPS:
azimuth = np.zeros((Nstack.shape[0],Nstack.shape[1]))
for i in range(0, Nstack.shape[0]):
	for j in range(0, Nstack.shape[1]):
		nasal = Nstack[i,j,:]
		temporal = Tstack[i,j,:]

		# nidx, val = min(enumerate(nasal), key=operator.itemgetter(1))
		nidx = min(xrange(len(nasal)),key=nasal.__getitem__)
		tidx = min(xrange(len(temporal)),key=temporal.__getitem__)
		azimuth[i,j] = nidx - tidx

cropA = azimuth[100:492, 0:525]
# A = (((cropA - cropA.min()) / (cropA.max() - cropA.min())) * 255.9).astype(np.uint8)

A = (cropA - cropA.min()) / (cropA.max() - cropA.min())
A = (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min())
plt.imshow(A)
plt.figure()
plt.imshow(plt.imshow(Nstack[:,:,1]))

# LOOK AT ELEV MAPS:
elevation = np.zeros((Ustack.shape[0],Ustack.shape[1]))
for i in range(0, Ustack.shape[0]):
	for j in range(0, Ustack.shape[1]):
		up = Ustack[i,j,:]
		down = Dstack[i,j,:]

		# nidx, val = min(enumerate(nasal), key=operator.itemgetter(1))
		uidx = min(xrange(len(up)),key=up.__getitem__)
		didx = min(xrange(len(down)),key=down.__getitem__)
		elevation[i,j] = uidx - didx

cropE = elevation[100:492, 0:525]
# A = (((cropA - cropA.min()) / (cropA.max() - cropA.min())) * 255.9).astype(np.uint8)
E = (cropE - cropE.min()) / (cropE.max() - cropE.min())
E = (elevation - elevation.min()) / (elevation.max() - elevation.min())
plt.imshow(E)
plt.figure()
plt.imshow(Ustack[:,:,1])
