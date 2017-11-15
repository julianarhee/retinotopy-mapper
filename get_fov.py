#!/usr/bin/env python2

from psychopy import monitors, visual
import os
import nump as np

monitor_dir = '~/Repositories/retinotopy-mapper/protocols/monitors'

calibs = os.listdir(monitor_dir)
calibs = [c for c in calibs if c.endswith('calib')]
calib_names = [c[:-6] for c in calibs]

for idx,calib in enumerate(calib_names):
    print idx, calib

mon_idx = raw_input('Select IDX of monitor to use: ')
mon = monitors.Monitor(calib_names[mon_idx])

distance = mon.getDistance()
width = mon.getWidth()
pix = mon.getSizePix()
aspect = float(pix[0])/float(pix[1])
height = width * (1./aspect))
pix_cm = float(width)/float(pix[0])


width_deg = 2*180*np.arctan((pix[0]*pix_cm)/(2*distance))/math.pi
height_deg = 2*180*np.arctan((pix[1]*pix_cm)/(2*distance))/math.pi

center = [width_deg/2., height_deg/2.]
interval = 10. 

azimuth_pts = list(np.arange(center[0], width_deg+10, interval))
prev = center[0]
while prev>0:
    azimuth_pts.append(prev-interval)
    prev -= interval
azimuth_pts = sorted(azimuth_pts)

elevation_pts = list(np.arange(center[1], height_deg+10, interval))
prev = center[1]
while prev>0:
    elevation_pts.append(prev-interval)
    prev -= interval
elevation_pts = sorted(elevation_pts)

D = dict()
for condition in conditions:
    condition_dir = os.path.join(outdir, condition, 'structs')
    if not os.path.exists(condition_dir):
         continue
    condition_structs = os.listdir(condition_dir)
    condition_structs = [f for f in condition_structs if '.pkl' in f and 'fft' in f]
    print "Found condition structs: ", condition_structs

                                                            D[condition] = dict()
                                                                    for condition_struct in condition_structs:
                                                                                    curr_condition_struct = os.path.join(condition_dir, condition_struct)
                                                                                                curr_cond_key = condition_struct.split('Target_fft_')[1].split('_.pkl')[0]
                                                                                                            with open(curr_condition_struct, 'rb') as f:
                                                                                                                                D[condition][curr_cond_key] = pkl.load(f)


