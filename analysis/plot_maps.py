#!/user/bin/env python2
import matplotlib as mpl
mpl.use('Agg')
import os
import h5py
import re
import pprint
import json
import datetime
import matplotlib.gridspec as gridspec
import numpy as np
import pylab as pl
import numpy as np
import pandas as pd
from libtiff import TIFF
from skimage.measure import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import generic_filter as gf
pp = pprint.PrettyPrinter(indent=4)

# In[4]:

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_trials(acquisition_dir, runs, excluded=None, averaged=False):
    if excluded is None:
        excluded = dict((run, []) for run in runs)

    trials = dict()
    for run in runs:
        if averaged is True:
            trial_dir = os.path.join(acquisition_dir, run, 'averaged_trials')
        else:
            trial_dir = os.path.join(acquisition_dir, run, 'raw')

        trial_list = [t for t in os.listdir(trial_dir) if 'trial' in t and os.path.isdir(os.path.join(trial_dir, t))]
        tinfo = []
        for trial in trial_list:
            if trial in excluded[run]:
                continue
            # laod frame info:
            finfo_path = os.path.join(trial_dir, trial, 'frame_info.txt')
            fdata = pd.read_csv(finfo_path, sep='\t')
            if 'cond_idx' in fdata.keys():
                cond = fdata['cond_idx'][0] #fdata['currcond'][0]
            else:
                cond = fdata['currcond'][0]
            if cond == 2:
                condname = 'right'
            elif cond == 4:
                condname = 'bottom'
            tinfo.append((trial, condname))
        trials[run] = tinfo

    return trials

def check_excluded_trials(runs):

    excluded = dict((run, []) for run in runs)
    for run in excluded.keys():
        excluded_str = raw_input('Enter comma-sep list of trials to exclude: ')
        excluded_trials = []
        if len(excluded_str) > 0:
            excluded_trials = ['trial%03d' % int(t) for t in excluded_str.split(',')]
        excluded[run] = excluded_trials

    return excluded


#%% Surface image:

def load_surface(acquisition_dir):

    if 'surface' in os.listdir(acquisition_dir):
        surface_dir = os.path.join(acquisition_dir, 'surface')
        if 'Surface' in os.listdir(surface_dir):
            surface_dir = os.path.join(surface_dir, 'Surface')
    elif 'Surface' in os.listdir(acquisition_dir):
        surface_dir = os.path.join(acquisition_dir, 'Surface')

    surface_fn = [f for f in os.listdir(surface_dir) if f.endswith(img_fmt)][0]
    print "Found surface at: %s" % surface_fn
    surface_impath = os.path.join(surface_dir, surface_fn)

    surftiff = TIFF.open(surface_impath, mode='r')
    surface = surftiff.read_image().astype('float')
    surftiff.close()

    # FIGURE:
    pl.figure()
    pl.imshow(surface, cmap='gray')
    pl.axis('off')
    pl.savefig(os.path.join(acquisition_dir, 'surface.png'))
    pl.close()

    print "Surface is size: ", surface.shape

    return surface


# In[80]:
def check_map_dims(mapimg, refimg):

    if not mapimg.shape==refimg.shape:
        reduce_val = refimg.shape[0]/mapimg.shape[0]
        print reduce_val
        if reduce_val>1:
            refimg = block_reduce(refimg, (reduce_val, reduce_val), func=np.mean)
            print refimg.shape
        elif reduce_val<1:
            reduce_val = 1./reduce_val
            mapimg = block_reduce(mapimg, (reduce_val, reduce_val), func=np.mean)

    return mapimg, refimg


def plot_az_el(surface, azmap, elmap, cmap='gist_rainbow', outdir='/tmp', figname='az_el'):

    pl.figure(figsize=(15,5))
    pl.subplot(1,2,1); pl.title('Azimuth')
    pl.imshow(surface, cmap='gray'); pl.axis('off')
    pl.imshow(azmap, alpha=0.5, cmap=cmap); pl.axis('off')

    #%
    pl.subplot(1,2,2); pl.title('Elevation')
    pl.imshow(surface, cmap='gray'); pl.axis('off')
    pl.imshow(elmap, alpha=0.5, cmap=cmap); pl.axis('off')
    pl.suptitle('phase maps')

    pl.savefig(os.path.join(outdir, "%s.png" % figname))

    pl.close()


def load_map(fftpath, maptype='phase_target'):
    fft = None
    try:
        fft = h5py.File(fftpath, 'r')
        mapimg = np.array(fft['maps'][maptype])
    except Exception as e:
        if fft is not None:
            print "Error loading map type: %s" % maptype
            print "Found map types:"
            for k in fft['maps'].keys():
                print k
        else:
            print "Unable to open: %s" % fftpath
        traceback.print_exc()
    finally:
        fft.close()

    return mapimg


def threshold_map(inputmap, refmap, thr=0.2):
    thr_map = np.copy(inputmap)
    thr_map[refmap<=thr] = np.nan
    return thr_map


def plot_cond_maps(phasemap, ratiomap, surface, thr_map=None, figname='map', writedir='/tmp',
                         thr=0.1, phasemin=0, phasemax=2*np.pi, cmap='hsv'):

#    phasemin = 0
#    phasemax = 2*np.pi

    pl.figure(figsize=(15,5))
    if thr_map is None:
        nplots = 2
    else:
        nplots = 3

    pl.subplot(1,nplots,1); pl.title('phase')
    pl.imshow(surface, cmap='gray')
    pl.imshow(phasemap, alpha=0.5, cmap=cmap, vmin=phasemin, vmax=phasemax); pl.axis('off')

    ax=pl.subplot(1,nplots,2); pl.title('mag ratio')
    pl.imshow(surface, cmap='gray')
    im=pl.imshow(ratiomap, alpha=0.5, cmap='hot'); pl.axis('off');
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)
#
#    phasemap_thr = np.copy(phasemap)
#    phasemap_thr[ratiomap<=thr] = np.nan

    if thr_map is not None:
        pl.subplot(1,nplots,3); pl.title('thresholded, %.3f' % thr)
        pl.imshow(surface, cmap='gray')
        pl.imshow(thr_map, alpha=0.5, cmap=cmap, vmin=phasemin, vmax=phasemax); pl.axis('off')
        figname = "%s_thr%.3f" % (figname, thr)

    pl.savefig(os.path.join(writedir, '%s.png' % figname))
    pl.close()

#%%
def extract_cond_maps(run_dir, trial_list, cond='AZ', threshold=0.1, cmap='gist_rainbow', output_figdir='/tmp'):

    # Get surface img for acquisition:
    acquisition_dir = os.path.split(run_dir)[0]
    surface = load_surface(acquisition_dir)

    trialmaps = dict()

    for curr_trial in trial_list:

        trialmaps[curr_trial] = dict()

        fft_fn = sorted([f for f in os.listdir(os.path.join(run_dir, 'processed')) if f.endswith('hdf5') and "%s_fft" % curr_trial in f])[-1]

        print "%s: trial - %s, file - %s" % (cond, curr_trial, fft_fn)
        fft_path = os.path.join(run_dir, 'processed', fft_fn)

        # Load maps:
        phasemap = load_map(fft_path, maptype='phase_target')
        ratiomap = load_map(fft_path, maptype='magnitude_ratios')

        # Check dimensions:
        phasemap, surface = check_map_dims(phasemap, surface)
        ratiomap, surface = check_map_dims(ratiomap, surface)

        # Make phase range continuous:
        contphase = -1 * phasemap
        contphase = contphase % (2*np.pi)

        # Threshold:
        phase_thr = threshold_map(contphase, ratiomap, thr=threshold)

        plot_cond_maps(contphase, ratiomap, surface, thr_map=phase_thr, figname='%s_%s' % (cond, curr_trial),
                             writedir=output_figdir, thr=threshold, cmap=cmap)

        trialmaps[curr_trial]['phase'] = contphase
        trialmaps[curr_trial]['ratio'] = ratiomap
        trialmaps[curr_trial]['surface'] = surface

    return trialmaps

def merge_dict_maps(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def smooth_maps(mapimg, radius=2):

    # Low-pass filter phase map w/ uniform circular kernel:
    radius = 2
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1

    map_filt = gf(mapimg, np.min, footprint=kernel)

    return map_filt

#%%


def convert_values(oldval, newmin, newmax, oldmin=0, oldmax=2*np.pi):
    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval


def phase_to_linear(phasemap, new_min=-65, new_max=65, orig_min=0, orig_max=2*np.pi):
    phasemap_lincoord = np.copy(phasemap)

    for x in range(phasemap.shape[0]):
        for y in range(phasemap.shape[1]):
            if not np.isnan(phasemap[x,y]):
                phasemap_lincoord[x,y] = convert_values(phasemap[x,y], new_min, new_max)

    return phasemap_lincoord


def get_linear_coords(width, height, resolution):
    C2A_cm = width/2.
    C2T_cm = height/2.
    C2P_cm = width/2.
    C2B_cm = height/2.
    print "center 2 Top/Anterior:", C2T_cm, C2A_cm

    # Convert to linear space centered about 0:
    mapx = np.linspace(-1*C2A_cm, C2P_cm, resolution[0])
    mapy = np.linspace(C2T_cm, -1*C2B_cm, resolution[1])

    lin_coord_x, lin_coord_y = np.meshgrid(mapx, mapy, sparse=False)

    mapcorX, mapcorY = np.meshgrid(range(resolution[0]), range(resolution[1]))

    return lin_coord_x, lin_coord_y, mapcorX, mapcorY

#%%
def get_contour_levels(coord_min, coord_max, interval=10):

    levels = range(int(np.floor(coord_min / interval) * interval),
                        int((np.ceil(coord_max / interval) + 1) * interval), interval)

    return levels

def get_legends(width, height, resolution, interval=10, cmap='gist_rainbow', contours=True, short_axis=False):

    lin_coord_x, lin_coord_y, mapcorX, mapcorY = get_linear_coords(width, height, resolution)

    linminW = lin_coord_x.min()
    linmaxW = lin_coord_x.max()

    linminH = lin_coord_y.min()
    linmaxH = lin_coord_y.max()

    f1 = pl.figure(figsize=(15,5))

    # Draw AZ legend:
    pl.subplot(1,2,1)
    pl.imshow(lin_coord_x, vmin=linminW, vmax=linmaxW,  cmap=cmap)
    if contours is True:
        # Get contours based on interval:
        levelsX = get_contour_levels(linminW, linmaxW, interval=interval)
        # Draw contour liens:
        im1 = pl.contour(mapcorX, mapcorY, lin_coord_x, levelsX, colors='k', linewidth=2)
        pl.clabel(im1, levelsX, fontsize=8, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos) #f1.colorbar(currfig, ticks=levels1)
    pl.axis('off')

    # Draw EL legend:
    pl.subplot(1,2,2)
    if short_axis is False:
        pl.imshow(lin_coord_y, vmin=linminH, vmax=linmaxH, cmap=cmap) #pl.colorbar()
    else:
        pl.imshow(lin_coord_y, vmin=linminW, vmax=linmaxW, cmap=cmap) #pl.colorbar()
    if contours is True:
        levelsY = get_contour_levels(linminH, linmaxH, interval=interval)
        im2 = pl.contour(mapcorX, mapcorY, lin_coord_y, levelsY, colors='k', linewidth=2)
        pl.clabel(im2, levelsY, fontsize=8, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
    pl.axis('off')

    pl.savefig(os.path.join(output_figdir, 'legends.png'))
    pl.close()

    return linminW, linmaxW, linminH, linmaxH


def add_inset_axes(ax,rect,axisbg='w'):
    fig = pl.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def plot_contour_maps(phasemap, width, height, resolution, surface=None, cond='AZ', output_figdir='/tmp', interval=10, cmap='gist_rainbow'):

    imsize = phasemap.shape
    if surface is None:
        surface = np.nan(imsize)

    imgX, imgY = np.meshgrid(range(imsize[1]), range(imsize[0]))
    lin_coord_x, lin_coord_y, mapcorX, mapcorY = get_linear_coords(width, height, resolution)

    if cond == 'AZ':
        lin_coords = lin_coord_x.copy()
    elif cond == 'EL':
        lin_coords = lin_coord_y.copy()

    linmin = lin_coords.min()
    linmax = lin_coords.max()

    phasemap_linear = phase_to_linear(phasemap, new_min=linmin, new_max=linmax, orig_min=0, orig_max=2*np.pi)

    fig = pl.figure(figsize=(15,10))
    ax1 = fig.add_subplot(111)
    subpos = [0.6,0.75,0.4,0.25] # These are in unitless percentages of the figure size. (0,0 is bottom left)
    ax2 = add_inset_axes(ax1, subpos)

    #pl.subplot(1,2,1)
    ax1.imshow(surface, cmap='gray')
    ax1.imshow(phasemap, vmin=phasemin, vmax=phasemax, cmap=cmap, alpha=0.3)
    levels = get_contour_levels(linmin, linmax, interval=interval)
    im1 = ax1.contour(imgX, imgY, phasemap_linear, levels, colors='k', linewidth=linewidth)
    ax1.clabel(im1, levels, fontsize=24, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
    pl.axis('off')

    pl.imshow(lin_coords, vmin=linmin, vmax=linmax, cmap=cmap, alpha=0.5)
    im2 = ax2.contour(mapcorX, mapcorY, lin_coords, levels, colors='k', linewidth=2)
    pl.clabel(im2, levels, fontsize=18, inline=3, fmt='%1.1f') #, inline_spacing=2, manual=label_pos)
    pl.axis('off')

    pl.savefig(os.path.join(output_figdir, '%s_contours.png' % cond))
    pl.close()

# In[21]:

rootdir = '/nas/volume1/2photon/data'
animalid = 'JR071'
session = '20180211'
acquisition = 'macro_fullfov'
#run = 'bar_013Hz_flash'
averaged = False

img_fmt = 'tif'
acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)


#%% ### Get all RUNS and TRIALS in each run:

runs = sorted([run for run in os.listdir(acquisition_dir) if os.path.isdir(os.path.join(acquisition_dir, run)) and not re.search('surface', run, re.IGNORECASE)], key=natural_keys)

print "RUNS:"
for r,run in enumerate(runs):
    print r, run


#%% ##### User-input specifies which trials to exclude:
excluded = check_excluded_trials(runs)
print "Excluded trials:"
pp.pprint(excluded)

# Get dict of trial-condition for each run:
trials = get_trials(acquisition_dir, runs, excluded=excluded, averaged=averaged)
print "Trials by run:"
pp.pprint(trials)

# In[37]:

runidx = 0
print runs


# In[57]:

curr_run = runs[runidx]
run_dir = os.path.join(acquisition_dir, curr_run)
raw_dir = os.path.join(run_dir, 'raw')

print "CURR RUN:", curr_run

el_trials = []
az_trials = []
for trial,cond in trials[runs[runidx]]:
    if cond=='bottom':
        el_trials.append(trial)
    if cond=='right':
        az_trials.append(trial)

az_trials = sorted(az_trials, key=natural_keys)
el_trials = sorted(el_trials, key=natural_keys)
print "AZ:", az_trials
print "EL:", el_trials

#%%

#% Plot Phase and Mag-ratios:
output_figdir = os.path.join(run_dir, 'figures')
output_figdir_trials = os.path.join(output_figdir, 'trials')
if not os.path.exists(output_figdir_trials):
    os.makedirs(output_figdir_trials)

threshold = 0.005
radius = 2
cmap='gist_rainbow'
phasemin = 0
phasemax = 2*np.pi

#%%%
azmaps = extract_cond_maps(run_dir, az_trials, cond='AZ', threshold=threshold, output_figdir=output_figdir_trials)
elmaps = extract_cond_maps(run_dir, el_trials, cond='EL', threshold=threshold, output_figdir=output_figdir_trials)

#%
trialmaps = merge_dict_maps(azmaps, elmaps)

#%% Save dicts

datestamp = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S.%f")
mapfile = h5py.File(os.path.join(output_figdir_trials, 'maps_%s.hdf5' % datestamp))

try:
    for trial in sorted(trialmaps.keys(), key=natural_keys):
        if trial not in mapfile.keys():
            trialgrp = mapfile.create_group(trial)
            trialgrp.attrs['condition'] = [t[1] for c in trials[curr_run] if c[0] == trial][0]
            trialgrp.attrs['run'] = curr_run
            trialgrp.attrs['acquisition_dir'] = acquisition_dir
            trialgrp.attrs['creation_date'] = datestamp
        else:
            trialgrp = mapfile[trial]

        for imgkey in trialmaps[trial].keys():
            imap = trialgrp.create_dataset(imgkey, trialmaps[trial][imgkey].shape, trialmaps[trial][imgkey].dtype)
            imap[...] = trialmaps[trial][imgkey]
            if imgkey == 'phase':
                imap.attrs['threshold'] = threshold
                imap.attrs['kernel'] = radius
except Exception as e:
    print "Error: TRIAL %s, IMG %s" % (trial, imgkey)
    traceback.print_exc()
finally:
    mapfile.close()

#%%
aztrial = 'trial002'
eltrial = 'trial001'

surface = trialmaps[aztrial]['surface']
az_phase = trialmaps[aztrial]['phase']
el_phase = trialmaps[eltrial]['phase']
az_ratio = trialmaps[aztrial]['ratio']
el_ratio = trialmaps[eltrial]['ratio']

#%% # Smooth maps with circular kernel:
az_phasemap_filt = smooth_maps(az_phase, radius=radius)
el_phasemap_filt = smooth_maps(el_phase, radius=radius)

#pl.figure(figsize=(10,5))
#pl.subplot(1,2,1); pl.imshow(az_phasemap_filt, cmap=cmap, vmin=phasemin, vmax=phasemax); pl.axis('off')
#pl.subplot(1,2,2); pl.imshow(el_phasemap_filt, cmap=cmap, vmin=phasemin, vmax=phasemax); pl.axis('off')
#pl.suptitle('Phase maps, smoothed')
#pl.savefig(os.path.join(output_figdir, 'az_el_smoothed_rad%i.png' % radius))
#pl.close()

#% # Threhsold smoothed maps:
az_phase_thresh = threshold_map(az_phasemap_filt, az_ratio, thr=threshold)
el_phase_thresh = threshold_map(el_phasemap_filt, el_ratio, thr=threshold)

pl.figure(figsize=(10,5))
pl.subplot(1,2,1); pl.imshow(az_phase_thresh, cmap=cmap, vmin=phasemin, vmax=phasemax); pl.axis('off')
pl.subplot(1,2,2); pl.imshow(el_phase_thresh, cmap=cmap, vmin=phasemin, vmax=phasemax); pl.axis('off')
pl.suptitle('Phase maps, smoothed + thresholded')
pl.savefig(os.path.join(output_figdir, 'az_el_smoothed_rad%i_thr%0.3f.png' % (radius, threshold)))
pl.close()

#%% Get screen info:
params_path = os.path.join(acquisition_dir, curr_run, 'parameters.json')
with open(params_path, 'r') as f:
    params = json.load(f)

width = params['screen']['width_cm']
height = params['screen']['height_cm']
#width = 102.87
#height = 57.86

resolution = params['screen']['resolution']


interval = 10
linminW, linmaxW, linminH, linmaxH = get_legends(width, height, resolution, interval=interval, contours=True)


#%% #### Set contour params:

linmin = linminW #lin_coord_x.min()
linmax = linmaxW #lin_coord_x.max()

fontsize = 24
linecolor = 'gray'
linewidth = 60

# #### Plot AZ contours:
plot_contour_maps(az_phase_thresh, width, height, resolution, cond='AZ', surface=surface, output_figdir=output_figdir, interval=interval, cmap=cmap)
plot_contour_maps(el_phase_thresh, width, height, resolution, cond='EL', surface=surface, output_figdir=output_figdir, interval=interval, cmap=cmap)

#%% Convert Phase maps to screen coords:


az_phase_linear = phase_to_linear(az_phase_thresh, new_min=linminW, new_max=linmaxW, orig_min=0, orig_max=2*np.pi)
el_phase_linear = phase_to_linear(el_phase_thresh, new_min=linminH, new_max=linmaxH, orig_min=0, orig_max=2*np.pi)

az_linear_min = np.nanmin(az_phase_linear)
az_linear_max = np.nanmax(az_phase_linear)

# Get limits
el_linear_min = np.nanmin(el_phase_linear)
el_linear_max = np.nanmax(el_phase_linear)

print "********************************************"
print "AZ limits: [%.2f, %.2f]" % (az_linear_min, az_linear_max)
print "EL limits: [%.2f, %.2f]" % (el_linear_min, el_linear_max)
print "********************************************"


#%% ### Plot overlaid contour lines:
