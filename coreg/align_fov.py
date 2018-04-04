#!/usr/bin/env python2

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import math
import os
import cv2
import h5py
import traceback
import hashlib
import datetime
import re
import pylab as pl
import cPickle as pkl
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import copy
import optparse
from dateutil.parser import parse
import pprint
pp = pprint.PrettyPrinter(indent=4)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


#def click_and_crop(event, x, y, flags, param):
#    # grab references to the global variables
#    global refPt, cropping, refPt_pre, cropRO#I
#
#    # if the left mouse button was clicked, record the starting
#    # (x, y) coordinates and indicate that cropping is being
#    # performed
##     if cropROI is False:
#    if event == cv2.EVENT_LBUTTONDOWN:
#        refPt.append((x, y))
#        cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
#        cv2.imshow("image", image)
#        cv2.putText(image, '%i' % len(refPt), refPt[-1]+5, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2,cv2.LINE_AA)

#     else:
#         if event == cv2.EVENT_LBUTTONDOWN:
#             refPt = [(x, y)]
#             cropping = True

#         # check to see if the left mouse button was released
#         elif event == cv2.EVENT_LBUTTONUP:
#             # record the ending (x, y) coordinates and indicate that
#             # the cropping operation is finished
#             refPt.append((x, y))
#             cropping = False

#             # draw a rectangle around the region of interest
#             cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
#             cv2.imshow("image", image)

#         if not refPt == refPt_pre:
#             print refPt
#             refPt_pre = refPt



def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

#%%

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, refPt_pre, cropROI, image

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
#     if cropROI is False:
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, refPt[-1], 1, (0,0,255), -1)
        cv2.imshow("image", image)
        #cv2.putText(image, '%i' % len(refPt), (refPt[-1][0]-5, refPt[-1][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


def get_registration_points(sample):

    #image = copy.copy(sample)
    #refPt = []
    #cropping = False
    global refPt, image

    clone = image.copy()

    cv2.startWindowThread()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt = []

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # close all open windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return image, refPt

def plot_transforms(sampleimg, referenceimg, out, npoints, out_path='/tmp'):

    print "Making figure..."
    # plt.figure(figsize=(10,0))
    plt.figure()

    plt.subplot(221)
    plt.imshow(sampleimg, cmap='gray')
    plt.axis('off')
    plt.title('original sample')

    plt.subplot(222)
    plt.imshow(referenceimg, cmap='gray')
    plt.axis('off')
    plt.title('original reference')

    plt.subplot(223)
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.title('warped sample')

    plt.subplot(224)
    # plt.imshow(SAMPLE, cmap='gray')
    # plt.imshow(out, cmap='jet', alpha=0.2)
    merged = np.zeros((reference.shape[0], reference.shape[1], 3), dtype=np.uint8)
    merged[:,:,0] = reference #cv2.cvtColor(reference)#, cv2.COLOR_RGB2GRAY)
    merged[:,:,1] = out #cv2.cvtColor(outi) #, cv2.COLOR_RGB2GRAY)
    plt.imshow(merged)
    plt.axis('off')
    plt.title('combined')

    plt.tight_layout()

    #npoints = len(sample_pts)
    # outpath = './output'
    imname = 'warp_transforms_npoints%i.png' % (npoints)
    print imname
    plt.savefig(os.path.join(out_path, imname))
    plt.show()

#%%
def plot_merged(reference, out, npoints=None, out_path='/tmp'):
    print "Getting MERGED figure..."

    plt.figure()
    merged = np.zeros((reference.shape[0], reference.shape[1], 3), dtype=np.uint8)
    merged[:,:,0] = reference
    merged[:,:,1] = out
    plt.imshow(merged)
    plt.axis('off')

    if npoints is None:
        imname = 'overlay_ALL_FOVs'
    else:
        imname = 'overlay_npoints%i' % npoints #(sample_fn, reference_fn, npoints)

    print os.path.join(out_path, imname)
    plt.savefig(os.path.join(out_path, imname))

    plt.show()
#%%
def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'r') as f:
        buf = f.read()
        hasher.update(buf)
        filehash = hasher.hexdigest()
    return filehash


#%%
def update_alignments(alignments_filepath, FOV):
    if os.path.exists(alignments_filepath):
        alignments = h5py.File(alignments_filepath, 'a')
        new_file = False
    else:
        alignments = h5py.File(alignments_filepath, 'w')
        new_file = True

    try:
        # Save reference to ALIGNMENT file, if doesn't exist:
        if new_file is True:
            sources = alignments.create_group('source_files')
        else:
            sources = alignments['source_files']

        sessions = FOV.keys()
        for session in sessions:
            for acquisition in FOV[session].keys():
                fov_key = '%s_%s' % (session, acquisition)
                img, imghash = load_image(FOV[session][acquisition]['filepath'])
                if img is None:
                    continue
                if fov_key not in sources.keys():
                    dset = sources.create_dataset('%s' % fov_key, img.shape, img.dtype)
                    dset[...] = img
                    dset.attrs['filepath'] = FOV[session][acquisition]['filepath']
                    dset.attrs['filehash'] = FOV[session][acquisition]['filehash']
                elif fov_key in sources.keys() and not sources[fov_key].attrs['filehash'] == FOV[session][acquisition]['filehash']:
                    print "For session %s, acq %s -- different img hashes!"
                    pl.figure()
                    pl.subplot(1,2,1); pl.title('stored file'); pl.imshow(sources[fov_key], cmap='gray')
                    pl.subplot(1,2,2); pl.title('requested file'); pl.imshow(img, cmap='gray')
                    pl.show()
                    while True:
                        user_select = raw_input("Select O to overwrite with new img (requested file), or C to create new: ")
                        if user_select == 'N':
                            dset = sources[fov_key]
                            dset.attrs['filepath'] = FOV[session][acquisition]['filepath']
                            dset.attrs['filehash'] = FOV[session][acquisition]['filehash']
                            break
                        elif user_select == 'O':
                            nimgs = len([i for i in sources.keys() if fov_key in i]) + 1
                            dset = sources.create_dataset('%s_%i' % (fov_key, nimgs), img.shape, img.dtype)
                            dset[...] = img
                            dset.attrs['filepath'] = FOV[session][acquisition]['filepath']
                            dset.attrs['filehash'] = FOV[session][acquisition]['filehash']
                            break
                    pl.close()
    except Exception as e:
        print "***ERROR: Unable to update FOVs for session %s, acq %s." % (session, acquisition)
        print "FOV KEY: %s" % fov_key
        traceback.print_exc()
    finally:
        alignments.close()

    print "UPDATE COMPLETE."


#%%
def load_image(image_path):
    imghash = get_file_hash(image_path)
    image = cv2.imread(image_path)
    # Make sure images are gray-scale:
    if image is None:
        return None, None
    if len(image.shape)==2: # not RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # make it 3D
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:,:,1]
    print "Image size is: ", image.shape

    return image, imghash
#%%
def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False
#%%
def get_sample_paths(animal_dir, verbose=False):
    '''For each SESSION for current animal, return dict:

        FOV[SESSION][ACQUISITION] = '/path/to/transformed/corrected/BV/image.tif'

        NOTE:  image should be 8-bit, corrected, summed, and transformed to match
        reference FOV (for 12k-2p, this is rotated-left, flipped horizontally in Fiji).
    '''
    # Find session list, for each ACQUISITION (i.e., FOV), find the corrected, transformed, 8-bit image
    sessions = [s for s in os.listdir(animal_dir) if os.path.isdir(os.path.join(animal_dir, s)) and is_date(s)]
    if verbose is True:
        print "SESSIONS:"
        print sessions

    #non_acquisitions = ['coregistration', 'macro_fullfov', 'ROIs']
    FOV = {}
    for session in sessions:
        session_dir = os.path.join(animal_dir, session)
        acquisition_list = [a for a in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, a)) and 'anatomical' in os.listdir(os.path.join(session_dir, a))]
        if verbose is True:
            print 'SESSION: %s -- Found %i acquisitions.' % (session, len(acquisition_list))

        if len(acquisition_list) > 0:
            FOV[session] = dict((acquisition, dict()) for acquisition in acquisition_list)
        #sample_paths = dict()
        for acquisition in acquisition_list:
            if verbose is True:
                print "-- ACQ: %s" % acquisition
            curr_anatomical_filepath = None
            bv_images = [f for f in os.listdir(os.path.join(session_dir, acquisition, 'anatomical')) if f.endswith('tif')]
            if len(bv_images) == 1:
                fn = bv_images[0]
                curr_anatomical_filepath = os.path.join(session_dir, acquisition, 'anatomical', fn)
            elif len(bv_images) > 1:
                print "Found multiple anatomicals for Acq. %s, session %s." % (acquisition, session)
                for idx,imgfn in enumerate(bv_images):
                    print idx, imgfn
                while True:
                    user_selection = input('Select IDX of image file to use: ')
                    fn = bv_images[int(user_selection)]
                    confirmation = raw_input('Use file: %s?  Press Y to confirm.' % fn)
                    if confirmation == 'Y':
                        break
                curr_anatomical_filepath = os.path.join(session_dir, acquisition, 'anatomical', fn)
            else:
                print "**WARNING** No anatomical image found in session %s, for acquisition %s." % (session, acquisition)
                print "Create processed blood vessel image, transform to MACRO fov, and save to dir:\n%s" % os.path.join(session_dir, acquisition, 'anatomical')

            if curr_anatomical_filepath is not None:
                FOV[session][acquisition]['filepath'] = curr_anatomical_filepath
                FOV[session][acquisition]['filehash'] = get_file_hash(curr_anatomical_filepath)
            #sample_paths[acquisition] = curr_anatomical_filepath
    return FOV

#%%
def align_to_reference(sample, reference, outdir, sample_name='sample'):

    global refPt, image

    # GET SAMPLE:
    print "Sample that will be aligned to ref is: ", sample.shape

    refPt = []
    image = copy.copy(sample)
    sample_pts_img, sample_pts = get_registration_points(image)
    #sample_pts = copy.copy(refPt)
    npoints = len(sample_pts)
    print "GOT %i SAMPLE POINTS: " % npoints
    print sample_pts

    # Save chosen SAMPLE points:
    sample_points_path = os.path.join(outdir, 'sample_points_%s.png' % sample_name)
    cv2.imwrite(sample_points_path, sample_pts_img)
    print "Saved SAMPLE points to:\n%s" % sample_points_path


    #% GET corresponding reference points:

    refPt = []
    image = copy.copy(reference)
    ref_pts_img, reference_pts = get_registration_points(reference)
    npoints = len(reference_pts)

    # DISPLAY REF IMAGE:
    print "GOT %i reference test POINTS: " % npoints
    print reference_pts
    # Save chosen REF points:
    ref_points_path = os.path.join(outdir, 'reference_points_%s.png' % sample_name)
    cv2.imwrite(ref_points_path, ref_pts_img)
    print "Saved REFERENCE points to:\n%s" % ref_points_path


    #%
    # Use SAMPLTE and TEST points to align:
    sample_mat = np.matrix([i for i in sample_pts])
    reference_mat = np.matrix([i for i in reference_pts])
    M = transformation_from_points(reference_mat, sample_mat)

    #Re-read sample image as grayscale for warping:
    #sampleG = cv2.imread(sample_path, 0)
    #out = warp_im(SAMPLE, M, REF.shape)
    out = warp_im(sample, M, reference.shape)
    #outbig = warp_im(sample, M, [reference.shape[1], reference.shape[1]])
    # out_map = warp_im(retinomap, M, SAMPLE.shape)

#        warpinfo = dict()
#        warpinfo['reference_points'] = reference_pts #tuple(p[0] for p in reference_pts)
#        warpinfo['sample_points'] = sample_pts #tuple(p[0] for p in sample_pts)
#        warpinfo['transform_mat'] = M # tuple(p[1] for p in sample_pts)
#        warpinfo['warped_sample'] = out

    coreg_info = dict()
#        coreg_info['reference_file'] = reference_path
#        coreg_info['sample_file'] = sample_path
    coreg_info['reference_points_x'] = tuple(p[0] for p in reference_pts)
    coreg_info['reference_points_y'] = tuple(p[1] for p in reference_pts)
    coreg_info['sample_points_x'] = tuple(p[0] for p in sample_pts)
    coreg_info['sample_points_y'] = tuple(p[1] for p in sample_pts)
    #coreg_info['transform_mat'] = M

    coreg_hash = hash(frozenset(coreg_info.items()))
    print "COREG HASH: %s" % coreg_hash

    return M, out, coreg_info, coreg_hash

#%%

parser = optparse.OptionParser()
parser.add_option('-D', '--root', action='store', dest='rootdir', default='/nas/volume1/2photon/data', help='data root dir (root project dir containing all animalids) [default: /nas/volume1/2photon/data, /n/coxfs01/2pdata if --slurm]')
parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
parser.add_option('--new', action='store_true', dest='align_new', default=False, help='Flag if there is a new FOV to align')
parser.add_option('--merge', action='store_true', dest='merge_all', default=False, help="Flag to create merged image of ALL found FOVs for current animal")
#parser.add_option('-R', '--run', action='store', dest='curr_run', default='', help="custom run name [e.g., 'barflash']")

#parser.add_option('-r', '--reference', action="store", dest="reference",
#                  default="", help="Path to reference image (to align to)")
#parser.add_option('-s', '--sample', action="store", dest="sample",
#                  default="", help="Path to sample image (to align to the reference")
#parser.add_option('-o', '--outpath', action="store", dest="outpath",
#                  default="/tmp", help="Path to the save ROIs")
#parser.add_option('--C', '--crop', action="store_true", dest="crop", default=False, help="Path to save ROI")

(options, args) = parser.parse_args()


#%%

# Get paths from options:
#reference_path = options.reference
#sample_path = options.sample
#out_path = options.outpath
rootdir = options.rootdir
animalid = options.animalid

align_new = options.align_new
merge_all = options.merge_all

#%%
#session = options.session
#acquisition = options.acquisition
#%%
#reference_path = '/nas/volume1/2photon/data/CE074/20180215/coregistration/window.tif'
#sample_path = '/nas/volume1/2photon/data/CE074/20180215/coregistration/V1_surface_sum_transformed.tif'
#out_path = '/nas/volume1/2photon/data/CE074/20180215/coregistration/output'

#%%

# First check animal dir to see if coregistration info already exists:
animal_dir = os.path.join(rootdir, animalid)
coreg_dir = os.path.join(animal_dir, 'coregistration')
if not os.path.exists(coreg_dir):
    os.makedirs(coreg_dir)

alignments_filepath = os.path.join(coreg_dir, 'alignments.hdf5')
if os.path.exists(alignments_filepath):
    alignments = h5py.File(alignments_filepath, 'a')
    new_file = False
else:
    alignments = h5py.File(alignments_filepath, 'w')
    new_file = True

# Get reference image -- look in "macro_fov" dir:
reference_path = os.path.join(coreg_dir, 'REFERENCE.tif')
try:
    reference, refhash = load_image(reference_path)

    # Save reference to ALIGNMENT file, if doesn't exist:
    if new_file is True:
        sources = alignments.create_group('source_files')
    else:
        sources = alignments['source_files']
    if 'reference' not in sources.keys():
        ref = sources.create_dataset('reference', reference.shape, reference.dtype)
        ref[...] = reference
        ref.attrs['filepath'] = reference_path
        ref.attrs['filehash'] = refhash
    else:
        ref = sources['reference']

    # Get list of paths to anatomical img for each acquisition:
    FOV = get_sample_paths(animal_dir, verbose=False)
    print "Coregistering %i acquisitions." % len(FOV.keys())
    pp.pprint(FOV)

    update_alignments(alignments_filepath, FOV)

except Exception as e:
    if not os.path.exists(reference_path):
        print "***ERROR: Unable to find REFERENCE.tif"
        print "Save 8-bit gray-scale image 'REFERENCE.tif' to:\n%s" % reference_path
    traceback.print_exc()
    print "Aborting."
    print "-------------------------------------------"
finally:
    alignments.close()


#%%


#%%
#coreg_info = dict()
#coreg_info['reference_file'] = reference_path
#coreg_info['sample_file'] = sample_path
#coreg_info['reference_points_x'] = tuple(p[0] for p in reference_pts)
#coreg_info['reference_points_y'] = tuple(p[1] for p in reference_pts)
#coreg_info['sample_points_x'] = tuple(p[0] for p in sample_pts)
#coreg_info['sample_points_y'] = tuple(p[1] for p in sample_pts)
##coreg_info['transform_mat'] = M
#
#coreg_hash = hash(frozenset(coreg_info.items()))
#print "COREG HASH: %s" % coreg_hash


if align_new == True:
    alignments = h5py.File(alignments_filepath, 'r')


    fov_keys = [k for k in alignments['source_files'].keys() if not k=='reference']
    outdir = os.path.join(coreg_dir, 'results')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for fov in fov_keys:
        curr_alignment_path = os.path.join(outdir, 'align_%s.hdf5' % fov)
        if os.path.exists(curr_alignment_path):
            results = h5py.File(curr_alignment_path, 'a')
        else:
            results = h5py.File(curr_alignment_path, 'w')
            make_new = True

        # Add coregistration info and results:
        if 'transforms' not in results.keys():
            transforms = results.create_group('transforms')
            make_new = True
        else:
            transforms = results['transforms']
            existing_registers = sorted(transforms.keys(), key=natural_keys)
            if len(existing_registers) > 0:
                while True:
                    print "Found existing transforms:"
                    for i, trans in enumerate(sorted(existing_registers, key=natural_keys)):
                        print i, trans
                    user_choice = raw_input('Create new alignment for fov: %s?\nPress <Y> to create new, or <n> to continue: ' % fov)
                    if user_choice == 'Y':
                        make_new = True
                        break
                    elif user_choice == 'n':
                        make_new = False
                        break
            else:
                make_new = True
        try:
            if make_new is True:
                if 'reference' not in results.keys():
                    refimg = alignments['source_files']['reference'][:]
                    reference = results.create_dataset('reference', refimg.shape, refimg.dtype)
                    reference[...] = refimg
                else:
                    reference = results['reference']
                if 'sample' not in results.keys():
                    fovimg = alignments['source_files'][fov][:]
                    sample = results.create_dataset('sample', fovimg.shape, fovimg.dtype)
                    sample[...] = fovimg
                else:
                    sample = results['sample']

                print "Current FOV: %s" % fov
                M, out, coreg_info, coreg_hash = align_to_reference(sample[:], reference[:], outdir, sample_name=fov)
                npoints = len(coreg_info['sample_points_x'])

                if coreg_hash not in transforms.keys():
                    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
                    transf_key = "%s_%s" % (fov, tstamp)
                    match = transforms.create_dataset(transf_key, M.shape, M.dtype)
                    match[...] = M
                    for info in coreg_info.keys():
                        match.attrs[info] = coreg_info[info]
                    match.attrs['coreg_hash'] = coreg_hash

                # Plot figures:
                curr_outdir = os.path.join(outdir, 'figures_%s' % fov)
                if not os.path.exists(curr_outdir):
                    os.makedirs(curr_outdir)
                plot_transforms(sample[:], reference[:], out, npoints, out_path=curr_outdir)
                plot_merged(reference[:], out, npoints, out_path=curr_outdir)
            else:
                continue

        except Exception as e:
            print "***ERROR in aligning sample to reference."
            print "--- SAMPLE: %s" % fov
            traceback.print_exc()
            print "----------------------------------------"
        finally:
            results.close()

#%%

if merge_all is True:
    alignments = h5py.File(alignments_filepath, 'r')
    reference = alignments['source_files']['reference']
    fov_files = [os.path.join(coreg_dir, 'results', f) for f in os.listdir(os.path.join(coreg_dir, 'results')) if f.endswith('hdf5')]
    for fov_fn in fov_files:
        results = h5py.File(fov_fn, 'a')
        try:
            transforms = results['transforms']
            sample = results['sample'][:]
            if 'warps' not in results.keys():
                warps = results.create_group('warps')
            else:
                warps = results['warps']
            transf_keys = [k for k in transforms.keys() if len(transforms[k].attrs['sample_points_x']) > 1]
            for key in transf_keys:
                if key not in warps.keys():
                    warpim = warp_im(sample[:], transforms[key][:], reference[:].shape)
                    dset = warps.create_dataset(key, warpim.shape, warpim.dtype)
                    dset[...] = warpim
        except Exception as e:
            print "***ERROR warping transform."
            print "--- FOV: %s" % fov_fn
            traceback.print_exc()
        finally:
            results.close()

    overlay = np.zeros(reference.shape, reference.dtype)
    min_vals = []
    max_vals = []
    for fov_fn in fov_files:
        results = h5py.File(fov_fn, 'r')

        try:
            if len(results['warps'].keys()) > 1:
                warp_keys = sorted(results['warps'].keys(), key=natural_keys) # For now, just take most recent...
                print "Found %i transforms for %s" % (len(warp_keys), fov_fn)
                print "Taking the most recent one..."
                warp_key = warp_keys[-1]
            else:
                warp_key = results['warps'].keys()[0]

            curr_warp = results['warps'][warp_key][:]
            #pl.figure(); pl.imshow(curr_warp); pl.colorbar()
            min_vals.append(curr_warp.min())
            max_vals.append(curr_warp.max())
            overlay += curr_warp
        except Exception as e:
            print "---- Error combining FOV to reference."
            print "---- Curr file: %s" % fov_fn
            traceback.print_exc()
        finally:
            results.close()


    plot_merged(reference, overlay, npoints=None, out_path=coreg_dir)
#
#
#except Exception as e:
#    print "ERROR saving results to coreg hdf5."
#    print "Sample: %s" % sample_path
#    print "Npoints: %i" % npoints
#    traceback.print_exc()
#finally:
#    results.close()
#
#
#T = dict()
#T['tMAT'] = M
#T['sample'] = sample_path
#T['reference'] = reference_path
##
##mat_fn = 'temp2sample'+'.mat'
### scipy.io.savemat(os.path.join(source_dir, condition, tif_fn), mdict=pydict)
##
##import scipy.io
##scipy.io.savemat(os.path.join(out_path, mat_fn), mdict=T)
##print os.path.join(out_path, 'mw_data', mat_fn)
##
#pkl_fn = 'temp2sample'+'.pkl'
#with open(os.path.join(out_path,pkl_fn), 'wb') as fn:
#    pkl.dump(T, fn)


 #%% PLOT:

#plot_transforms(sampleimg, referenceimg, out, out_path=out_path)
#plot_merged(reference, out, out_path=out_path)

#%%
## PAD images to the same size so that can draw overlay:
#maxreference = max(reference.shape)
#maxsample = max(sample.shape)
#maxdim = max([maxsample, maxreference])
#print "MAXDIM:", maxdim
#overlay = np.zeros((maxdim, maxdim, 3), dtype=np.uint8)
#
#adjust_dims_reference = maxdim - np.array(reference.shape)
#adjust_dims_reference_idx = np.where(adjust_dims_reference!=0)[0]
#print "Ref, adjust dims by: ", adjust_dims_reference
#print "Ref, adjust dim idxs: ", adjust_dims_reference_idx
#adjust_dims_sample = maxdim - np.array(outbig.shape)
#adjust_dims_sample_idx = np.where(adjust_dims_sample!=0)[0]
#print "Sample, adjust dims by: ", adjust_dims_sample
#print "Sample, adjust dims by: ", adjust_dims_sample_idx
#if adjust_dims_reference_idx==0:
#    temppad = np.pad(reference, ((0, maxdim-reference.shape[0]), (0,0)), 'constant')
#else:
#    temppad = np.pad(reference, ((0,0), (maxdim-reference.shape[1], 0)), 'constant')
#if len(adjust_dims_sample_idx)>0:
#    if adjust_dims_sample_idx==0:
#	outpad = np.pad(outbig, ((0, maxdim-outbig.shape[0]), (0,0)), 'constant')
#    else:
#	outpad = np.pad(outbig, ((0,0), (maxdim-outbig.shape[1], 0)), 'constant')
#else:
#    outpad = np.copy(outbig)
#print "temppad: ", temppad.shape
#print "outpad: ", outpad.shape
#
#overlay[:,:,0] = temppad
#overlay[:,:,1] = outpad
#
#plt.figure()
#plt.imshow(overlay)
#plt.axis('off')
#imname = 'merged_npoints%i.png' % npoints
#
#plt.savefig(os.path.join(out_path, imname))
##temppad = np.pad(temp, ((), ()), 'constant')
#



#
## Overlay phase map
#print "Displaying PHASE map onto figure..."
##out_map = np.ma.masked_where(out == 0, out)
##out_map_mask = np.ma.masked_where(out_map == 0, out_map)
#print rmap.shape
##gray = cv2.cvtColor(outpad, cv2.COLOR_RGB2GRAY)
#
##out_mask = np.full((out.shape[0], out.shape[1]), 0, dtype=np.uint8)
#mask1 = np.full((out.shape[0], out.shape[1]), 0, dtype=np.uint8)
#print "mask1 shape:", mask1.shape
#mask1[np.where(outpad>0)] = 255
#fg = cv2.bitwise_or(out, out, mask=mask1)
#mask1 = cv2.bitwise_not(mask1)
#background = np.full(out.shape, 255, dtype=np.uint8)
#bk = cv2.bitwise_or(background, background, mask=mask1)
#final=cv2.bitwise_or(fg, bk)
#
#plt.figure()
#plt.imshow(reference)
#plt.imshow(final, alpha=.5555)
#if no_map is False:
#    plt.imshow(rmap, cmap=colormap, alpha=0.5)
#plt.show()
##=======
##out_map = np.ma.masked_where(out == 0, out)
##out_map_mask = np.ma.masked_where(out_map == 0, out_map)
##
##plt.figure()
#merged_gray = cv2.cvtColor(merged, cv2.COLOR_RGB2GRAY)
#print "merged map:", merged_gray.shape
#print "retino map: ", rmap.shape
##plt.imshow(reference, cmap='gray', alpha=0.5)
#rmap_rgb = cv2.cvtColor(rmap, cv2.COLOR_RGB2BGR)
##plt.imshow(merged_gray, cmap='gray', alpha=1) #out_map_mask, cmap='gray', alpha=.5)
##plt.imshow(reference, cmap='gray', alpha=1.0)
##plt.imshow(rmap_rgb, cmap=colormap, alpha=0.8, vmin=0, vmax=math.pi*2)
#plt.imshow(out, cmap='gray', alpha=0.7)
##plt.imshow(merged_gray, cmap='gray', alpha=0.5) #out_map_mask, cmap='gray', alpha=.5)
##plt.imshow(rmap_rgb, cmap=colormap, alpha=0.5, vmin=0, vmax=math.pi*2)
##plt.imshow(out_map_mask, cmap='gray', alpha=.75)
#
#plt.axis('off')
#plt.show()
#imname = 'points_svd_S-%s_T-%s_npoints-%i_RETINO' % (sample_fn, reference_fn, npoints)
#print os.path.join(out_path, imname)
#plt.savefig(os.path.join(out_path, imname))
#
