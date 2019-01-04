
import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd)
print('here')
from retino_functions import * 

import optparse



def visualize_run(options):
    root = options.rootdir
    animalid = options.animalid
    project_dir = options.project_dir
    session = options.session
    run_list = options.run_list

    source_root = os.path.join(root,'raw_data',project_dir,animalid,session)
    target_root = os.path.join(root,'analyzed_data',project_dir,animalid,session)

    framerate, stimfreq = get_run_parameters(source_root, animalid, session, run_list[0])

    interp = options.interpolate
    exclude_edges= options.exclude_edges
    rolling_mean= options.rolling_mean
    time_average = options.time_average
    if time_average is not None:
        time_average = int(time_average)
    motion = options.motion

    ratio_thresh = options.ratio_thresh
    if ratio_thresh is not None:
        ratio_thresh = float(ratio_thresh)
    smooth_fwhm = options.smooth_fwhm
    if smooth_fwhm is not None:
        smooth_fwhm = int(smooth_fwhm)
    flip = options.flip

    analysis_root = os.path.join(target_root,'Analyses')
    analysis_dir=get_analysis_path_timecourse(analysis_root, interp, exclude_edges, rolling_mean, \
        motion, time_average)

    visualize_average_run(source_root, target_root, animalid, session, run_list, smooth_fwhm, ratio_thresh, analysis_dir, motion,flip)

def extract_options(options):

      parser = optparse.OptionParser()

      # PATH opts:
      parser.add_option('-R', '--root', action='store', dest='rootdir', default='/n/coxfs01/widefield-data', help='data root dir (root project dir containing all animalids) [default: /widefield]')
      parser.add_option('-p', '--project', action = 'store', dest = 'project_dir', default = 'Retinotopy/phase_encoding/Images_Cartesian_Constant', help = 'project directoy [default: Retinotopy/phase_enconding/Images_Cartesian_Constant]')
      parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
      parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD')
      parser.add_option('-r', '--run_list', action='callback', dest='run_list', default='',type='string',callback=get_comma_separated_args, help='comma-separated names of run dirs containing tiffs to be processed (ex: run1, run2, run3)')


      #specifications of analysis to visualize
      parser.add_option('-m', action='store_true', dest='motion', default=False, help="use motion corrected data")
      parser.add_option('-n', action='store_true', dest='interpolate', default=False, help="interpolate to an assumed steady frame rate")
      parser.add_option('-e', action='store_true', dest='exclude_edges', default=True, help="exclude first and last cycle of run")
      parser.add_option('-g', '--rollingmean', action='store_true', dest='rolling_mean', default=True, help='Boolean to indicate whether to subtract rolling mean from signal')
      parser.add_option('-w', '--timeaverage', action='store', dest='time_average', default=None, help='Size of time window with which to average frames (integer)')

      #visualization options
      parser.add_option('-f', '--fwhm', action='store', dest='smooth_fwhm', default=None, help='full-width at half-max size of kernel for smoothing')
      parser.add_option('-t', '--thresh', action='store', dest='ratio_thresh', default=None, help='magnitude ratio cut-off threshold')
      parser.add_option('-l', '--flip', action='store_true', dest='flip', default=False, help='boolean to indicate whether to perform horizontal flip on phase map images (to match actual orientation of FOV)')

      parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

      (options, args) = parser.parse_args(options)


      return options


def main(options):
      options = extract_options(options)
      print('Making pretty images...')
      visualize_run(options)
      print('Done!')


if __name__ == '__main__':
    main(sys.argv[1:])

