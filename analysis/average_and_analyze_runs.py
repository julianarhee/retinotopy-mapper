import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd)
from retino_functions import * 
import optparse


def run_timecourse_analysis(options):
      root = options.rootdir
      animal_id = options.animalid
      session = options.session
      project_dir = options.project_dir
      run_list = options.run_list

      source_root = os.path.join(root,'raw_data',project_dir,animal_id,session)
      target_root = os.path.join(root,'analyzed_data',project_dir,animal_id,session)


      print(run_list)
      framerate, stimfreq = get_run_parameters(source_root, animal_id, session, run_list[0])


      interp = options.interpolate
      exclude_edges= options.exclude_edges
      rolling_mean= options.rolling_mean
      time_average = options.time_average
      if time_average is not None:
            time_average = int(time_average)
      motion = options.motion

      analyze_complete_timecourse(source_root, target_root, animal_id, session, run_list, stimfreq, framerate, \
                            interp, exclude_edges, rolling_mean, \
                            motion,time_average, groupPeriods = 2, loadCorrectedFrames=False,\
                            SDmaps=True,makeSingleRunMovies=False,makeSingleCondMovies=True)

def run_phase_analysis(options):
      root = options.rootdir
      animal_id = options.animalid
      session = options.session
      project_dir = options.project_dir
      run_list = options.run_list

      source_root = os.path.join(root,'raw_data',project_dir,animal_id,session)
      target_root = os.path.join(root,'analyzed_data',project_dir,animal_id,session)


      framerate, stimfreq = get_run_parameters(source_root, animal_id, session, run_list[0])


      interp = options.interpolate
      exclude_edges= options.exclude_edges
      rolling_mean= options.rolling_mean
      time_average = options.time_average
      if time_average is not None:
            time_average = int(time_average)
      motion = options.motion

      analyze_periodic_data_from_timecourse(source_root, target_root, animal_id, session, run_list, stimfreq, framerate, \
                            interp, exclude_edges, rolling_mean, \
                            motion,time_average, stimType = 'bar')

def extract_options(options):

      parser = optparse.OptionParser()

      # PATH opts:
      parser.add_option('-R', '--root', action='store', dest='rootdir', default='/n/coxfs01/widefield-data', help='data root dir (root project dir containing all animalids) [default: /widefield-data]')
      parser.add_option('-p', '--project', action = 'store', dest = 'project_dir', default = 'Retinotopy/phase_encoding/Images_Cartesian_Constant', help = 'project directoy [default: Retinotopy/phase_enconding/Images_Cartesian_Constant]')
      parser.add_option('-i', '--animalid', action='store', dest='animalid', default='', help='Animal ID')
      parser.add_option('-S', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
      parser.add_option('-r', '--run_list', action='callback', dest='run_list', default='',type='string',callback=get_comma_separated_args, help='comma-separated names of run dirs containing tiffs to be processed (ex: run1, run2, run3)')


      parser.add_option('-m', action='store_true', dest='motion', default=False, help="use motion corrected data")
      parser.add_option('-n', action='store_true', dest='interpolate', default=False, help="interpolate to an assumed steady frame rate")
      parser.add_option('-e', action='store_true', dest='exclude_edges', default=True, help="exclude first and last cycle of run")

      parser.add_option('-g', '--rollingmean', action='store_true', dest='rolling_mean', default=True, help='Boolean to indicate whether to subtract rolling mean from signal')
      parser.add_option('-w', '--timeaverage', action='store', dest='time_average', default=None, help='Size of time window with which to average frames (integer)')


      parser.add_option('--default', action='store_true', dest='default', default='store_false', help="Use all DEFAULT params, for params not specified by user (prevent interactive)")

      (options, args) = parser.parse_args(options)


      return options


def main(options):
      options = extract_options(options)
      print('Averaging and analyzing runs...')
      run_timecourse_analysis(options)
      print('Done!')
      print('Analyzing phase from average timecourse...')
      run_phase_analysis(options)
      print('Done!')

if __name__ == '__main__':
    main(sys.argv[1:])

