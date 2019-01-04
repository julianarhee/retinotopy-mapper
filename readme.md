# Intial Setup

1. create environments. Separate environment for acquisition (retino_acq) and analysis (retino_analysis). 

```
conda create env -f [filename].yml
```

* for some reason libtiff not properly installing from retino_acq.yml file.... will troubleshoot later

# Acquisition


1. Acquiring a small stack of images at the surface.

```
python acqusition/getSurface.py -i [animal ID] -S [session] --save-images --output-path [path to folder]
```

2. Present periodic stimuli (bar with naural images) and present images

```
python acquisition/Retinotopy_phaseEncoding_imageBar_constantImage.py -i [animal ID] -S [session] --save-images --output-path [path to folder]
```

* Both of these steps can be run by calling a wrapper script edited with proper options

```
python acquisition/wrapperScript.py
```

# Analysis

1. Analyze individual runs separately

```
python analysis/analyze_runs.py -i [animal ID] -S [session] -r [comma-separated list of runs] -m [Boolean for motion correction]
-n [Boolean for interpolation of data points to constant rate] -g [Boolean for removal of rolling mean] -w [integer indicating size of boxcar window for timecourse averaging of each pixel]
```

Typical example:
```
python analysis/analyze_runs.py -i JC026 -S 20181207 -r 'run1, run2, run3, run4, run5, run6' -m True -g True -w 11
```

This script analyzes individual runs separately. It also gets the image of the surface, performs a basic quality control check so the user can make sure that there wasn't an awful lot of movement between runs or shit data was acquired. The script also performs motion registration and correction, if indicated. Outputs unsmoothed maps for each run, which can be smoothed and thresholded with the script below.


2. Visualize single run results

```
python analysis/visualize_runs.py -i [animal ID] -S [session] -r [comma-separated list of runs] -m [Boolean for motion correction]
-n [Boolean for interpolation of data points to constant rate] -g [Boolean for removal of rolling mean] -w [integer indicating size of boxcar window for timecourse averaging of each pixel] -f [full-width at half-max size of kernel for smoothing] -t [magnitude ratio threshold]
```

Typical example:

```
python analysis/visualize_runs.py -i JC026 -S 20181207 -r 'run1, run2, run3, run4, run5, run6' -m True -g True -w 11 -f 7 -t .02
```

This script smooths phase map and threshold values based on magnitude ratio values per pixel for each run.

* At this point the user can look through individual run results and choose best runs for averaging in the subsequent steps. This tends to yield maps of better quality.


3. Average multiple runs and analyze

```
python analysis/average_and_analyze_runs.py -i [animal ID] -S [session] -r [comma-separated list of runs] -m [Boolean for motion correction]
-n [Boolean for interpolation of data points to constant rate] -g [Boolean for removal of rolling mean] -w [integer indicating size of boxcar window for timecourse averaging of each pixel]
```

Typical example:

```
python analysis/average_and_analyze_runs.py -i JC026 -S 20181207 -r 'run1, run2, run3, run4' -m True -g True -w 11
```

This script averages the timecourse of multiple runs and analyzes the magnitude and phase of the resulting timecourse.  Outputs unsmoothed maps for each condition, which can be smoothed and thresholded with the script below.

4. Visualize analysis results from multiple-run average

```
python analysis/visualize_average_run.py -i [animal ID] -S [session] -r [comma-separated list of runs] -m [Boolean for motion correction]
-n [Boolean for interpolation of data points to constant rate] -g [Boolean for removal of rolling mean] -w [integer indicating size of boxcar window for timecourse averaging of each pixel] -f [full-width at half-max size of kernel for smoothing] -t [magnitude ratio threshold]
```

Typical example:

```
python analysis/visualize_average_run.py -i JC026 -S 20181207 -r 'run1, run2, run3, run4' -m True -g True -w 11 -f 7 -t .02
```

This script smooths phase map and threshold values based on magnitude ratio values per pixel for each condition.

