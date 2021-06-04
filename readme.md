# retinotopy-mapper package
Code for retinotopic mapping of rat visual cortex using wide-field imaging. Stimulus presentation, data acquisition, and data processing uses custom Python code. All software developed for imaging experiments in the Cox Lab, Harvard University. 

NOTE:  Current (final) version in `cleanup` branch.

## Basic Setup
Install anaconda. Then, create the conda environment (python2.7)
	
		conda create env -f retino-opencv.yml
		source activate retinodev
		
## After Installing:
1.  Setup monitor.

		python setupMonitor.py
		
* This will prompt you for information about the monitor. Input the info for each question directly in the command line. 
* You will be prompted to save at the end.

2.  Run a stimulus protocol with the selected monitor.
		
		python PROTOCOL.py --monitor='MONITORNAME'
		

	If you forget which monitors have been set on the computer, you can use the -h flag to get the list of saved monitor configs:

		
		python PROTOCOL.py -h
		

3.  Neural data will be ignored with the --no-camera option. To save image data, add the --save-images option. If no pvapi camera is found, the default is the built-in camera on the computer. For now, use the png output-format option:

		python PROTOCOL.py --monitor='MONITOR-NAME' --save-images --output-format='png' --output-path='DATA-DIR-NAME'

4.  For more options, their uses, and default settings, go to help.

		
		python PROTOCOL.py -h
		
## Sources and Contributors
This package uses a phase-encoding paradigm adapted from a number of different studies in mouse visual cortex (1-4). Stimulus parameters were determined experimentally for imaging in rats. In addition to custom code based on previously published methods, we found additional guidance from [zhuangjun1981/retinotopic_mapping](https://github.com/zhuangjun1981/retinotopic_mapping) to be particularly helpful for area segmentation (2-4).

**Code contributors**
Juliana Rhee (julianarhee)
Cesar Echavarria (cechava)
David Cox (davidcox)


## Tips and Troubleshooting:

- Error with 'gcc' during pygame install:


		export CC='/usr/bin/gcc' 
		

- Import error with 'cv2' module:

		
		conda install opencv


- `fwpy` should go somewhere on your path.  You can start the psychopy UI by calling 

		fwpy `which psychopyApp.py`

## References
1. Kalatsky VA, Stryker MP (2003) New paradigm for optical imaging: temporally encoded maps of intrinsic signal. _Neuron_ 38:529-545.

2. Garrett ME, Nauhaus I, Marshel JH, Callaway EM (2014) Topography and areal organization of mouse visual cortex. _J Neurosci_ 34:12587-12600.

3. Juavinett AL, Nauhaus I, Garrett ME, Zhuang J, Callaway EM (2017). Automated identification of mouse visual areas with intrinsic signal imaging. _Nature Protocols_. 12: 32-43.

4. Zhuang J, Ng L, Williams D, Valley M, Li Y, Garrett M, Waters J (2017) An extended retinotopic map of mouse cortex. _eLife_ 6: e18372.


