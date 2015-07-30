
`fwpy` should go somewhere on your path.  You can start the psychopy UI by calling 

```
fwpy `which psychopyApp.py`

```

Yes, it's a bit convoluted

# Basic Setup
Setting up the mapper on a new computer or with a new monitor (MAC):

```
pip install -r requirements.txt
```

Once everything is successfully installed, do the following:

1.  Setup the monitor.

		```
		python setupMonitor.py
		```

	This will prompt you for information about the monitor. Input the info for each question directly in the command line.
	You will be prompted to save at the end.

2.  Run a stimulus protocol with the selected monitor.

		```
		python PROTOCOL.py --monitor='MONITORNAME'
		```

	If you forget which monitors have been set on the computer, can use the -h flag to get the list of saved monitor configs:

		```
		python PROTOCOL.py -h
		```

3.  Imaging data will be ignored with the --no-camera option. To save image data, add the --save-images option. If no pvapi camera is found, the default is the built-in camera on the computer.

4.  For more options, their uses, and default settings, go to help.

		```
		python PROTOCOL.py -h
		```

# Tips and Troubleshooting:

1. Error with 'gcc' during pygame install:

		```
		export CC='/usr/bin/gcc' 
		```

2. Import error with 'cv2' module:

		```
		conda install opencv
		```