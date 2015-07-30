
`fwpy` should go somewhere on your path.  You can start the psychopy UI by calling 

```
fwpy `which psychopyApp.py`

```

Yes, it's a bit convoluted

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

2.  Run your stimulus protocol with your chosen monitor:
```
python PROTOCOL.py --monitor='MONITORNAME'
```

If you forget which monitors have been set on the computer, can use the -h flag to get the list of saved monitor configs:
```
python PROTOCOL.py -h
```

3.  Imaging data will automatically be recorded, unless you specify the --no-camera option. If no pvapi camera is found, the default is the built-in camera on the computer.


Tips and Troubleshooting:

1. Error with 'gcc' during pygame install:
```
export CC='/usr/bin/gcc' 
```

2. Import error with 'cv2' module:
```
conda install opencv
```