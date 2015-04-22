# /Users/julianarhee/anaconda/envs/retinodev/bin 
# set up a new monitor

from psychopy import visual, monitors
import re


if __name__ == '__main__':
	
	print "Enter monitor name to create new calibration: "
	monitor_name = str(raw_input())
	t = monitors.Monitor(monitor_name)

	screenW = float(raw_input("Enter WIDTH (cm): "))
	t.setWidth(screenW)
	screen_size_pix = list(input("Enter screen size (pix), [screenW, screenH]: "))
	t.setSizePix(screen_size_pix)
	distance = float(raw_input("Distance from eye to monitor (cm)? "))
	t.setDistance(distance)

	t.setCalibDate()

	print "**************************************************************"
	print "Save new calibration file for monitor, %s? [y/n]" % monitor_name
	print "**************************************************************"
	print "Properties:"
	print "-----------"
	print "screenW (cm): %i " % screenW
	print "screen size in pixels: %s" % screen_size_pix
	print "distance (cm): %f " % distance
	agree = str(raw_input())
	if re.match(r'y', agree):
		t.saveMon()

	print monitors.getAllMonitors()




