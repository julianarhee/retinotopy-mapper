

from math import atan2, degrees


def pix2deg(size_in_px, height, vres, distance):
	height = 25 # Monitor height in cm
	distance = 60 # Distance between monitor and participant in cm
	vres = 768 # Vertical resolution of the monitor
	size_in_px = 100 # The stimulus size in pixels
	# Calculate the number of degrees that correspond to a single pixel. This will
	# generally be a very small value, something like 0.03.
	deg_per_px = degrees(atan2(.5*height, distance)) / (.5*vres)
	print '%s degrees correspond to a single pixel' % deg_per_px
	# Calculate the size of the stimulus in degrees
	size_in_deg = size_in_px * deg_per_px
	print 'The size of the stimulus is %s pixels and %s visual degrees' \
		% (size_in_px, size_in_deg)

	return size_in_dev


def deg2pix(size_in_deg, height, vres, distance):
	height = 25 # Monitor height in cm
	distance = 60 # Distance between monitor and participant in cm
	vres = 768 # Vertical resolution of the monitor
	size_in_deg = 3. # The stimulus size in pixels
	# Calculate the number of degrees that correspond to a single pixel. This will
	# generally be a very small value, something like 0.03.
	deg_per_px = degrees(atan2(.5*height, distance)) / (.5*vres)
	print '%s degrees correspond to a single pixel' % deg_per_px
	# Calculate the size of the stimulus in degrees
	size_in_px = size_in_deg / deg_per_px
	print 'The size of the stimulus is %s pixels and %s visual degrees' \
		% (size_in_px, size_in_deg)

	return size_in_px