#wrapper script for calling stim presentation
import os


outPath='./outputFiles/'


os.system("python getSurface.py -i JC121 -S 20191115_high_res\
	--save-images\
	 --output-path "+outPath)



# os.system("python Retinotopy_phaseEncoding_imageBar_constantImage.py \
# 	 -i JC121 -S 20191115\
# 	--match-dim\
#  	--save-images\
#  	 --output-path "+outPath)


