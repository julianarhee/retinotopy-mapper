#wrapper script for calling stim presentation
import os


outPath='./outputFiles/'


# os.system("python getSurface.py -i JC026 -S 20181212\
# 	--save-images\
# 	 --output-path "+outPath)



os.system("python Retinotopy_phaseEncoding_imageBar_constantImage.py  -i JC026 -S 20181212\
 	--save-images\
 	 --output-path "+outPath)


