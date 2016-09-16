
from subprocess import Popen
from os import mkdir

argfile = open('commandops.txt')
for number, line in enumerate(argfile):    
    # newpath = 'scatter.%03i' % number 
    # mkdir(newpath)
    cmd = 'python get_fft_bar.py ' + line.strip()
    print 'Running %r in %i' % (cmd, number)
    Popen(cmd, shell=True)

# import os, subprocess
# n = 0
# for cmd in open('commands.txt'):
#     # newpath = 'scatter.%03d' % n 
#     # os.mkdir(newpath)
#     subprocess.Popen("./get_fft_bar.py " + cmd, shell=True)
#     n += 1