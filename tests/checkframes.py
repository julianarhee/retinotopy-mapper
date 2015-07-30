
import os

pathdir = '/Volumes/MAC/data/201517_test/V-Left'
flist = os.listdir(pathdir)
t= []
for i in flist:
    fn = pathdir + '/' + i
    with open(fn, 'rb') as fp:
        F=pkl.load(fp)
    t.append(F['time'])

interval = []
for x in range(1,len(t)):
    interval.append(int(t[x])-int(t[x-1]))

insecs = [i/1E6 for i in interval]