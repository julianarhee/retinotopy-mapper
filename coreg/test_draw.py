import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import math

class Painter(object):
    def __init__(self, ax, img):
        self.showverts = True
        self.figure = plt.figure(1)
        self.button_pressed = False
        self.img = img
        self.brush_size = 50
        self.ax = ax
        self.color = 255

        canvas = self.figure.canvas
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.on_move)

    def button_press_callback(self, event):
        if(event.button == 1):
            self.button_pressed = True
            x = int(math.floor(event.xdata))
            y = int(math.floor(event.ydata))
            cv2.circle(self.img, (x, y), int(self.brush_size / 2), (self.color, self.color, self.color), -1)
            #update the image

    def button_release_callback(self, event):
        self.button_pressed = False
        self.ax.images.pop()
        self.ax.imshow(self.img, interpolation='nearest', alpha=0.6)
        plt.draw()
        cv2.imwrite('test.png', self.img)

    def on_move(self, event):
        if(self.button_pressed):
            x = int(math.floor(event.xdata))
            y = int(math.floor(event.ydata))
            cv2.circle(self.img, (x, y), int(self.brush_size / 2), (self.color, self.color, self.color), -1)
            #update the image

def draw_demo():
    global imgMain
    imgOver = np.zeros((717,1465,3), np.uint8)
    imgMain = mpimg.imread('./images/REF.tif')
    #imgMain = np.random.uniform(0, 255, size=(500, 500))

    ax = plt.subplot(111)
    ax.imshow(imgMain, interpolation='nearest', alpha=1)
    ax.imshow(imgOver, interpolation='nearest', alpha=0.6)

    pntr = Painter(ax, imgOver)
    plt.title('Click on the image to draw')
    plt.show()

if __name__ == '__main__':
    draw_demo()