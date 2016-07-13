#! /usr/bin/env python
import os
import wx
import numpy as nump
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.figure as fg
import matplotlib.backends.backend_wxagg as wxagg


class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Title')
        self.create_main_panel()
        self.draw_figure()
        self._is_pick_started = False
        self._picked_indices = None

    def create_main_panel(self):
        self.panel = wx.Panel(self)
        self.dpi = 100
        self.fig = fg.Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = wxagg.FigureCanvasWxAgg(self.panel, -1, self.fig)
        self.axes = self.fig.add_subplot(111)
        self.toolbar = wxagg.NavigationToolbar2WxAgg(self.canvas)
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.AddSpacer(25)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


    def draw_figure(self):
        self.axes.clear()
        self._x_data, self._y_data = [[2,3], [4,5]]
        self.axes.scatter(self._x_data, self._y_data, picker=5)
        self.canvas.draw()

    def on_exit(self, event):
        self.Destroy()

    def picked_points(self):
        if self._picked_indices is None:
            return None
        else:
            return [ [self._x_data[i], self._y_data[i]]
                    for i in self._picked_indices ]

    def on_pick(self, event):
        if not self._is_pick_started:
            self._picked_indices = []
            self._is_pick_started = True

        for index in event.ind:
            if index not in self._picked_indices:
                self._picked_indices.append(index)
        print self.picked_points()

    def on_key(self, event):
        """If the user presses the Escape key then stop picking points and
        reset the list of picked points."""
        if 'escape' == event.key:
            self._is_pick_started = False
            self._picked_indices = None
        return


if __name__ == '__main__':
    app = wx.PySimpleApp()
    app.frame = MyFrame()
    app.frame.Show()
    app.MainLoop()