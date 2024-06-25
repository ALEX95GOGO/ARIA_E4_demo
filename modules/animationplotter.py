import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class AnimationPlotter():
    def __init__(self, data):
        self.data = data
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        self.x_lim = 2
        self.y_lim = 2
        self.z_lim = 2

        self.annotation = self.ax.annotate(
            'annotation', xy=(1,0), xytext=(-1,0),
            arrowprops={'arrowstyle': '->'}
        )
        self.marker_txt = ['1', '2', '3', '4', '5', '6', '7']

    def set_lim(self):
        self.ax.set_xlim([-self.x_lim, self.x_lim])
        self.ax.set_ylim([-self.y_lim, self.y_lim])
        self.ax.set_zlim([-self.z_lim, self.z_lim])

    def animate(self, i):
        self.ax.clear()
        coords = self.data[i, :, :]
        self.ax.scatter(coords[:,0], coords[:,1], coords[:,2], marker='o')
        for j, txt in enumerate(self.marker_txt):
            self.ax.text(coords[j, 0], coords[j, 1], coords[j, 2], txt,
                         fontsize=12)
        self.ax.set_xlabel('x axis')
        self.ax.set_ylabel('y axis')
        self.ax.set_zlabel('z axis')

    def run(self):
        ani = FuncAnimation(self.fig, self.animate, frames=len(self.data)-1,
                            repeat=False, interval=20)
        plt.show()
        return ani

class AnimationPlotter2D():
    def __init__(self, buff_len, nlines=2):
        self.buff_len = buff_len
        self.nlines = nlines

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.x_lim = [0, buff_len]
        self.y_lim = [-3, 3]

        self.ax.set_xlabel("index")
        self.ax.set_ylabel("signal magnitude")

        self.lines=[]
        for n in range(nlines):
            l, = self.ax.plot(np.arange(0, buff_len), np.zeros(buff_len))
            self.lines.append(l)
        self.ax.legend(['est', 'lbl'])
        self.set_lim()

    def set_lim(self):
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

    def set_legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def animate(self, ys):
        # self.ax.clear()
        for n in range(self.nlines):
            self.lines[n].set_ydata(ys[n])
        plt.pause(0.001)
    # def run(self):
    #     # plt.show()
    #     return ani
