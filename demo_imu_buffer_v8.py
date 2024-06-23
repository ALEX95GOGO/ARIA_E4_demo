import numpy as np
from sktime.transformations.panel.rocket import MiniRocketMultivariate
# transformed_data now contains the transformed features
# You can now export or use these features as needed

import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import asyncio
import logging
import os
import dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from g3pylib import connect_to_glasses
import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import time
from modules.digitalsignalprocessing import *
import pandas as pd
import random
from matplotlib.patches import Circle, Wedge, Rectangle
import joblib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.quiver import Quiver

import imufusion
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import threading
from collections import deque
from numpy import linalg as LA
from PyQt5.QtCore import QTimer

import socket
import time
import pylsl
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

import gc
from os.path import join
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

import tensorflow as tf
from dataclasses import dataclass

# Most functions imported from pretrain.py
#from pretrain import *


# NOTE: CKPT_PATH is subject to change

PROJ_PATH = r'C:\Users\11550\OneDrive\Documents\project\ARIA\\'
WEIGHT_DIR = 'aria-bvp-mwl'
SBJ_RUN = 'S28'

WIN_PATH = join(PROJ_PATH, WEIGHT_DIR)
CKPT_PATH = join(WIN_PATH, SBJ_RUN, f'{SBJ_RUN}_ckpt')

OUTPUT_CSV = 'test.csv'

def cnn1d(n_classes=3, lr=1e-3, ckpt_path=None):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='gelu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 2, activation='gelu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(256, 2, activation='gelu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
    )
    if ckpt_path is not None:
        model.load_weights(ckpt_path)
    return model

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=axis)
    return y

def bvp_signal_processing(data, fs:int=64):
    bp = butter_bandpass_filter(data,
                                45/60,
                                150/60, fs=fs, order=2)
    ma = movingaverage(bp, 4, axis=0)
    return ma

def movingaverage(data, window_size, axis=0, **kwargs):
    data_in = data.copy()
    return uniform_filter1d(data_in, size=window_size, mode='constant',
                            axis=axis,
                            **kwargs)

def filter_samples(data, fs=64, sd_thold=2, time_thold=0.05):
    '''
    Reject if RMS above SD_THOLD*SD for TIME_THOLD*N duration.
    SD is standard deviation
    SD_THOLD is standard deviation threshold
    TIME_THOLD is time threshold
    N is number of samples in the window

    :param data: timestamps of each sample in the window
    :type data: numpy.ndarray or numpy like array
    :param fs: sampling frequency
    :type fs: int
    :param sd_thold: standard deviation threshold
    :type sd_thold: float
    :param time_thold: time threshold, maximum number of instances where signal
        crosses the standard deviation threshold
    :type time_thold: float

    :return: True if accepted, False if rejected
    :return type: bool
    '''
    n = len(data)
    rms = np.sqrt(data**2/n)
    mu = np.mean(rms)
    sd = np.std(rms)
    mask = rms > mu + sd_thold*sd
    sect = sum(mask)
    if sect >= time_thold*n:
        return False
    else:
        return True

def bvp_mwl_classify(classifier, bvp_win, fs=64):
    '''
    Outputs MWL classification from raw BVP data 

    Process: signal filtering -> rejection criteria -> cnn1d -> output

    :param bvp_win: timestamps of each sample in the window
    :type bvp_win: numpy.ndarray or numpy like array
    :param fs: sampling frequency
    :type fs: int
    :return: MWL value from 0 (low) - 2 (high).
    :return type: int
    '''
    data = bvp_signal_processing(bvp_win, fs=fs)

    win_accept = filter_samples(data, fs=fs)

    # do not use this window
    #if not win_accept:
    #    return None
    if not win_accept:
        return 0

    if data.ndim < 3:
        data = np.expand_dims(data, axis=0)

    pred = classifier.predict(data, verbose=0)
    print(pred)
    return pred.argmax()


# SELECT DATA TO STREAM
acc = True      # 3-axis acceleration
bvp = True      # Blood Volume Pulse
gsr = True      # Galvanic Skin Response (Electrodermal Activity)
tmp = True      # Temperature

serverAddress = '127.0.0.1'
serverPort = 28000
bufferSize = 4096

deviceID = '6A535C'
CKPT_PATH = r'C:\Users\11550\OneDrive\Documents\project\ARIA\aria-bvp-mwl\60_30__cl_1__pt_hpo__ft_0__y_wtlx_lbl__seed_42__weights\S28\S28_ckpt'
#CKPT_PATH = r'C:\Users\11550\OneDrive\Documents\project\ARIA\aria-bvp-mwl\60_30__cl_1__pt_hpo__ft_0__y_wtlx_lbl__seed_42__weights\S25\S25_ckpt'
#CKPT_PATH = r'C:\Users\11550\OneDrive\Documents\project\ARIA\aria-bvp-mwl\60_30__cl_1__pt_hpo__ft_0__y_wtlx_lbl__seed_42__weights\S30\S30_ckpt'
print(CKPT_PATH)
classifier = cnn1d(n_classes=3, ckpt_path=CKPT_PATH)


class RealTimePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.setCentralWidget(self.canvas)
        
        # Create a 2x2 grid of subplots
        self.axes = self.canvas.figure.subplots(2, 2)

        #self.ax3d = self.axes[1, 2]  # This assumes you want the 3D plot in the bottom-right
        self.fig = self.canvas.figure
        
        # ax2 = self.fig.add_subplot(111, projection='3d', position=[0.15, -0.03, 0.3, 0.7])
        # ax2.set_xlabel('X-axis')
        # ax2.set_ylabel('Y-axis')
        # ax2.set_zlabel('Z-axis')
        # ax2.set_xlim([-1, 1])
        # ax2.set_ylim([-1, 1])
        # ax2.set_zlim([-1, 1])

        # # Pre-create quiver objects
        # quiver1 = quiver2 = quiver3 =None
        # quiver1 = ax2.quiver(0, 0, 0, 1, 0, 0, color='r', label='Glasses x')
        # quiver2 = ax2.quiver(0, 0, 0, 0, 1, 0, color='g', label='Glasses y')
        # quiver3 = ax2.quiver(0, 0, 0, 0, 0, 1, color='b', label='Glasses z')

        self.x_data, self.y_data, self.z_data = [], [], []
        self.bvp_data, self.gsr_data, self.temp_data = [], [], []
        # Assuming you're plotting X, Y, Z accelerometer data in the first three subplots
        ##for ax in self.axes.flat[:3]:
        #    ax.set_xlim(0, 100)  # Adjust the X-axis fixed range
        #    ax.set_ylim(-10, 10)  # Adjust the Y-axis limits according to expected accelerometer values
        
        self.lines = [
            #self.axes[0, 0].plot(self.x_data, label='ACC')[0],
            self.axes[0, 1].plot(self.y_data, label='BVP')[0],
            self.axes[1, 0].plot(self.z_data, label='GSR')[0],
            self.axes[1, 1].plot(self.z_data, label='temperature')[0],
        ]

        self.ax3 = self.axes[1, 0]  # Assuming ax4 is defined here for workload plotting
        self.ax4 = self.axes[1, 1]  # Assuming ax4 is defined here for workload plotting


        self.lines_acc_x = self.axes[0, 0].plot([], [], label='ACC X')[0]
        self.lines_acc_y = self.axes[0, 0].plot([], [], label='ACC Y')[0]
        self.lines_acc_z = self.axes[0, 0].plot([], [], label='ACC Z')[0]


        self.lines_bvp = self.axes[0, 1].plot(self.x_data, self.y_data, self.z_data)
        self.lines_gsr = self.axes[1, 0].plot(self.x_data, self.y_data, self.z_data)
        self.lines_temp = self.axes[1, 1].plot(self.x_data, self.y_data, self.z_data)

        # Optionally, use the fourth subplot (self.axes[1, 1]) for another type of data or leave it empty
        self.bvp_win = np.zeros((64*60,1),dtype=np.float32)
        # Add legends to the subplots
        for ax in self.axes.flat[:4]:
            ax.legend(loc='upper right')

    def update_acc_plot(self, data):
        # Assuming `data` is a list [x, y, z]

        #print(data[0])
        self.x_data.append(data[0])
        self.y_data.append(data[1])
        self.z_data.append(data[2])

        print(len(self.x_data))
        # If the data exceeds a certain length, start removing old data
        if len(self.x_data) >20:  # Adjust as needed
            self.x_data.pop(0)
            self.y_data.pop(0)
            self.z_data.pop(0)
        # Update each line plot with new data
        self.lines_acc_x.set_data(range(self.x_data[0].shape[0]), self.x_data[-1])
        self.lines_acc_y.set_data(range(self.y_data[0].shape[0]), self.y_data[-1])
        self.lines_acc_z.set_data(range(self.z_data[0].shape[0]), self.z_data[-1])


        #print(len(self.x_data))

        # Redraw each axis
        for ax in self.axes.flat[:3]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

        # Introduce a sleep function to increase stability
        #time.sleep(0.1)  # Adjust the sleep duration as needed


    def update_bvp_plot(self, data):
        # Assuming `data` is a single BVP value
        self.bvp_data+=data
        
        if len(self.bvp_data) > 200:  # Adjust as needed
            self.bvp_data = self.bvp_data[-200:]

        # if len(self.bvp_data) > 64*60:  # Adjust as needed
        #     self.bvp_data = self.bvp_data[-64*60:]
        # Update the BVP line plot with new data
        self.lines_bvp[0].set_data(range(len(self.bvp_data)), self.bvp_data)

        # Redraw the BVP axis
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
        
        self.canvas.draw()
        self.canvas.flush_events()

    def update_gsr_plot(self, data):
        # Assuming `data` is a single BVP value
        self.gsr_data.append(data)
        
        if len(self.gsr_data) > 100:  # Adjust as needed
            self.gsr_data.pop(0)

        # Update the BVP line plot with new data
        self.lines_gsr[0].set_data(range(len(self.gsr_data)), self.gsr_data)
        # Redraw the BVP axis
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()

    def update_temp_plot(self, data):
        # Assuming `data` is a single BVP value
        self.temp_data.append(data)
        if len(self.temp_data) > 100:  # Adjust as needed
            self.temp_data.pop(0)
        # Update the BVP line plot with new data
        self.lines_temp[0].set_data(range(len(self.temp_data)), self.temp_data)
        # Redraw the BVP axis
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()


    def update_walking_speed_plot(self, predictions):
        # Clear the current plot
        self.ax4.clear()
        # Set plot limits
        self.ax4.set_xlim(0, 1.2)  # Limits of workload levels from 1 to 4
         # Set color based on workload level
        colors = plt.cm.RdYlGn_r((4 - predictions) / 2.0)  # Inverse the color scale
        # Create horizontal bar plot
        bars = self.ax4.barh(['Speed (m/s)'], [predictions], color=[colors])
        # Set labels and titles
        #self.ax4.set_xticklabels(['1', '2', '3', '4'], fontsize=12)
        self.ax4.set_title('Walking Speed', fontsize=20)
        self.ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Redraw the updated plot
        self.canvas.draw()
        self.canvas.flush_events()

    def update_workload_plot(self, predictions):
        # Clear the current plot
        self.ax3.clear()

        # Set plot limits
        self.ax3.set_xlim(0, 4)  # Limits of workload levels from 1 to 4

        bvp_win = np.array(self.bvp_data)
        self.bvp_win = np.concatenate((self.bvp_win[len(bvp_win):],np.expand_dims(bvp_win,1)),axis=0)

        fs = 64
        if self.bvp_win.shape[0] > fs*60:  # Adjust as needed
            self.bvp_win = self.bvp_win[-fs*60:,:]

        mwl = bvp_mwl_classify(classifier, self.bvp_win, fs=fs)
        # print(bvp_win.shape)
        # print(time.perf_counter())
        print(self.bvp_win)
        print("predicted class: ", mwl)
        workload_levels = mwl
        colors = plt.cm.RdYlGn_r((4 - workload_levels) / 4.0)  # Inverse the color scale

        # Create horizontal bar plot
        bars = self.ax3.barh(['Workload'], [workload_levels], color=[colors])

        # Set labels and titles
        self.ax3.set_xticklabels(['0', '1', '2', '3'], fontsize=12)
        self.ax3.set_title('Workload Level', fontsize=20)
        self.ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Redraw the updated plot
        self.canvas.draw()
        self.canvas.flush_events()

    def vector_to_rotation_matrix(self, vector):
        # Normalize vector
        norm_vector = vector / np.linalg.norm(vector)
        # Create rotation object
        rotation = R.from_rotvec(np.cross([0, 0, 1], norm_vector) * np.arcsin(np.linalg.norm(np.cross([0, 0, 1], norm_vector))))
        return rotation.as_matrix()

    def update_quiver_plot(self, data):
        # Assume data is an array where the last slice contains the latest directional vector
        directional_vector = data[0, 0:3, -1]
        rotation_matrix = self.vector_to_rotation_matrix(directional_vector)

        # Update vectors
        vector1 = rotation_matrix @ np.array([1, 0, 0])
        vector2 = rotation_matrix @ np.array([0, 1, 0])
        vector3 = rotation_matrix @ np.array([0, 0, 1])

        # Update quiver objects
        self.quiver1.set_segments([[[0, 0, 0], vector1]])
        self.quiver2.set_segments([[[0, 0, 0], vector2]])
        self.quiver3.set_segments([[[0, 0, 0], vector3]])

        # Redraw the plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

# Simulate data update
def generate_data():

    start = 0
    #i = i + 1
    sample_rate = 100
    #import pdb; pdb.set_trace()
    #data = np.ones((6, data_length))
    try:
        imu_data = imu_update_data
        now = time.time() - start

        x = np.linspace(now-2, now, 2500)
        y = np.zeros((2500,))
        y2 = np.zeros((2500,))
        y3 = np.zeros((2500,))
        data = imu_data
        accelerations_x = [data[1]['accelerometer'][0] for data in imu_data]
        #import pdb; pdb.set_trace()
        if len(accelerations_x)>2500:
            accelerations_x=accelerations_x[-2500:]
        y[-len(accelerations_x):] = accelerations_x
        accelerations_y = [data[1]['accelerometer'][1] for data in imu_data]
        if len(accelerations_y)>2500:
            accelerations_y=accelerations_y[-2500:]
        y2[-len(accelerations_y):] = accelerations_y
        
        accelerations_z = [data[1]['accelerometer'][2] for data in imu_data]
        if len(accelerations_z)>2500:
            accelerations_z=accelerations_z[-2500:]
        y3[-len(accelerations_z):] = accelerations_z
        g1 = np.zeros((2500,))
        g2 = np.zeros((2500,))
        g3 = np.zeros((2500,))
        gyro_x = [data[1]['gyroscope'][0] for data in imu_data]
        if len(gyro_x)>2500:
            gyro_x=gyro_x[-2500:]
        g1[-len(gyro_x):] = gyro_x
        gyro_y = [data[1]['gyroscope'][1] for data in imu_data]
        if len(gyro_y)>2500:
            gyro_y=gyro_y[-2500:]
        g2[-len(gyro_y):] = gyro_y
        
        gyro_z = [data[1]['gyroscope'][2] for data in imu_data]
        if len(gyro_z)>2500:
            gyro_z=gyro_z[-2500:]
        g3[-len(gyro_z):] = gyro_z
        #let data be 10s of single axis imu sensing captured at 250Hz
        data = np.random.random((2500, 6))
        data = np.concatenate((np.expand_dims(y,1), np.expand_dims(y2,1), np.expand_dims(y3,1),np.expand_dims(g1,1), np.expand_dims(g2,1), np.expand_dims(g3,1)), axis=1)
        data = np.expand_dims(data.transpose(1,0), 0)
    except:
        data = np.expand_dims(np.zeros((6,2500)),0)
    W=5
    imudata = data[0].transpose(1,0)
    T = np.zeros(int(np.floor(imudata.shape[0]/W)+1))
    zupt = np.zeros(imudata.shape[0])
    a = np.zeros((1,3))
    w = np.zeros((1,3))
    var_a = 10
    var_w = 250
    inv_a = (1/var_a)
    inv_w = (1/var_w)
    acc = imudata[:,0:3]
    gyro = imudata[:,3:6]

    i=0
    
    for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
        smean_a = np.mean(acc[k:k+W,:],axis=0)
        for s in range(k,k+W):
            a.put([0,1,2],acc[s,:])
            w.put([0,1,2],gyro[s,:])
            T[i] += inv_a*( (a - 9.81*smean_a/LA.norm(smean_a)).dot(( a - 9.81*smean_a/LA.norm(smean_a)).T)) #acc terms
            T[i] += inv_w*( (w).dot(w.T) )
        zupt[k:k+W].fill(T[i])
        i+=1
    zupt = zupt/W
    print('data shape:')
    print(data.shape)
    #import pdb; pdb.set_trace()

    y1 = data[0,0,:]
    y2 = data[0,1,:]
    y3 = data[0,2,:] 

    data[:,0,zupt<0.5]=0
    data[:,2,zupt<0.5]=0
    #data = data[:,:,-500:]
    #data = data[:,:,-50:]
    data = data[:,:,-100:]
    x = np.arange(500)/sample_rate

    #print(acc_data)
    #print(data[0, 3, :])
    #mask = np.logical_and.reduce([data[0, 3, :] < 25, data[0, 4, :] < 25, data[0, 5, :] < 25])
    mask = np.logical_and.reduce([abs(data[0, 3, :]) < 25, abs(data[0, 4, :]) < 25, abs(data[0, 5, :]) < 25])

    print('variance')
    print(np.var(data[0, 3:6, :]))
    #if np.var(data[0, 3:6, :])>200 or abs(np.mean(data[0, 0, :]))>2 or abs(np.mean(data[0, 2, :]))>2 :
    #if np.var(data[0, 3:6, :])>150 or np.mean(abs(data[0, 0, :]))>2 or np.mean(abs(data[0, 2, :]))>5:
    #if np.var(data[0, 3:6, :])>200:
    #if np.any(np.abs(data[0, 3:6, :]) > 100):


    if np.any(np.abs(data[0, 3:6, :]) > 100) or np.var(data[0, 3:6, :])>1000 or np.mean(abs(data[0, 0, :]))>2 or np.mean(abs(data[0, 2, :]))>5:
        data[:,0,:]=0
        data[:,2,:]=0
        #print(np.var(data[0, 3, :]))
        #mask = np.logical_and.reduce([np.var(data[0, 3, :]) <25, np.var(data[0, 4, :]) <25, np.var(data[0, 5, :]) <25, np.var(data[0, 3, :]) > 2500, np.var(data[0, 4, :]) > 2500, np.var(data[0, 5, :]) > 2500])
        #mask = np.logical_and.reduce([np.var(data[0, 3, :]) < 25, np.var(data[0, 4, :]) < 25, np.var(data[0, 5, :]) < 25])


    acc_data =[y1,y2,y3]
    #acc_magnitude = np.sqrt(y1[-50:]**2 + y3[-50:]**2)
    acc_magnitude = np.sqrt(y3[-50:]**2)
    sampling_time = 0.01
    #confidence = (sum(data[0,3,mask])+sum(data[0,4,mask])+sum(data[0,5,mask]))/1000
    #walking_speed = np.sum(acc_magnitude*sampling_time)/confidence
    walking_speed = np.sum(acc_magnitude*sampling_time)/2

    ahrs = imufusion.Ahrs()
    # Instantiate algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                    0.5,  # gain
                                    2000,  # gyroscope range
                                    10,  # acceleration rejection
                                    10,  # magnetic rejection
                                    5 * sample_rate)  # recovery trigger period = 5 seconds

    # Process sensor data
    timestamp = x
    gyroscope = gyro
    accelerometer = acc/9.81
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 3))
    acceleration = np.empty((len(timestamp), 3))
    flags = np.empty((len(timestamp), 4))

    for index in range(len(timestamp)):
        gyroscope[index] = offset.update(gyroscope[index])
        #ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])
        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

        euler[index] = ahrs.quaternion.to_euler()
        #print(euler)

        
        ahrs_internal_states = ahrs.internal_states
        internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                            ahrs_internal_states.accelerometer_ignored,
                                            ahrs_internal_states.acceleration_recovery_trigger])

        acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s
    
    # print("acceleration")
    # print(acceleration)
    # import matplotlib.pyplot as pyplot

    # # Plot sensor data
    # figure, axes = pyplot.subplots(nrows=6, sharex=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})

    # figure.suptitle("Sensors data, Euler angles, and AHRS internal states")

    # # Plot Euler angles
    # axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
    # axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
    # axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
    # axes[2].set_ylabel("Degrees")
    # axes[2].grid()
    # axes[2].legend()

    # pyplot.show()

    # Identify moving periods
    is_moving = np.empty(len(timestamp))

    for index in range(len(timestamp)):
        #is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s
        is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 1  # threshold = 3 m/s/s

    margin = int(0.1 * sample_rate)  # 100 ms

    for index in range(len(timestamp) - margin):
        is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

    for index in range(len(timestamp) - 1, margin, -1):
        is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

    # Calculate velocity (includes integral drift)
    velocity = np.zeros((len(timestamp), 3))

    for index in range(len(timestamp)):
        if is_moving[index]:  # only integrate if moving
            velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

    '''
    # Find start and stop indices of each moving period
    is_moving_diff = np.diff(is_moving, append=is_moving[-1])
    @dataclass
    class IsMovingPeriod:
        start_index: int = -1
        stop_index: int = -1


    is_moving_periods = []
    is_moving_period = IsMovingPeriod()

    for index in range(len(timestamp)):
        if is_moving_period.start_index == -1:
            if is_moving_diff[index] == 1:
                is_moving_period.start_index = index

        elif is_moving_period.stop_index == -1:
            if is_moving_diff[index] == -1:
                is_moving_period.stop_index = index
                is_moving_periods.append(is_moving_period)
                is_moving_period = IsMovingPeriod()

    # Remove integral drift from velocity
    velocity_drift = np.zeros((len(timestamp), 3))

    for is_moving_period in is_moving_periods:
        start_index = is_moving_period.start_index
        stop_index = is_moving_period.stop_index

        t = [timestamp[start_index], timestamp[stop_index]]
        x = [velocity[start_index, 0], velocity[stop_index, 0]]
        y = [velocity[start_index, 1], velocity[stop_index, 1]]
        z = [velocity[start_index, 2], velocity[stop_index, 2]]

        t_new = timestamp[start_index:(stop_index + 1)]

        velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
        velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
        velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

    velocity = velocity - velocity_drift
    '''
    print('velocity')
    print(velocity)
    # #walking_speed=velocity[-1][0]
    # print(walking_speed)
    #acc_data = [random.randint(-2000, 2000) for _ in range(3)]  # Generate random accelerometer data
    #print(acc_data[0].shape)
    #temp_data = random.randint(-20, 45)  # Generate random temperature data
    workload = random.randint(1,25)  # Generate random temperature data
    #walking_speed = random.randint(0,1)  # Generate random temperature data

    window.update_acc_plot(acc_data)
    #window.update_acc_plot(gyro.T)
    #window.update_acc_plot(acceleration.T)
    try:
        window.update_bvp_plot(bvp_buffer)
    except:
        pass
    window.update_walking_speed_plot(walking_speed)
    window.update_workload_plot([workload])



def euler_to_rot_matrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])

    return np.dot(R_z, np.dot(R_y, R_x))


# Function to rotate a vector around an axis
def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    rotation_matrix = np.array([[np.cos(angle) + axis[0]**2 * (1 - np.cos(angle)),
                                axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                                axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                               [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                                np.cos(angle) + axis[1]**2 * (1 - np.cos(angle)),
                                axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                               [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                                axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                                np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))]])

    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points
def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


async def subscribe_to_signal():
    async with connect_to_glasses.with_hostname(G3_HOSTNAME) as g3:
        imu_queue, unsubscribe = await g3.rudimentary.subscribe_to_imu()

        async def imu_receiver():
            count = 0
            while True:
                imu_message = await imu_queue.get()
                #print(imu_message)
                if 'accelerometer' in imu_message[1].keys():
                    imu_data.append(imu_message)  # Store the IMU data
                count += 1
                
                #if count % 300 == 0:
                #    logging.info(f"Received {count} IMU messages")
                #    logging.info(f"IMU message snapshot: {imu_message}")
                imu_queue.task_done()
            print(count)
        await g3.rudimentary.start_streams()
        receiver = asyncio.create_task(imu_receiver(), name="imu_receiver")
        await asyncio.sleep(0.5)
        #await asyncio.sleep(1.5)
        #await asyncio.sleep(1200)
        await g3.rudimentary.stop_streams()
        await imu_queue.join()
        receiver.cancel()
        await unsubscribe
    return imu_data



def gauge(labels=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
          colors = [
    'darkred', 'firebrick', 'indianred', 'lightcoral', 'pink',
    'lavender', 'aliceblue', 'lightskyblue', 'lightsteelblue', 'powderblue',
    'lightblue', 'skyblue', 'dodgerblue', 'deepskyblue', 'cornflowerblue', 'royalblue', 'blue', 'mediumblue', 'darkblue', 'navy'
],
          arrow="", 
          title="", 
          fname=False):     
    
    """
    some sanity checks first
    
    """
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N)) 
 

    """
    begins the plotting
    """
    
    # Reorder the colors to resemble the viridis colormap
    viridis_order = [0, 1, 2, 5, 4, 3, 6, 9, 8, 7, 10, 13, 12, 11, 14]
    ordered_colors = [colors[i] for i in viridis_order]

    print(ordered_colors)
    #fig.subplots_adjust(0,0,2,1)

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4,*ang, facecolor='w', lw=2 ))
        # arcs
        patches.append(Wedge((0.,0.), .4,*ang, width=0.2, facecolor=c, lw=2, alpha=0.5,))
    
    [ax3.add_patch(p) for p in patches]

    
    """
    set the labels
    """

    for mid, lab in zip(mid_points, labels): 

        ax3.text(0.42 * np.cos(np.radians(mid)), 0.42 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=20, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax3.add_patch(r)
    

    
    ax3.text(0, -0.1, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=18 )

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax3.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax3.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax3.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax3.set_frame_on(False)
    ax3.axes.set_xticks([])
    ax3.axes.set_yticks([])
    ax3.axis('equal')


# Create ax4 for the animated horizontal bar plot


# Function to update the horizontal bar plot
def update_bar(arrow_value):
    bars4[0].set_width(arrow_value)

def normalize_vector(v):
    """ Normalize a 3D vector. """
    return v / np.linalg.norm(v)

def vector_to_rotation_matrix(vector, align_with=np.array([0, 0, 1])):
    """ Convert a 3D directional vector to a rotation matrix. """
    # Normalize the input vector and the aligning vector
    v = normalize_vector(vector)
    w = normalize_vector(align_with)

    # Calculate the rotation axis (cross product)
    axis = np.cross(w, v)
    axis_length = np.linalg.norm(axis)
    if axis_length == 0:
        # The vectors are already aligned
        return np.identity(3)

    # Normalize the axis
    axis = axis / axis_length

    # Calculate the rotation angle
    angle = np.arccos(np.dot(v, w))

    # Rodrigues' rotation formula
    skew = np.array([[0, -axis[2], axis[1]], 
                     [axis[2], 0, -axis[0]], 
                     [-axis[1], axis[0], 0]])
    rotation_matrix = np.identity(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * np.dot(skew, skew)

    return rotation_matrix

# Function to run asyncio in a separate thread
def update_data():
    #await asyncio.run(subscribe_to_signal())
    new_data = asyncio.run(subscribe_to_signal())
    # Update global data array
    # Assume the range for the sensor readings is from -10 to 10 for both sensors
    # new_data = {{},
    #     'gyroscope': np.random.uniform(-10, 10, (3,500)).astype(np.float32),
    #     'accelerometer': np.random.uniform(-10, 10, (3,500)).astype(np.float32)
    # }

    # new_data = np.ones((6,500), dtype=np.float32)
    global imu_update_data
    imu_update_data = new_data
    #data = np.roll(data, -1, axis=1)
    #data[:, -1] = new_data

# Function to periodically update data
def fetch_data_periodically():
    while True:
        update_data()
    

def main():

    # Start a separate thread to fetch data
    #data_thread = threading.Thread(target=fetch_data_periodically, daemon=True)
    #data_thread.start()

    time.sleep(0.1)
    i = 0
    

    start = time.time()

    logging.basicConfig(level=logging.INFO)

    
    # Number of data points for the spider plot
    num_points = 6
    n_batches = 1
    n_features = 6  # (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    n_window_steps = 500  # Adjust this based on your actual data
    sample_rate = 100

    app = QApplication(sys.argv)
    window = RealTimePlotter()
    window.show()

    # Setup timer to update plots every 100 milliseconds
    timer = QTimer()
    timer.timeout.connect(generate_data)
    timer.start(500)

    sys.exit(app.exec_())

async def subscribe_to_signal():
    async with connect_to_glasses.with_hostname(G3_HOSTNAME) as g3:
        imu_queue, unsubscribe = await g3.rudimentary.subscribe_to_imu()

        async def imu_receiver():
            count = 0
            while True:
                imu_message = await imu_queue.get()
                #print(imu_message)
                if 'accelerometer' in imu_message[1].keys():
                    imu_data.append(imu_message)  # Store the IMU data
                count += 1
                
                #if count % 300 == 0:
                #    logging.info(f"Received {count} IMU messages")
                #    logging.info(f"IMU message snapshot: {imu_message}")
                imu_queue.task_done()
            print(count)
        await g3.rudimentary.start_streams()
        receiver = asyncio.create_task(imu_receiver(), name="imu_receiver")
        await asyncio.sleep(0.5)
        #await asyncio.sleep(1.5)
        #await asyncio.sleep(12)
        await g3.rudimentary.stop_streams()
        await imu_queue.join()
        receiver.cancel()
        await unsubscribe
    return imu_data

def connect():
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)

    print("Connecting to server")
    s.connect((serverAddress, serverPort))
    print("Connected to server\n")

    print("Devices available:")
    s.send("device_list\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Connecting to device")
    s.send(("device_connect " + deviceID + "\r\n").encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Pausing data receiving")
    s.send("pause ON\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))


def suscribe_to_data():
    if acc:
        print("Suscribing to ACC")
        s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if bvp:
        print("Suscribing to BVP")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if gsr:
        print("Suscribing to GSR")
        s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if tmp:
        print("Suscribing to Temp")
        s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    print("Resuming data receiving")
    s.send("pause OFF\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))



def prepare_LSL_streaming():
    print("Starting LSL streaming")
    if acc:
        infoACC = pylsl.StreamInfo('acc','ACC',3,32,'int32','ACC-empatica_e4');
        global outletACC
        outletACC = pylsl.StreamOutlet(infoACC)
    if bvp:
        infoBVP = pylsl.StreamInfo('bvp','BVP',1,64,'float32','BVP-empatica_e4');
        global outletBVP
        outletBVP = pylsl.StreamOutlet(infoBVP)
    if gsr:
        infoGSR = pylsl.StreamInfo('gsr','GSR',1,4,'float32','GSR-empatica_e4');
        global outletGSR
        outletGSR = pylsl.StreamOutlet(infoGSR)
    if tmp:
        infoTemp = pylsl.StreamInfo('tmp','Temp',1,4,'float32','Temp-empatica_e4');
        global outletTemp
        outletTemp = pylsl.StreamOutlet(infoTemp)


def reconnect():
    print("Reconnecting...")
    connect()
    suscribe_to_data()
    stream()

def update_bvp_plot(buffer):
    global bvp_buffer
    try:
        bvp_buffer = buffer.copy()
        #window.update_bvp_plot(buffer.copy())
    except Exception as e:
        print(f"Error updating plot: {e}")

def stream():
    try:
        bvp_data=[]
        #global buffer
        buffer=[]
        #plot_thread = threading.Thread(target=update_plot, args=(buffer,))
        #plot_thread.start()
        buffer_size=50
        print("Streaming...")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        #s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
        while True:
            try:
                response = s.recv(bufferSize).decode("utf-8")
                #print(response)
                if "connection lost to device" in response:
                    print(response.decode("utf-8"))
                    reconnect()
                    break
                #cleaned_response = response.replace('\r', '')
                #samples = cleaned_response.split("\n")
                samples = response.split("\n")
                #print(samples)
                for i in range(len(samples)-1):
                    #print(samples)
                    #if samples[0] =='':
                    #    samples=samples[1]
                    stream_type = samples[i].split()[0]
                    
                    if stream_type == "E4_Bvp":
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        print(timestamp)
                        buffer.append(data)
                        if len(buffer) >= buffer_size:
                            update_bvp_plot(buffer)
                            #window.update_bvp_plot(buffer)
                            buffer = []  # Clear the buffer after plotting
            except Exception as e:
                print(e)
    except KeyboardInterrupt:
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()




if __name__ == "__main__":
    

    #G3_HOSTNAME = 'tg03b-080200018761'
    G3_HOSTNAME = 'tg03b-080200027081'
    # Initialize a list to store the IMU data for plotting
    imu_data = []

    # Initialize empty lists for gyro data
    time_data = []
    gyro_data = []

    fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    # Specify ax3 for tight layout
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust rect as needed
    # Hide the original ax4 to ax6
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax4 = fig.add_subplot(325)  # You may need to adjust the position and size accordingly
    

    # Initialize MiniRocketMultivariate
    minirocket = MiniRocketMultivariate()
    
    # Initialize data array
    data_length = 100
    # Initialize the horizontal bar plots
    labels = ['Very Low', 'Low', 'Medium', 'High']
    dotenv.load_dotenv()

    
    #time.sleep(0.1)
    i = 0
    

    start = time.time()

    logging.basicConfig(level=logging.INFO)

    

    connect()

    time.sleep(1)
    suscribe_to_data()
    prepare_LSL_streaming()

    time.sleep(1)

    app = QApplication(sys.argv)
    window = RealTimePlotter()
    window.show()

    #Start a separate thread to fetch data
    data_thread = threading.Thread(target=fetch_data_periodically, daemon=True)
    data_thread.start()

    #stream()
    stream_thread = threading.Thread(target=stream, daemon=True)
    stream_thread.start()

    #data_thread.join()
    #stream_thread.join()
    # Setup timer to update plots every 100 milliseconds
    timer = QTimer()
    timer.timeout.connect(generate_data)
    #timer.timeout.connect(stream)
    timer.start(100)

    sys.exit(app.exec_())

    #main()


