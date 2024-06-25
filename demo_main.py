import asyncio
import logging
import random
import socket
import sys
import threading
import time
from dataclasses import datadataclasses
from os.path import join

# Third-party imports for data handling and numerical operations
import numpy as np
import pandas as pd
import joblib

# Imports for time series transformation
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.quiver import Quiver
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Signal processing and filtering
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt
from modules.digitalsignalprocessing import *

# Tensorflow for machine learning models
import tensorflow as tf

# Networking and real-time data handling
import dotenv
from g3pylib import connect_to_glasses

# Qt for GUI applications
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow

# IMU data fusion and processing
import imufusion


# NOTE: CKPT_PATH is subject to change

CKPT_PATH = 'checkpoint'

OUTPUT_CSV = 'test.csv'

# bvp to workload model
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
acc = False      # 3-axis acceleration
bvp = True      # Blood Volume Pulse
gsr = False      # Galvanic Skin Response (Electrodermal Activity)
tmp = False      # Temperature

serverAddress = '127.0.0.1'
serverPort = 28000
bufferSize = 4096

# E4 device ID
deviceID = '6A535C'
CKPT_PATH = r'C:\Users\11550\OneDrive\Documents\project\ARIA\aria-bvp-mwl\60_30__cl_1__pt_hpo__ft_0__y_wtlx_lbl__seed_42__weights\S28\S28_ckpt'

print(CKPT_PATH)
classifier = cnn1d(n_classes=3, ckpt_path=CKPT_PATH)


class RealTimePlotter(QMainWindow):
    """Class to plot real-time data from various sensors in a multi-subplot configuration."""

    def __init__(self):
        super().__init__()
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))  # Canvas for plotting
        self.setCentralWidget(self.canvas)
        self.axes = self.canvas.figure.subplots(2, 2)  # Creates a 2x2 grid of subplots

        # Data lists for storing sensor readings
        self.x_data, self.y_data, self.z_data = [], [], []
        self.bvp_data, self.gsr_data, self.temp_data = [], [], []

        # Line objects for updating plots
        self.lines_acc_x = self.axes[0, 0].plot([], [], label='ACC X')[0]
        self.lines_acc_y = self.axes[0, 0].plot([], [], label='ACC Y')[0]
        self.lines_acc_z = self.axes[0, 0].plot([], [], label='ACC Z')[0]
        self.lines_bvp = self.axes[0, 1].plot([], [], label='BVP')[0]
        self.lines_gsr = self.axes[1, 0].plot([], [], label='GSR')[0]
        self.lines_temp = self.axes[1, 1].plot([], [], label='Temperature')[0]

        # Placeholder for real-time data buffer
        self.bvp_win = np.zeros((64 * 60, 1), dtype=np.float32)

        # Add legends to all subplots for clarity
        for ax in self.axes.flat:
            ax.legend(loc='upper right')

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

    def update_acc_plot(self, data):
        """Updates the accelerometer plots with new data."""
        # Append new data
        self.x_data.append(data[0])
        self.y_data.append(data[1])
        self.z_data.append(data[2])

        # Remove old data to keep plot size constant
        if len(self.x_data) > 20:
            self.x_data.pop(0)
            self.y_data.pop(0)
            self.z_data.pop(0)

        # Update plots
        self.lines_acc_x.set_data(range(len(self.x_data)), self.x_data)
        self.lines_acc_y.set_data(range(len(self.y_data)), self.y_data)
        self.lines_acc_z.set_data(range(len(self.z_data)), self.z_data)

        # Redraw the plot
        for ax in self.axes.flat[:3]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

    def update_bvp_plot(self, data):
        """Updates the BVP plot with new data."""
        self.bvp_data += data
        if len(self.bvp_data) > 200:
            self.bvp_data = self.bvp_data[-200:]

        self.lines_bvp.set_data(range(len(self.bvp_data)), self.bvp_data)

        self.axes[0, 1].relim()
        self.axes[0, 1].autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

    def update_gsr_plot(self, data):
        """Updates the GSR plot with new data."""
        self.gsr_data.append(data)
        if len(self.gsr_data) > 100:
            self.gsr_data.pop(0)

        self.lines_gsr.set_data(range(len(self.gsr_data)), self.gsr_data)

        self.axes[1, 0].relim()
        self.axes[1, 0].autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

    def update_temp_plot(self, data):
        """Updates the temperature plot with new data."""
        self.temp_data.append(data)
        if len(self.temp_data) > 100:
            self.temp_data.pop(0)

        self.lines_temp.set_data(range(len(self.temp_data)), self.temp_data)

        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()

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

def generate_data():
    """Generates simulated IMU data, processes it to detect stationary periods, and computes walking speed."""
    
    start = time.time()
    imu_data = imu_update_data  # Presumed to be previously defined or externally provided

    try:
        # Generate timestamps for the data
        now = time.time() - start
        x = np.linspace(now - 2, now, 2500)

        # Prepare containers for accelerometer and gyroscope data
        y, y2, y3, g1, g2, g3 = (np.zeros(2500) for _ in range(6))

        # Extract and window data from the last 2500 samples
        accelerations = [np.array([d[1]['accelerometer'][i] for d in imu_data])[-2500:] for i in range(3)]
        gyros = [np.array([d[1]['gyroscope'][i] for d in imu_data])[-2500:] for i in range(3)]
        y, y2, y3 = accelerations
        g1, g2, g3 = gyros

        # Stack the data for further processing
        data = np.stack([y, y2, y3, g1, g2, g3]).T
        data = np.expand_dims(data, 0)

        # Placeholder for IMU data, assuming a window of 5 samples for filtering
        W = 5
        imudata = data[0].transpose(1, 0)
        T = np.zeros(int(np.floor(imudata.shape[0] / W) + 1))
        zupt = np.zeros(imudata.shape[0])

        acc, gyro = imudata[:, :3], imudata[:, 3:6]
        var_a, var_w = 10, 250
        inv_a, inv_w = 1 / var_a, 1 / var_w

        # Zero-velocity update processing
        i = 0
        for k in range(0, imudata.shape[0] - W + 1, W):
            smean_a = np.mean(acc[k:k + W], axis=0)
            for s in range(k, k + W):
                a = acc[s, :]
                w = gyro[s, :]
                T[i] += inv_a * ((a - 9.81 * smean_a / LA.norm(smean_a)).dot((a - 9.81 * smean_a / LA.norm(smean_a)).T))
                T[i] += inv_w * (w.dot(w.T))
            zupt[k:k + W].fill(T[i])
            i += 1
        zupt = zupt / W

        # Update data based on zero-velocity detection
        data[:, 0, zupt < 0.5] = 0
        data[:, 2, zupt < 0.5] = 0
        data = data[:, :, -100:]

        # Analyze data variance and conditionally zero out data if thresholds are exceeded
        if np.any(np.abs(data[0, 3:6, :]) > 100) or np.var(data[0, 3:6, :]) > 1000 or np.mean(np.abs(data[0, :3, :])) > 2:
            data[:, :3, :] = 0  # Zero out accelerometer data under certain conditions

        # Calculate walking speed using accelerometer data
        y1, y2, y3 = data[0, :3, :]
        acc_data =[y1,y2,y3]
        acc_magnitude = np.sqrt(y3[-50:] ** 2)
        sampling_time = 0.01
        walking_speed = np.sum(acc_magnitude * sampling_time) / 2  # Simplified walking speed calculation

        # AHRS processing
        ahrs = imufusion.Ahrs()
        offset = imufusion.Offset(100)  # sample_rate is 100 Hz for this example
        ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU, 0.5, 2000, 10, 10, 5 * 100)
        timestamp = x
        delta_time = np.diff(timestamp, prepend=timestamp[0])
        
        # Process each sample
        euler = np.empty((len(timestamp), 3))
        for index in range(len(timestamp)):
            ahrs.update_no_magnetometer(gyro[index], acc[index] / 9.81)
    except Exception as e:
        print(e)


    workload = 1

    window.update_acc_plot(acc_data)

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
    new_data = asyncio.run(subscribe_to_signal())
    global imu_update_data
    imu_update_data = new_data


# Function to periodically update data
def fetch_data_periodically():
    while True:
        update_data()
    

def main():


    logging.basicConfig(level=logging.INFO)

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

        buffer=[]

        buffer_size=50
        print("Streaming...")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        while True:
            try:
                response = s.recv(bufferSize).decode("utf-8")
                if "connection lost to device" in response:
                    print(response.decode("utf-8"))
                    reconnect()
                    break
                samples = response.split("\n")
                for i in range(len(samples)-1):
                    stream_type = samples[i].split()[0]
                    
                    if stream_type == "E4_Bvp":
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        print(timestamp)
                        buffer.append(data)
                        if len(buffer) >= buffer_size:
                            update_bvp_plot(buffer)
                            buffer = []  # Clear the buffer after plotting
            except Exception as e:
                print(e)
    except KeyboardInterrupt:
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()




if __name__ == "__main__":
    
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

    i = 0
    
    start = time.time()

    logging.basicConfig(level=logging.INFO)
  

    connect()

    time.sleep(1)
    suscribe_to_data()

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

    timer = QTimer()
    timer.timeout.connect(generate_data)
    timer.start(100)

    sys.exit(app.exec_())



