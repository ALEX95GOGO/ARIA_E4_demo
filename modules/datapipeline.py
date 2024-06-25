import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import pandas as pd
import dask.dataframe as dd
import tensorflow as tf
from os import environ, mkdir, makedirs, listdir, stat, walk
from sys import platform
from os.path import exists as path_exists
from os.path import join as path_join
from os.path import isdir, splitext, sep
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from functools import partial
from ast import literal_eval

import numpy as np
import glob
import ipdb
import mat73
import re
import json
from scipy.io import loadmat
from sklearn.cluster import MiniBatchKMeans

# import cv2

from config import DEBUG, NROWS, N_MARKERS\
        ,TIME_COLS, NO_HACC_ID, DATA_DIR

def datetime_to_ms(time_in, is_iso=False):
    dstr = datetime.today()
    try:
        fmt ="%Y-%m-%d %I.%M.%S.%f %p" 
        dstr = datetime.strptime(time_in, fmt)
    except ValueError:
        if 'Take' in time_in:
            time_section = time_in[5:-4]
            if 'Task' or 'MR' in time_in:
                start_ind = re.search(r'\d{4}', time_in)
                end_ind = re.search(r'M_', time_in)
                time_section = time_in[start_ind.start():end_ind.start()+1]
            dstr = datetime.strptime(time_section, "%Y-%m-%d %I.%M.%S %p")
        elif '_' in time_in:
            fmt = "%Y_%m_%d-%H_%M_%S" 
            dstr = datetime.strptime(time_in, fmt)
        elif '/' in time_in:
            fmt = "%d/%m/%Y %H:%M:%S.%f" 
            dstr = datetime.strptime(time_in, fmt)
        elif 'Z' in time_in:
            fmt = "%Y%m%dT%H%M%SZ"
            dstr = datetime.strptime(time_in, fmt)

    millisec = dstr.timestamp()*1000
    # td = timedelta(hours=dstr.hour,
    #                minutes=dstr.minute,
    #                seconds=dstr.second,
    #                microseconds=dstr.microsecond)
    # millisec = td.total_seconds()*1000
    return millisec

def ms_to_datetime(ms):
    return datetime.fromtimestamp(ms/1000.0)

def mat_to_ms(time):
    if sum([':'==ch for ch in time]) == 1:
        dstr = datetime.strptime(time, '%m/%d/%Y %H:%M')
    elif time[-1] == 'M':
        dstr = datetime.strptime(time, '%m/%d/%Y  %I:%M:%S %p')
    elif '.' not in time:
        dstr = datetime.strptime(time, '%m/%d/%Y  %H:%M:%S')
    else:
        dstr = datetime.strptime(time, '%m/%d/%Y %H:%M:%S.%f')
    millisec = dstr.timestamp()*1000
    # td = timedelta(hours=dstr.hour,
    #                minutes=dstr.minute,
    #                seconds=dstr.second,
    #                microseconds=dstr.microsecond)
    # millisec = td.total_seconds()*1000
    return millisec

def cond_to_label(cond_str:str):
    my_dict = {'M': 0, 'R': 1, 'L0': 2, 'L1': 3, 'L2': 4, 'L3': 5}
    lbl = my_dict[cond_str]
    return lbl

def get_conditions(fname:str):
    return fname.split(sep)[-1].split("_")[0]

def split_csv_method(fname:str, skip_ratio=0.8, is_train=True, skiprows=None,
                    **kwargs):
    nrows = None
    with open(fname) as f:
        nrows_tot = sum(1 for line in f)
    nrows_tot -= 1

    if skiprows is not None:
        nrows_tot = nrows_tot - skiprows

    if skip_ratio > 0:
        if is_train:
            nrows = int(nrows_tot*skip_ratio)
        else:
            if skiprows is not None:
                skiprows += int(nrows_tot*skip_ratio)
            else:
                skiprows = int(nrows_tot*skip_ratio)
            skiprows = range(1, skiprows+1)

    df = pd.read_csv(fname, skipinitialspace=True, skiprows=skiprows,
                     header=0, nrows=nrows, **kwargs)
    cond = get_conditions(fname)
    df['condition'] = cond
    return df

def read_csv_method(fname:str, **kwargs):
    df = pd.read_csv(fname, skipinitialspace=True, 
                     header=0, **kwargs)
    cond = get_conditions(fname)
    df['condition'] = cond
    return df

def load_files_conditions(f_list:list, skip_ratio=None, **kwargs):
    if skip_ratio is not None:
        method = partial(split_csv_method, skip_ratio=skip_ratio, **kwargs)
    else:
        method = partial(read_csv_method, **kwargs)
    with Pool(processes=cpu_count()) as p:
        df_list = p.map(method, f_list)
    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by='ms', inplace=True)
    return df

def load_and_snip(f_list:list, ratios=[0.3, 0.1]):
    method = partial(read_csv_method)
    with Pool(processes=cpu_count()) as p:
        df_list = p.map(method, f_list)

    if len(ratios) == 2:
        for i, df in enumerate(df_list):
            l = len(df)
            skiprows = int(ratios[0]*l)
            nrows = int((1-sum(ratios))*l)+1
            df_list[i] = df.iloc[skiprows:(skiprows+nrows)]
    return df_list

def get_file_list(starts_with:str, sbj=None):
    if sbj is not None:
        f_glob = path_join(DATA_DIR, sbj, '**')
    else:
        f_glob = path_join(DATA_DIR, 'S*', '**')

    if starts_with is not None:
        f_glob = path_join(f_glob, f'{starts_with}*.csv')
    else:
        f_glob = path_join(f_glob, '*.csv')

    f_list = sorted(glob.glob(f_glob, recursive=True))
    # pop zero size files
    return f_list

def get_windowed_data(time, data, vsw, thold=100):
    out = []
    for i, w_inds in tqdm(enumerate(vsw), total=vsw.shape[0]):
        if w_inds[-1] == 0: break
        t0, t1 = time[w_inds][0], time[w_inds][-1]
        diff = time[w_inds[1:]] - time[w_inds[0:-1]]
        mask = diff>thold
        diff_chk = np.any(mask)
        if diff_chk:
            continue
        out.append(data[w_inds])
    
    return np.array(out)

def get_imu_numpy(x_in):
    try:
        acc_data = np.array(x_in['accelerometer']\
                            .map(literal_eval).tolist())
        gyr_data = np.array(x_in['gyroscope']\
                            .map(literal_eval).tolist())
    except:
        acc_data = np.stack(x_in['accelerometer'].values)
        gyr_data = np.stack(x_in['gyroscope'].values)
    data = np.concatenate((acc_data, gyr_data), axis=1)
    return data

def parallelize_dataframe(df, func):
    num_processes = cpu_count()
    df_split = np.array_split(df, num_processes)
    with Pool(num_processes) as p:
        df = pd.concat(p.map(func, df_split))
    return df

class DataImporter():
    def __init__(self):
        self.imu_fname        = ''
        self.marker_fname     = ''
        self.timeline_fname   = ''
        self.summary_fname    = ''
        self.video_fname      = ''
        if DEBUG:
            self.nrows_to_import  = NROWS
        else:
            self.nrows_to_import  = None

        if platform =='linux' or platform == 'linux2':
            self.sep = "/"
        else:
            self.sep = "\\"

        if platform =='linux' or platform == 'linux2':
            self.parent_dir = '/data/rqchia/ARIA/Data'
        else:
            self.parent_dir = 'D:\\Raymond Chia\\UTS\\Howe Zhu - Data\\'

    def import_rigid_body_data(self):
        col_keys = ['frame', 'time (seconds)', 'mean marker error',
                    'marker quality', 'rigid body', 'position', 'rotation',
                    'x', 'y', 'z']
        filename = self.marker_fname
        header = pd.read_csv(self.marker_hdr_fname, nrows=1, usecols=list(range(0,22)),
                             header=None)
        header = dict(header.values.reshape((11,2)).tolist())
        if self.nrows_to_import is None:
            df = pd.read_csv(
                filename, header=list(range(0,5))
            )
        else:
            df = pd.read_csv(
                filename, nrows=self.nrows_to_import, header=list(range(0,5))
            )
        shape = df.shape
        if shape[1] > 10:
            diff = shape[1] - 10
            df = df.drop(df.columns[-diff::], axis=1)
        cols = df.columns.values
        new_cols = []
        for i, lstr in enumerate(cols):
            col_val = []
            lstr_list = [ls for ls in lstr]
            if 'Rigid Body Marker' in lstr_list: continue
            for j, str_val in enumerate(lstr):
                if str_val.lower() in col_keys or 'glasses' in str_val.lower():
                    if ' ' in str_val:
                        str_val = str_val.replace(' ', '_')
                    col_val.append(str_val)
            new_cols.append('_'.join(col_val))
        df.columns = new_cols

        return df, header

    def cleanup_marker_data(self, filename):
        chunksize = 10
        file_size_mb = stat(filename).st_size/(1024*1024)
        ff = filename.split(self.sep)[:-1]
        if file_size_mb > 0.5:
            print("processing: ", filename)
            header = pd.read_csv(
                filename, nrows=1, usecols=list(range(0,22)), header=None)
            hdr_name = path_join(self.sep.join(ff),
                                 filename[:-4] + '_header.csv')
            header.to_csv(hdr_name, index=False)
            df_hdr = pd.read_csv(filename, skiprows=2, header=list(range(0,5)),
                                 nrows=0)
            df = pd.read_csv(filename, skiprows=6, usecols=list(range(38)))
            df.columns = df_hdr.columns[:38]
            amended_df_name = path_join(
                self.sep.join(ff), filename[:-4] + '_amended.csv')
            df.to_csv(amended_df_name, index=False)
            print("saved: ", amended_df_name)

    def import_marker_file(self, filename):
        if self.nrows_to_import is None:
            df = pd.read_csv(
                filename, header=list(range(0,5))
            )
        else:
            df = pd.read_csv(
                filename, nrows=self.nrows_to_import,
                header=list(range(0,5))
            )
        return df
    
    def import_header_file(self, filename):
        df = pd.read_csv(filename, skiprows=1, nrows=1,
                         usecols=list(range(0,22)), header=None)
        return df

# Import .mat files from markers
    def import_marker_data(self):
        col_keys = ['frame', 'time (seconds)', 'mean marker error',
                    'marker quality', 'marker', 'position', 'rotation',
                    'x', 'y', 'z']
        filename = self.marker_fname
        header = self.import_header_file(self.marker_hdr_fname)
        df = self.import_marker_file(filename)

        shape = df.shape
        if shape[1] > 38:
            diff = shape[1] - 38
            df = df.drop(df.columns[-diff::], axis=1)
        cols = df.columns.values
        new_cols = []
        for i, lstr in enumerate(cols):
            col_val = []
            if type(lstr[0]) is str and "('" in lstr[0]:
                tmp = lstr[0][1:-1].split(',')
                lstr = [ll.replace(" '", '').replace("'", "") for ll in tmp]
            for j, str_val in enumerate(lstr):
                if str_val.lower() in col_keys or 'glasses' in str_val.lower():
                    if ' ' in str_val:
                        str_val = str_val.replace(' ', '_')
                    col_val.append(str_val)
            new_cols.append('_'.join(col_val))
        df.columns = new_cols

        header = dict(header.values.reshape((11,2)).tolist())
        return df, header

    # Import labels from csv
    def import_labels(self, filename):
        if self.nrows_to_import is None:
            df = pd.read_csv(filename, skipinitialspace=True)
        else:
            df = pd.read_csv(filename, nrows=self.nrows_to_import, skipinitialspace=True)

        return df

    def import_mat_data(self, filename):
        try:
            data_dict = mat73.loadmat(filename)
        except TypeError:
            data_dict = loadmat(filename)

        times = data_dict['StoreData']
        df = pd.DataFrame(times, columns=TIME_COLS)
        df = df.applymap(np.squeeze)
        #  a few nested lists, repeat once more
        df = df.applymap(np.squeeze)
        return df

    def import_time_data(self):
        filename = self.timeline_fname
        if '.mat' in filename:
            return self.import_mat_data(filename)
        elif '.csv' in filename:
            return pd.read_csv(filename)

    def import_imu_data(self):
        filename = self.imu_fname
        try:
            df = pd.read_json(filename, lines=True, compression='gzip')
        except EOFError:
            df = pd.read_json(splitext(filename)[0], lines=True)
        data_df = pd.DataFrame(df['data'].tolist())
        df = pd.concat([df.drop('data', axis=1), data_df], axis=1)
        hdr = self.import_imu_header()
        hdr = hdr.to_dict().pop(0)
        return df, hdr
    
    def import_imu_header(self):
        return pd.read_json(self.imu_hdr_fname, orient='index')
    
    # def import_video(self):
    #     return cv2.VideoCapture(self.import_video)

class DataSynchronizer():
    def __init__(self):
        self.start_ind = None
        self.end_ind = None

    # Sync and downsample to match frequences across the datasets
    def sync_df_start(self, df):
        ''' sync dataframe '''
        my_df = df.drop(index=df.index[:self.start_ind],
                axis=0,
                inplace=False)
        return my_df

    def sync_df_end(self, df):
        ''' sync dataframe '''
        diff = self.end_ind - self.start_ind + 1
        my_df = df.drop(index=df.index[diff::],
                axis=0,
                inplace=False)
        return my_df

    def sync_df(self, df):
        ''' sync to mat data '''
        my_df = df.iloc[self.start_ind:self.end_ind+1]
        # my_df = self.sync_df_start(df)
        # my_df = self.sync_df_end(my_df)
        return my_df

    def set_bounds(self, times, t_start, t_end):
        ''' sync to using masking method '''
        # find the index that is closest to t_start
        start_mask0 = times <= t_start
        start_mask1 = times > t_start
        if not start_mask0.any() and t_start < times[0]:
            start0 = times[0]
        else:
            start0 = times[start_mask0][-1]
        start1 = times[start_mask1][0]

        # take lowest
        dt0 = np.abs(t_start-start0)
        dt1 = np.abs(t_start-start1)
        if dt0 < dt1:
            start_val = start0
        else:
            start_val = start1
        start_ind = np.where(times==start_val)[0][0]

        end_mask0 = times <= t_end
        end_mask1 = times > t_end
        end0 = times[end_mask0][-1]
        
        times_end = times[-1]
        if np.isnan(times_end): 
            times_end = times[-2] + 1000
            times[-1] = times_end

        if not end_mask1.any() and t_end >= times_end:
            end1 = times_end
        else:
            end1 = times[end_mask1][0]

        # take dt1
        dt0 = np.abs(t_end-end0)
        dt1 = np.abs(t_end-end1)
        if dt0 < dt1:
            end_val = end0
        else:
            end_val = end1
        end_ind = np.where(times==end_val)[0][0]
        # end_diff = end_ind - start_ind + 1

        self.start_ind = start_ind
        self.end_ind   = end_ind

class SubjectData(DataImporter):
    ''' Loads in data for the rigid body, marker, breathing rate, summary files
    and syncs them accordingly '''
    def __init__(self, condition='M', subject='S01'):
        super().__init__()
        self.condition   = condition
        self.subject     = subject
        if subject[0] != 'S':
            self.subject_id = subject
        else:
            self.subject_id  = int(re.search(r'\d+', subject).group())
        self.study_start = 0
        self.study_end   = 0

        self.subject_dir = path_join(self.parent_dir, self.subject)

        self.marker_df = pd.DataFrame()
        self.pressure_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.accel_df = pd.DataFrame()
        self.imu_df = pd.DataFrame()

        self.read_marker_data = False

    def get_cond_file(self, files):
        for f in files:
            if self.condition in split(f)[-1] and \
               self.subject in split(f)[-1]:
                return f
        return ''

    def list_sub_dirs(self, parent_dir, endswith=None):
        reg_str = r'[0-9]+$'
        if endswith is not None:
            reg_str = r'[0-9]+{0}$'.format(endswith)
        regex = re.compile(reg_str)
        sub_dirs = [
            path_join(parent_dir, d) for d in listdir(parent_dir) if \
            (
                isdir(path_join(parent_dir,d)) and bool(regex.search(d))
            )
        ]
        return sorted(sub_dirs)

    def check_times(self, sub_dirs, is_utc=False):
        ''' Parses sub directory names to datetime and checks against mat_start
        and end '''
        sep = self.sep

        if is_utc:
            imu_hdr_files = [path_join(sub_dir, 'recording.g3')\
                             for sub_dir in sub_dirs]
            hdrs = [pd.read_json(imu_hdr_file, orient='index')\
                    for imu_hdr_file in imu_hdr_files]
            times = [hdr.to_dict().pop(0)['created'] \
                     for hdr in hdrs]
            times = [datetime.fromisoformat(time[:-1]) for time in times]
            times = [(time.timestamp()+ timedelta(hours=11).seconds)*1000 for time in times]
        else:
            times = [datetime_to_ms(sub_dir.split(sep)[-1])\
                     for sub_dir in sub_dirs]

        sel_dir = sub_dirs[-1]
        for i, time in enumerate(times[:-1]):
            if self.study_start > time and self.study_start < times[i+1]:
                sel_dir = sub_dirs[i]
        return sel_dir

    def set_marker_fname(self):
        subject_dir = self.subject_dir
        data_dir = path_join(subject_dir, 'Motive Logs')
        if not path_exists(data_dir):
            data_dir = subject_dir
            data_glob = path_join(data_dir, "*Take*_amended.csv")
        else:
            data_glob = path_join(data_dir, "*_amended.csv")

        data_files = sorted(glob.glob(data_glob))
        if self.subject_id > 16:
            condition_chk = self.condition in 'MR'
            if not condition_chk:
                data_files = [dg for dg in data_files if 'MR' not in dg]
            else:
                data_files = [dg for dg in data_files if 'MR' in dg]
        self.marker_fname = data_files[-1]
        if len(data_files)> 1:
            # Check directory times with timeline
            self.marker_fname = self.check_times(data_files)
            # set marker header name
        self.marker_hdr_fname = self.marker_fname.split('_amended')[0] + \
                '_header.csv'

    def set_pressure_fname(self):
        subject_dir = self.subject_dir
        sub_dirs = self.list_sub_dirs(subject_dir)
        sub_dir = sub_dirs[0]
        if len(sub_dirs)> 1:
            # Check directory times with timeline
            sub_dir = self.check_times(sub_dirs)

        pressure_glob   = path_join(sub_dir, 'BR*.csv')
        pressure_files   = sorted(glob.glob(pressure_glob))
        if not pressure_files:
            dt_info = split(sub_dir)[-1]
            pressure_glob = path_join(sub_dir, '*_Breathing.csv')
            pressure_files = sorted(glob.glob(pressure_glob))
        self.pressure_fname = pressure_files[-1]

    def set_summary_fname(self):
        subject_dir = self.subject_dir
        sub_dirs = self.list_sub_dirs(subject_dir)
        sub_dir = sub_dirs[0]
        if len(sub_dirs)> 1:
            # Check directory times with timeline
            sub_dir = self.check_times(sub_dirs)

        summary_glob = path_join(sub_dir, 'Summary*.csv')
        summary_files = sorted(glob.glob(summary_glob))
        if not summary_files:
            dt_info = split(sub_dir)[-1]
            summary_glob = path_join(sub_dir, dt_info+'_Summary*.csv')
            summary_files = sorted(glob.glob(summary_glob))
        self.summary_fname  = summary_files[-1]

    def set_imu_fname(self):
        subject_dir = self.subject_dir
        sub_dirs = self.list_sub_dirs(subject_dir, endswith='Z')
        sub_dir = sub_dirs[0]
        if len(sub_dirs)> 1:
            sub_dir = self.check_times(sub_dirs, is_utc=True)

        imu_glob = path_join(sub_dir, 'imu*')
        imu_files = sorted(glob.glob(imu_glob))
        self.imu_fname  = imu_files[-1]

        imu_hdr_glob = path_join(sub_dir, 'recording.g3')
        imu_hdr_files = sorted(glob.glob(imu_hdr_glob))
        self.imu_hdr_fname  = imu_hdr_files[-1]
        video_glob = path_join(sub_dir, 'scenevideo.mp4')
        # self.video_fname = glob.glob(video_glob)[-1]

    def set_accel_fname(self):
        subject_dir = self.subject_dir
        sub_dirs = self.list_sub_dirs(subject_dir)
        sub_dir = sub_dirs[0]
        if len(sub_dirs)> 1:
            # Check directory times with timeline
            sub_dir = self.check_times(sub_dirs)

        accel_glob = path_join(sub_dir, 'Accel*.csv')
        accel_files = sorted(glob.glob(accel_glob))
        if not accel_files:
            dt_info = split(sub_dir)[-1]
            accel_glob = path_join(sub_dir, '*_Accel.csv')
            accel_files = sorted(glob.glob(accel_glob))
        accel_files = [f for f in accel_files if 'g' not in \
                       split(f.lower())[-1]]
        self.accel_fname  = accel_files[-1]

    def set_timeline(self):
        times_glob  = path_join(self.subject_dir,f'*.csv')
        times_files = sorted(glob.glob(times_glob))
        self.timeline_fname = self.get_cond_file(times_files)
        self.timeline_df = self.import_time_data()

        mat_time = self.timeline_df['Timestamps'].map(mat_to_ms)
        mat_start_ind = self.timeline_df.index[
            self.timeline_df['Event']=='Start Test'
        ].tolist()[0]
        mat_start = mat_time.values[mat_start_ind]
        mat_end = mat_time.values[-1]

        self.study_start = mat_start
        self.study_end = mat_end

    def set_fnames(self):
        if self.read_marker_data:
            self.set_marker_fname()
        self.set_pressure_fname()
        self.set_summary_fname()
        if self.subject_id > 11:
            self.set_imu_fname()
        if self.subject_id not in NO_HACC_ID:
            self.set_accel_fname()

    def load_dataframes(self):
        self.timeline_df = self.import_time_data()
        self.pressure_df = self.import_labels(self.pressure_fname)
        self.summary_df = self.import_labels(self.summary_fname)
        if self.read_marker_data:
            try:
                self.marker_df, self.mkr_hdr = self.import_marker_data()
                self.rb_df, self.rb_hdr = self.import_rigid_body_data()
            except:
                print("error reading marker data on {0} - {1}".format(
                    self.subject_id, self.condition))
        if self.subject_id not in NO_HACC_ID:
            try:
                self.accel_df = self.import_labels(self.accel_fname)
            except:
                print("error reading accel data on {0} - {1}".format(
                    self.subject_id, self.condition))
        if self.subject_id > 11:
            try:
                self.imu_df, self.imu_hdr = self.import_imu_data()
            except:
                print("error reading imu data on {0} - {1}".format(
                    self.subject_id, self.condition))

    def sync_marker_df(self):
        data_sync = DataSynchronizer()

        cst = self.mkr_hdr['Capture Start Time']

        time_start = datetime_to_ms(cst)
        marker_time = self.marker_df['Time_(Seconds)'].values*1000 + time_start
        time_end = marker_time[-1]

        self.marker_df['Time_(Seconds)'] = marker_time/1000
        self.marker_df['ms'] = marker_time
        self.rb_df['Time_(Seconds)'] = marker_time/1000
        self.rb_df['ms'] = marker_time
        data_sync.set_bounds(marker_time, self.study_start, self.study_end)

        self.marker_df = data_sync.sync_df(self.marker_df).fillna(0)
        self.rb_df = data_sync.sync_df(self.rb_df)

    def sync_pressure_df(self):
        data_sync = DataSynchronizer()

        cols = self.pressure_df.columns
        if 'Year' in cols:
            year = int(self.pressure_df['Year'].values[0])
            month = int(self.pressure_df['Month'].values[0])
            day = int(self.pressure_df['Day'].values[0])
            dt_fmt = "%Y/%m/%d"
            dt_str = f"{year}/{month}/{day}"
            dt_obj = datetime.strptime(dt_str, dt_fmt)
            pressure_time = self.pressure_df['ms'].interpolate().values
            pressure_time = pressure_time + dt_obj.timestamp()*1000
        else:
            pressure_time = self.pressure_df['Time'].map(datetime_to_ms).values

        self.pressure_df['ms'] = pressure_time
        data_sync.set_bounds(pressure_time, self.study_start, self.study_end)
        self.pressure_df = data_sync.sync_df(self.pressure_df)

    def sync_accel_df(self):
        data_sync = DataSynchronizer()

        cols = self.accel_df.columns
        if 'Year' in cols:
            year = int(self.accel_df['Year'].values[0])
            month = int(self.accel_df['Month'].values[0])
            day = int(self.accel_df['Day'].values[0])
            dt_fmt = "%Y/%m/%d"
            dt_str = f"{year}/{month}/{day}"
            dt_obj = datetime.strptime(dt_str, dt_fmt)
            accel_time = self.accel_df['ms'].interpolate().values
            accel_time = accel_time + dt_obj.timestamp()*1000
        else:
            accel_time = self.accel_df['Time'].map(datetime_to_ms).values

        self.accel_df['ms'] = accel_time
        data_sync.set_bounds(accel_time, self.study_start, self.study_end)
        self.accel_df = data_sync.sync_df(self.accel_df)

    def sync_summary_df(self):
        data_sync = DataSynchronizer()

        cols = self.summary_df.columns
        if 'Year' in cols:
            year = int(self.summary_df['Year'].values[0])
            month = int(self.summary_df['Month'].values[0])
            day = int(self.summary_df['Day'].values[0])
            dt_fmt = "%Y/%m/%d"
            dt_str = f"{year}/{month}/{day}"
            dt_obj = datetime.strptime(dt_str, dt_fmt)
            summary_times = self.summary_df['ms'].values + dt_obj.timestamp()*1000
        else:
            summary_times = self.summary_df['Time'].map(datetime_to_ms).values

        self.summary_df['ms'] = summary_times
        data_sync.set_bounds(summary_times, self.study_start, self.study_end)
        self.summary_df = data_sync.sync_df(self.summary_df)

    def sync_imu_df(self):
        na_inds = self.imu_df\
                .loc[pd.isna(self.imu_df['accelerometer']), :].index.values
        self.imu_df.drop(index=na_inds, inplace=True)
        imu_times = self.imu_df['timestamp'].values

        ''' S21, S30 has strange time recordings '''
        mask = imu_times > 3*60*60
        if mask.any():
            bad_args = np.arange(0, len(mask))[mask]
            self.imu_df.drop(index=self.imu_df.iloc[bad_args].index,
                             inplace=True)
            # self.imu_df['timestamp'] = self.imu_df['timestamp'].values - \
            #         self.imu_df['timestamp'].values[0]
            imu_times = self.imu_df['timestamp'].values

        print(np.mean(1/(imu_times[1:] - imu_times[:-1])))
        self.imu_df['timestamp_interp'] = imu_times
        self.imu_df['timestamp_interp'] = self.imu_df['timestamp_interp']\
                .interpolate()

        data_sync = DataSynchronizer()

        iso_tz = self.imu_hdr['created']
        tzinfo = pytz.timezone(self.imu_hdr['timezone'])
        # adjust for UTC
        start_time = datetime.fromisoformat(iso_tz[:-1]) + timedelta(hours=11)
        imu_times = self.imu_df['timestamp_interp'].values

        imu_datetimes = [start_time + timedelta(seconds=val) \
                         for val in imu_times]
        imu_ms = np.array([time.timestamp()*1000 for time in imu_datetimes])
        self.imu_df['ms'] = imu_ms
        data_sync.set_bounds(imu_ms, self.study_start, self.study_end)
        self.imu_df = data_sync.sync_df(self.imu_df)

    def sync_all_df(self):
        if self.study_start == 0 or self.study_start is None:
            self.set_timeline()
        self.sync_pressure_df()
        self.sync_summary_df()
        if self.subject_id not in NO_HACC_ID:
            try:
                self.sync_accel_df()
            except:
                print("Error syncing accel data on {0} - {1}".format(
                    self.subject_id, self.condition))
                self.marker_df = pd.DataFrame()
        if self.read_marker_data:
            try:
                self.sync_marker_df()
            except:
                print("Error syncing marker data on {0} - {1}".format(
                    self.subject_id, self.condition))
                self.marker_df = pd.DataFrame()
        if self.subject_id > 11:
            try:
                self.sync_imu_df()
            except:
                print("Error syncing imu data on {0} - {1}".format(
                    self.subject_id, self.condition))
                self.imu_df = pd.DataFrame()

    def get_rigid_body_data(self, col_str='Rigid_Body'):
        data_cols = [col for col in self.marker_df.columns.values \
                     if col_str in col]
        marker_data = np.zeros((marker_df.shape[0], 3))
        for col in data_cols:
            if 'position_x' in col.lower():
                marker_data[:, 0] = marker_df[col].values
            elif 'position_y' in col.lower():
                marker_data[:, 1] = marker_df[col].values
            elif 'position_z' in col.lower():
                marker_data[:, 2] = marker_df[col].values
        return marker_data

    def get_marker_data(self, col_str='Marker'):
        data_cols = [col for col in self.marker_df.columns.values \
                     if col_str in col]
        marker_data = np.zeros((self.marker_df.shape[0], N_MARKERS, 3))
        for col in data_cols:
            for i in range(N_MARKERS):
                if str(i+1) not in col:
                    continue
                if 'position_x' in col.lower():
                    marker_data[:, i, 0] = self.marker_df[col].values
                elif 'position_y' in col.lower():
                    marker_data[:, i, 1] = self.marker_df[col].values
                elif 'position_z' in col.lower():
                    marker_data[:, i, 2] = self.marker_df[col].values
        return marker_data

    def get_marker_quality(self, col_str='quality'):
        data_cols = [col for col in self.marker_df.columns.values \
                     if col_str in col.lower()]
        marker_quality = np.zeros((self.marker_df.shape[0], N_MARKERS))
        for col in data_cols:
            for i in range(N_MARKERS):
                if str(i+1) not in col:
                    continue
                marker_quality[:, i] = self.marker_df[col].values
        return marker_quality

    def get_accel_data(self):
        accel_cols = self.accel_df.columns
        if 'Time' in accel_cols:
            data_cols = ['Vertical', 'Lateral', 'Sagittal']
        else:
            data_cols = ['X Data', 'Y Data', 'Z Data']
        return self.accel_df[data_cols].values

    def get_marker_ms(self):
        return self.marker_df['Time_(Seconds)'].values*1000

class TFDataPipeline():
    def __init__(self, window_size=60, batch_size=32):
        self.window_size = window_size
        self.window_shift = self.window_size
        self.batch_size = batch_size
        self.mb_kmeans = MiniBatchKMeans(n_clusters=int(self.batch_size//2))
        self.shuffle_flag = True

    def kcenter(self, x, y):
        '''
        * get some batches
        * perform iterative kcenter_greedy on these batches
        * get their scores and weight these batches (x, y, weights)
        * returns final centers and max distances
        '''
        self.mb_kmeans.partial_fit(x, y)
        x, dist = self.mb_kmeans.transform(x, y)
        # apply threshold
        # return weighted labels for each batch, or downsample the batch_size
        # to some percentage
        pass

    def sub_batch(self, x):
        sub_x = x.batch(self.window_size, drop_remainder=True)
        return sub_x

    def get_dataset(self, x, reduce_mean=False):
        ds = tf.data.Dataset.from_tensor_slices((x))\
                .window(self.window_size, shift=self.window_shift,
                        drop_remainder=True)\
                .flat_map(self.sub_batch)\
                .batch(self.batch_size, drop_remainder=True)
        if reduce_mean:
            ds = ds.map(lambda y: tf.reduce_mean(y, axis=1))
        ds = ds.prefetch(1)
        return ds
    
    def zip_datasets(self, x_ds, y_ds):
        ds = tf.data.Dataset.zip((x_ds, y_ds))
        if self.shuffle_flag:
            ds.shuffle(3000, reshuffle_each_iteration=True)
        return ds

# Data Feature Handler Class:
    # Deal with metafile
    # Load given a configuration dict
    # Set the directory and filename
class ProjectFileHandler():
    def __init__(self, config:dict):
        self.config  = config
        self.fset_id = -1
        self.metafile_name = 'metafile.json'
        self.set_home_directory()

    def set_home_directory(self, home_directory=DATA_DIR):
        self.home_directory = home_directory

    def set_parent_directory(self, parent_directory='imu_mwl'):
        self.parent_directory = path_join(self.home_directory,
                                          parent_directory)
        makedirs(self.parent_directory, exist_ok=True)

    def set_id(self, fset_id:int=-1):
        if fset_id != -1:
            self.fset_id = fset_id
        else:
            ww = walk(self.parent_directory, topdown=True)
            for _, dirs, _ in ww:
                tmp = len(dirs)
                break
            self.fset_id = tmp

    def set_project_directory(self):
        if self.fset_id == -1:
            self.set_id()
            print("Data id not set, auto assigned to: ", self.fset_id)

        self.project_directory = path_join(self.parent_directory,
                                         str(self.fset_id).zfill(2))
        makedirs(self.project_directory, exist_ok=True)

    def get_metafile_path(self):
        fname = path_join(self.project_directory, self.metafile_name)
        return fname

    def get_id_from_config(self):
        glob_pattern = path_join(self.parent_directory, '**', '*.json')
        mfiles = glob.glob(glob_pattern, recursive=True)
        cfg_id = {}
        for mfile in mfiles:
            fset_id_in = mfile.split(sep)[-2]
            with open(mfile, 'r') as f:
                cfg_in = json.load(f)
            cfg_id[fset_id_in] = cfg_in
            if self.config == cfg_in:
                return fset_id_in
        
        print("unable to find matching config id")

    def save_metafile(self):
        fname = self.get_metafile_path()
        with open(fname, 'w') as f:
            json.dump(self.config, f)
