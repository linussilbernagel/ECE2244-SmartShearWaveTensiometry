# sw_load - loads a raw shear wave data collection
#           this version can convert old format data to new format
#
# DGT 9/16/2021
# 
# 

from scipy.io import loadmat
import numpy as np
import h5py

class sw_load_notSensor:
    # SWC - shear wave speed data collection class
    #
    # Author:
    #   9/11/2021  Darryl Thelen, dgthelen@wisc.edu
    def __init__(self):
        # ad - A/D data
        self.ad = {'data': [], 'ch': [], 'freq': []}
                # data  raw A/D sampled data
                # ch    - AD channel #'s sampled
                # gains - A/D gains 
                # freq  - sample frequency
                # time  - sample time
        # info stores basic information about the data collection
        self.info = {'path': '', 'file': '', 'date': '', 'time': '', 'task': '', 'subj': '', 'note': '', 'trial': []}
                # 'path     path where the data is stored
                # 'file     name of the file where the raw data is stored
                # 'date     date when the data was collected
                # 'task     type of activity
                # 'subj     subject ID
                # 'note     any test notes
        # Tensiometer structure
        self.tens = {'info': {'tendon': '', 'note': ''}, 'cp': {'taps': [], 'tapdur': [], 'tapfreq': [], 'tapms': [], 'tapsig': [], 'tapdc': [], 'accch': [], 'accdis': [], 'acccal': [], 'accgain': []}}
                #
                # info - normally set previously by the collection program
                #     tendon - tendon the tensiometer is placed on
                #     note - any test notes
                # 
                # cp - collection parameters
                #    taps - indices of the tap onsets
                #    tapms - duration of the tap in ms
                #    tapdur - duration of the tap in # of points
                #    tapdc - duty cycle of the tap sequency
                #    tapsig - profile of the the tap signal over a cycle
                #    accch  - indices to the AD columns containing raw data for the accelerometers
                #    accdis - distance from the tapper to each of the accelerometers, expressed in mm
                #    acccal - calibration parameter for the accelerometer, should be m/s^2/V
                #    accgain - accelerometer gains used on the amplifier
        # Footstrike sensor structure --- DGS ---
        self.fs = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 
        # Load cell sensor structure --- DGS ---
        self.lc = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 
        # Sync sensor structure --- DGS ---
        self.sync = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 
        # Sync sensor structure --- DGS ---
        self.emgsol = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 
        # Sync sensor structure --- DGS ---
        self.emgmg = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 
        # Sync sensor structure --- DGS ---
        self.emglg = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 
        # Sync sensor structure --- DGS ---
        self.emgta = {'accch': []}
                # accch = indices to the AD column containin graw data for the footstrike accelerometer 

    def load_data(self, path, file=None):
        # Define class initialization function
        #   Inputs:
        #       path: filepath where data is stored 
        #       file: file to load
        #   Outputs:
        #       obj: Initialized class object
        #
        # if file is not None:
        self.info['path'] = path
        self.info['file'] = file
        data = h5py.File(path + '/' + file)
        # elif file is None:
        #     self.info['path'] = path
        #     file = uigetfile('*.mat', 'Select a wavespeed data file.', self.info['path'])
        #     self.info['file'] = file
        #     load(path + '/' + file, '-mat', 'info', 'sampleRate', 'sensorTypes', 'tapper')
        # else:
        #     file, path = uigetfile('*.mat', 'Select a wavespeed data file.')
        #     self.info['path'] = path
        #     self.info['file'] = file
        #     load(path + '/' + file, '-mat', 'info', 'sampleRate', 'sensorTypes', 'tapper')
        # Convert from the old data format to the new 
        try:
            self.info['note'] = data['info']['note']
            self.info['task'] = data['info']['activity']
            self.info['trial'] = data['info']['trial']
        except:
            exit

        self.ad['freq'] = data['sampleRate'][0][0]
        self.ad['ch'] = [data['sensorTypes']['accel']['AchillesOrRight']['channels'][0][0], data['sensorTypes']['accel']['AchillesOrRight']['channels'][1][0]]
        self.ad['data'] = np.array(data['sensorTypes']['accel']['AchillesOrRight']['signal'])

        # Assume tapper signal is a pulse - can simply find signal rise and fall
        tappersignal = np.array([0] + list(data['tapper']['signal'][0][:-1]))
        jj = np.where(np.diff(tappersignal) > 0)[0]
        kk = np.where(np.diff(tappersignal) < 0)[0]
        self.tens['cp']['taps'] = jj
        tapon = kk[1] - jj[0]
        self.tens['cp']['accch'] = list(range(1, len(data['sensorTypes']['accel']['AchillesOrRight']['channels']) + 1))
        self.tens['cp']['acccal'] = 1 / data['sensorTypes']['accel']['AchillesOrRight']['sensitivity'][0][0]
        self.tens['cp']['accgain'] = data['sensorTypes']['accel']['AchillesOrRight']['gain'][0][0]
        self.tens['cp']['accdis'] = [18] + [data['sensorTypes']['accel']['AchillesOrRight']['travelDist'][0][0] + 18] # mm from tapper to accelerometer
        self.tens['cp']['tapfreq'] = data['sampleRate'][0][0] / (jj[1] - jj[0])
        self.tens['cp']['tapdur'] = tapon
        self.tens['cp']['tapms'] = 1000 * tapon / self.ad['freq']
        self.tens['cp']['tapsig'] = [1] * tapon + [0] * (jj[1] - jj[0] - tapon)
        self.tens['cp']['tapdc'] = tapon / (jj[1] - jj[0])
        # Include heelstrike acc in 'data' --- DGS ---
        if 'footstrike' in data['sensorTypes']:
            self.fs['accch'] = len(self.ad['ch']) + 1
            self.ad['ch'].extend(data['sensorTypes']['footstrike']['footstrike']['channels'])
            self.ad['data'].extend(data['sensorTypes']['footstrike']['footstrike']['signal'])
        if 'other' in data['sensorTypes']:
            # Include load cell acc in 'data' --- DGS ---
            if 'LoadCell' in data['sensorTypes']['other']:
                self.lc['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['LoadCell']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['LoadCell']['signal'])
            # Include sync acc in 'data' --- DGS ---
            if 'sync' in data['sensorTypes']['other']:
                self.sync['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['sync']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['sync']['signal'])
            elif 'trigger' in data['sensorTypes']['other']:
                self.sync['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['trigger']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['trigger']['signal'])
            # Include emg acc in 'data' --- DGS ---
            if 'EMGSOL' in data['sensorTypes']['other']:
                self.emgsol['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['EMGSOL']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['EMGSOL']['signal'])
            if 'EMGMG' in data['sensorTypes']['other']:
                self.emgmg['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['EMGMG']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['EMGMG']['signal'])
            if 'EMGLG' in data['sensorTypes']['other']:
                self.emglg['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['EMGLG']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['EMGLG']['signal'])
            if 'EMGTA' in data['sensorTypes']['other']:
                self.emgta['accch'] = len(self.ad['ch']) + 1
                self.ad['ch'].extend(data['sensorTypes']['other']['EMGTA']['channels'])
                self.ad['data'].extend(data['sensorTypes']['other']['EMGTA']['signal'])


