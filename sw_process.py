# Shear wave tensiometry processing code
# primary input - a shear wave data collection consisting of acceleromettry
# data, tap information and user defined parameters
# outputs - wave speed computed via traditional cross-correlation and
# kalman filtering

import numpy as np
from scipy import signal
import copy

class sw_process:
    def __init__(self, swc):
        # ad - A/D data
        self.ad = {'data': [], 'ch': [], 'freq': []}
                # data  raw A/D sampled data
                # ch    - AD channel #'s sampled
                # gains - A/D gains 
                # freq  - sample frequency
                # time  - sample time
        
        self.ad = swc.ad.copy()
        
        # info stores basic information about the data collection
        self.info = {'path': '', 'file': '', 'date': '', 'time': '', 'task': '', 'subj': '', 'note': '', 'trial': []}
                # 'path     path where the data is stored
                # 'file     name of the file where the raw data is stored
                # 'date     date when the data was collected
                # 'task     type of activity
                # 'subj     subject ID
                # 'note     any test notes
        self.info = swc.info.copy()
        
        # Tensiometer structure
        self.tens = {'info': {'tendon': '', 'note': ''}, 
                      'cp': {'taps': [], 'tapdur': [], 'tapfreq': [], 'tapms': [],'tapsig': [], 'tapdc': [], 'accch': [], 'accdis': [], 'acccal': [], 'accgain': []}, 
                      'pp': {'bpf': [], 'window': [], 'windowSTD': [], 'TABmax': [], 'TABmincorr': [], 'dTAmax': [], 'dTAmincorr': [], 'mfw': [], 'accdis_sd': [], 'af': [], 
                             'bf': [], 'template': [], 'TABsearch': [], 'TABshift': [], 'dTAsearch': [], 'dTAshift': [], 'winseed': [], 'kalseed': []}, 
                      'lags': {'TA': [], 'TAB': [], 'TABcorr': [], 'TABvar': [], 'dTA': [], 'dTAcorr': [], 'dTAvar': []}, 
                      'time': [], 'ws': [], 'wsa': [], 'wsk': []}
                # cp - collection parameters
                #    taps - indices of the tap onsets
                #    tapms - duration of the tap in ms
                #    tapdur - duration of the tap in # of points
                #    tapdc - duty cycle of the tap sequence
                #    tapsig - profile of the the tap signal over a cycle
                #    accch  - indices to the AD columns containing raw data for the accelerometers
                #    accdis   - Distance from the tapper to each of the accelerometers, expressed in mm
                #    acccal - calibration parameter for the accelerometer, should be m/s^2/V
                #    accgain - accelerometer gains used on the amplifier
                # pp - processing parameters
                #   bpf         bandpass filter cutoff frequencies
                #   window      window after tap onset to use for the cross-correlation
                #   windowSTD   window for standard cross-correlation computation, only used for Kalman seed
                #   TABmax      maximum travel time between accelerometer pairs
                #   TABmincorr	minimum correlation to accept the time lag as valid
                #   dTAmax      maximum change in arrival time between successive taps 
                #   dTAmincorr	minimum correlation to accept the change in wave arrival time at A
                #   mfw         median filter window length to filter TAB, dTA
                #   accdis_sd - estimated s.d. of distance from the tapper to each of the accelerometers, expressed in mm
                #   winseed - reference point for window definition. [0] = index of tap event, [1] = predicted wave arrival from previous event
                #   kalseed - Kalman seed point selection method. 'first' (default), 'fwdbkwd' (-> and <-, average), 'auto' (max r in tau), 'manual' (ginput)
                #   ..........following are derive processing parameters....................
                #   af'         bandpass filter parameters, set from bpf and frequency
                #   bf'         bandpass filter parameters, set from bpf and frequency
                #   template	derived from correlation window and frequency at processing time
                #   templateSTD
                #   TABsearch	derived from correlation window, frequency and TABmax at processing time
                #   TABshift - derived from correlation window, frequency and TABmax at processing time
                #   TABsearchSTD
                #   TABshiftSTD
                #   dTAsearch - derived from correlation window, frequency and dTAmax at processing time
                #   dTAshift - derived from correlation window, frequency and dTAmax at processing time
                # var - output variables
                #   TAB         wave travel time between successive accelerometers
                #   TABcorr     normalized correlation of waves between successive accelerometers
                #   dTA         change in time of wave arrival at an accelerometer between taps
                #   dTAcorr     normalized correlation ofchange in time of wave arrival at an accelerometer between taps
                #   TA          time of wave arrival at an accelerometer - estimated via a kalman filter
                #   TAvar       variance in the estimating of the time of wave arrival at an accelerometer - estimated via a kalman filter
                #   ws          wave speed computed using the traditional cross-correlation approach
                #   wsa         wave speed computed using the new auto-correlation approach
                #   wsk         wave speed computed using the kalman filtering approach 
        self.tens['info'] = swc.tens['info'].copy()
        self.tens['cp'] = swc.tens['cp'].copy()
        
        # Initialize all the processing parameters
        self.tens['pp']['bpf'] = [100, 5000]
        self.tens['pp']['window'] = [0, 2]
        self.tens['pp']['windowSTD'] = [0, 2]
        self.tens['pp']['TABmax'] = 1.5

        self.tens['pp']['TABmincorr'] =.5     # minimum correlation to accept the time lag as valid
        self.tens['pp']['dTAmax'] =.3        # maximum change in arrival time between successive taps 
        self.tens['pp']['dTAmincorr'] =.5     # minimum correlation to accept the change in wave arrival time at A
        self.tens['pp']['mfw'] =3             # median filter window length
        self.tens['pp']['accdis_sd'] =.2     # s.d. expresses uncertainty in accelerometer distance from tapper, in mm
        self.tens['pp']['TABvarp']= [0.00658, -0.014, 0.00742]  # polynomial coefficients used to estimate variance in ms^2 from corr coef
        self.tens['pp']['dTAvarp']= [0.00658, -0.014, 0.00742]  # polynomial coefficients used to estimate variance in ms^2 from corr coef
        self.tens['pp']['winseed'] = 1 # use adaptive window center
        self.tens['pp']['kalseed'] = 'first'

        # Create space to stored all the processed information
        ntaps=len(self.tens['cp']['taps'])
        nacc=len(self.tens['cp']['accch'])
        self.tens['lags']['TA']= np.zeros((ntaps,nacc))
        self.tens['lags']['TAvar']= np.zeros((ntaps,nacc))
        self.tens['lags']['TAB']= np.zeros((ntaps,nacc-1))
        self.tens['lags']['TABcorr']= np.zeros((ntaps,nacc-1))
        self.tens['lags']['TABvar']= np.zeros((ntaps,nacc-1))
        self.tens['lags']['dTA']= np.zeros((ntaps,nacc))
        self.tens['lags']['dTAcorr']= np.zeros((ntaps,nacc))
        self.tens['lags']['dTAvar']= np.zeros((ntaps,nacc))
        self.tens['time']= np.zeros((ntaps,1))
        self.tens['ws']= np.zeros((ntaps,1))
        self.tens['wsk']= np.zeros((ntaps,1))       

        # Footstrike sensor structure --- DGS ---
        self.fs = {'accch': [], 'fsidx': [], 'fsidx_acc': []}
                # accch - indices to the AD columns containing raw data for the accelerometer
                # fsidx - indices of heelstrikes, in wave speed time
        
        # add footstrike sensor data --- DGS ---
        if hasattr(swc, 'fs') and hasattr(swc.fs, 'accch') and swc.fs['accch']:
           self.fs['accch'] = swc.fs['accch']
        
        # Plotting option
        self.plot = {'figs': [], 'option': 1, 'hold': 0}

        # TODO: change to either swc.ad or self.ad
        self.updateParams(swc.ad)
    
    def updateParams(self, ad):
            # Pre-computes parameters that are used repeatedly in wave
            # speed calculations
        
        nacc = len(self.tens['cp']['accch'])

        # Compute the bandpass filter parameters            
        self.tens['pp']['bf'], self.tens['pp']['af'] = signal.butter(3, [x / (ad['freq'] / 2) for x in self.tens['pp']['bpf']], 'bandpass')
        # Pre-determine the template window for TAB and dDTA
        jw = [round((x / 1000) * ad['freq']) for x in self.tens['pp']['window']]
        self.tens['pp']['template'] = list(range(jw[0], jw[1] + 1))

        jwSTD = [round((x / 1000) * ad['freq']) for x in self.tens['pp']['windowSTD']]
        self.tens['pp']['templateSTD'] = list(range(jwSTD[0], jwSTD[1] + 1))

        # The search window for TAB cross-correlations
        k = round((self.tens['pp']['TABmax'] / 1000) * ad['freq'])
        self.tens['pp']['TABsearch'] = list(range(jw[0], jw[1] + k + 1))
        self.tens['pp']['TABshift'] = 0
        self.tens['pp']['TABsearchSTD'] = list(range(jwSTD[0], jwSTD[1] + k + 1))
        self.tens['pp']['TABshiftSTD'] = 0

        # The search window for dTA cross-correlations, must go negative because it can shift either direction tap to tap
        k = round((self.tens['pp']['dTAmax'] / 1000) * ad['freq'])
        self.tens['pp']['dTAsearch'] = list(range(jw[0] - k, jw[1] + k + 1))
        self.tens['pp']['dTAshift'] = k

        # Dynamic Model of System:  x(k) = F*x(k-1)+B*u(k-1)+w(k)
        self.tens['kf'] = {}
        self.tens['kf']['xe'] = [0] * nacc
        self.tens['kf']['u'] = [0] * nacc
        self.tens['kf']['F'] = np.eye(nacc)
        self.tens['kf']['B'] = np.eye(nacc)
        # Process noise covariance - uncertainty in the model, set at run time
        self.tens['kf']['Q'] = np.zeros((nacc, nacc))
        # Measurement equation: z(k) = H*x(k) + v(k)
        # measurements - travel time between successive accelerometers plus constraint on arrival time at successive accelerometers
        self.tens['kf']['z'] = np.zeros((2*(nacc-1), 1))
        self.tens['kf']['H'] = [[-1 if j == k else 1 if j == k + 1 else 0 for j in range(nacc)] for k in range(nacc - 1)] + [[self.tens['cp']['accdis'][j + 1], -self.tens['cp']['accdis'][j]] for j in range(nacc - 1)]
        # R is the mesurement noise covariance - set at run-time
        self.tens['kf']['R'] = np.zeros((2 * (nacc - 1), 2 * (nacc - 1)))
        self.tens['kf']['P'] = np.zeros((nacc, nacc))
    
    def process(self, ad):
        # Main function called to process wavespeed data

        # update processing params
        self.updateParams(ad)

        # calculate wave speeds
        acc = self.wsCalc(ad)

        # plot computed wavespeed and ancillary data
        if self.plot['option']:
            for i in range(len(self.tens)):
                self.wsPlot(self.tens, acc, ad['freq'])
        if self.fs['accch']:
            self.fsID(ad)
    
    def wsCalc(self, ad):
        ntaps = len(self.tens['cp']['taps'])
        nacc = len(self.tens['cp']['accch'])
        kalseed = self.tens['pp']['kalseed']
        
        acc = ad['data'][:, self.tens['cp']['accch']] * self.tens['cp']['acccal'] / self.tens['cp']['accgain']
        zs = np.zeros((round(ad['freq'] / self.tens['cp']['tapfreq']), acc.shape[1]))
        acc = np.vstack((zs, acc, zs))

        self.tens['cp']['taps'] += zs.shape[0]

        # Bandpass filter each accelerometer signal
        for i in range(nacc):
            acc[:, i] = np.convolve(acc[:, i], self.tens['pp']['bf'], mode='same')

        # Run the Kalman filter, segment method
        sgn = lambda x: 1 - 2*x
        if kalseed == 'first':
            bkwd = 0
            i1 = 0
            i2 = ntaps
        else:
            bkwd = np.zeros((len(self.tens['cp']['taps']) - 1) * 2 + 2)
            i1 = np.zeros(len(self.tens['cp']['taps']) + 1)
            i2 = np.zeros(len(self.tens['cp']['taps']) + 1)
            
            if kalseed == 'fwdbkwd':
                bkwd[1] = 1
            else:
                basic = self.waveSpeedSTD(ad, acc)
                
                if kalseed == 'auto':
                    pass  # Implement 'auto' logic
                elif kalseed == 'manual':
                    pass  # Implement 'manual' logic
                else:
                    print('Invalid Kalman seed method.')
                    return
        
        # Run Kalman for each segment
        for s in range(len(i1)):
            seg = self.waveSpeedKF(ad, acc, int(i1[s]), int(i2[s]), bkwd[s])
            
            # Save which columns are which variables (same for all)
            if s == 0:
                nxt = 0
                col_s = list(range(1, seg['s'].shape[1] + 1))
                col_TA = list(range(seg['s'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + 1))
                col_TAvar = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + 1))
                col_TAB = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + 1))
                col_TABcorr = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + 1))
                col_TABvar = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + 1))
                col_dTA = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + 1))
                col_dTAcorr = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + seg['dTAcorr'].shape[1] + 1))
                col_dTAvar = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + seg['dTAcorr'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + seg['dTAcorr'].shape[1] + seg['dTAvar'].shape[1] + 1))
                col_wsk = list(range(seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + seg['dTAcorr'].shape[1] + seg['dTAvar'].shape[1] + 1, seg['s'].shape[1] + seg['TA'].shape[1] + seg['TAvar'].shape[1] + seg['TAB'].shape[1] + seg['TABcorr'].shape[1] + seg['TABvar'].shape[1] + seg['dTA'].shape[1] + seg['dTAcorr'].shape[1] + seg['dTAvar'].shape[1] + seg['wsk'].shape[1] + 1))
            segsA = np.zeros((len(seg['s']), col_wsk[-1]))

            segsA[:, col_s] = seg['s']
            segsA[:, col_TA] = seg['TA']
            segsA[:, col_TAvar] = seg['TAvar']
            segsA[:, col_TAB] = seg['TAB']
            segsA[:, col_TABcorr] = seg['TABcorr']
            segsA[:, col_TABvar] = seg['TABvar']
            segsA[:, col_dTA] = seg['dTA']
            segsA[:, col_dTAcorr] = seg['dTAcorr']
            segsA[:, col_dTAvar] = seg['dTAvar']
            segsA[:, col_wsk] = seg['wsk']
        
        self.tens['s'] = np.round(segsA[:, col_s])
        self.tens['lags']['TA'] = segsA[:, col_TA]
        self.tens['lags']['TAvar'] = segsA[:, col_TAvar]
        self.tens['lags']['TAB'] = segsA[:, col_TAB]
        self.tens['lags']['TABcorr'] = segsA[:, col_TABcorr]
        self.tens['lags']['TABvar'] = segsA[:, col_TABvar]
        self.tens['lags']['dTA'] = segsA[:, col_dTA]
        self.tens['lags']['dTAcorr'] = segsA[:, col_dTAcorr]
        self.tens['lags']['dTAvar'] = segsA[:, col_dTAvar]
        self.tens['wsk'] = segsA[:, col_wsk]
        
        # Time of the wave speed measurement
        self.tens['time'] = np.array(self.tens['cp']['taps']) / ad['freq']
        
        # Compute wave speeds via traditional cross-correlation approach
        D = np.diff(self.tens['cp']['accdis'])
        self.tens['ws'] = (np.ones((ntaps, 1)) * D) / self.tens['lags']['TAB']
        
        # Compute wave speeds via new auto-correlation approach
        qwer = np.vstack((self.tens['lags']['TA'][0, :], self.tens['lags']['TA'][:-1, :] + self.tens['lags']['dTA'][1:, :]))
        self.tens['wsa'] = self.tens['cp']['accdis'] / qwer

    def computeLag(self, A, B, freq, shift):
        # Find the lag time that maximizes normalized cross correlation
        # between a template A and search window B

        # Compute the normalized cross-correlations
        r = normxcorr1(self, A, B)
                
        # Finding the peak correlation
        peakCorr = np.max(r)
        peakInd = np.argmax(r)
                
        # Performing cosine interpolation to estimate lags with sub-frame
        # precision. See Cespedes et al., Ultrason Imaging 17, 142-171 (1995).
        if peakInd > 0 and peakInd < len(r) - 1:
            wo = np.arccos((r[peakInd-1] + r[peakInd+1]) / (2 * r[peakInd]))
            theta = np.arctan((r[peakInd-1] - r[peakInd+1]) / (2 * r[peakInd] * np.sin(wo)))
            delta = - theta / wo
            lagFrames = peakInd - 1 - shift + delta
        else:
            lagFrames = peakInd - 1 - shift
        
        # Time lag between wave arrival at two measurement locations
        lagTime = lagFrames / freq * 1000  # [ms]
        
        return lagFrames, lagTime, peakCorr, r

    def waveSpeedSTD(self, ad, acc):
        # Standard between-acc cross-correlation method. Used to give
        # estimate of best correlation points
        
        ntaps = len(self.tens['cp']['taps'])
        nacc = len(self.tens['cp']['accch'])
        
        seg = {}
        seg['TAB'] = np.zeros((ntaps, nacc - 1))
        seg['TABcorr'] = np.zeros((ntaps, nacc - 1))
        
        # for each accelerometer
        for i in range(ntaps):
            for j in range(nacc - 1):
                # between accelerometers
                template = acc[self.tens['cp']['taps'] + self.tens['pp']['templateSTD'], j]
                search = acc[self.tens['cp']['taps'][i] + self.tens['pp']['TABsearchSTD'], j + 1]
                lagFrames, _, seg['TABcorr'][i, j], _ = self.computeLag(template, search, ad['freq'], self.tens['pp']['TABshiftSTD'])
                seg['TAB'][i, j] = lagFrames
        
        # time
        seg['time'] = np.array(self.tens['cp']['taps']) / ad['freq']
        
        # wave speed
        D = np.diff(self.tens['cp']['accdis'])  # distance between accelerometers
        seg['ws'] = np.tile(D, (ntaps, 1)) / seg['TAB']
        
        return seg
    
    def waveSpeedKF(self, ad, acc, i1, i2, bkwd):
        # Use of Kalman filter to compute estimate of the time-varying wave speed
        # using:
        # dTA - tap-to-tap time lag at a single accelerometer
        # TAB - time for wave to travel between successive accelerometers
        
        nacc = len(self.tens['cp']['accch'])
        
        # instantiate
        seg = {}
        seg['s'] = np.zeros((i2, nacc))
        seg['TA'] = np.zeros((i2, nacc))
        seg['TAvar'] = np.zeros((i2, nacc))
        seg['TAB'] = np.zeros((i2, nacc - 1))
        seg['TABcorr'] = np.zeros((i2, nacc - 1))
        seg['TABvar'] = np.zeros((i2, nacc - 1))
        seg['dTA'] = np.zeros((i2, nacc))
        seg['dTAcorr'] = np.zeros((i2, nacc))
        seg['dTAvar'] = np.zeros((i2, nacc))
        seg['wsk'] = np.zeros((i2, 1))
        
        for ii in range(1, i2 - i1 + 2):
            # forward or backward
            if bkwd:
                i = i2 - ii + 1
                ilst = i + 1
            else:
                i = ii + i1 - 1
                ilst = i - 1

            # for each accelerometer
            for j in range(nacc):
                # set up adaptive window center
                if self.tens['pp']['winseed']:
                    if ii == 1:
                        seg['s'][i][ j] = -self.tens['pp']['window'][0] / 1000 * ad['freq']
                    else:
                        seg['s'][i][j] = np.round(seg['TA'][ilst][j] / 1000 * ad['freq'])
                else:
                    seg['s'][i][j] = 0
                s = seg['s'][i][j]

                # between accelerometers
                if j < nacc:
                    template = acc[self.tens['cp']['taps'][i] + self.tens['pp']['template'] + s][j]
                    search = acc[self.tens['cp']['taps'][i] + self.tens['pp']['TABsearch'] + s][j + 1]
                    lagframes, seg['TAB'][i, j], seg['TABcorr'][i, j] = self.computeLag(template, search, ad['freq'], self.tens['pp']['TABshift'])
                    seg['TABvar'][i, j] = np.dot(self.tens['pp']['TABvarp'], [seg['TABcorr'][i, j] ** 2, seg['TABcorr'][i, j], 1])

                # between events
                if ii > 1:
                    template = acc[self.tens['cp']['taps'][ilst] + self.tens['pp']['template'] + s, j]
                    search = acc[self.tens['cp']['taps'][i] + self.tens['pp']['dTAsearch'] + s, j]
                    lagframes, seg['dTA'][i, j], seg['dTAcorr'][i, j] = self.computeLag(template, search, ad['freq'], self.tens['pp']['dTAshift'])
                    seg['dTAvar'][i, j] = np.dot(self.tens['pp']['dTAvarp'], [seg['dTAcorr'][i, j] ** 2, seg['dTAcorr'][i, j], 1])

            # Kalman
            if ii == 1:
                # Initialize the estimate of the arrival times TA and TB, since DA and DB are  in mm, comes out in ms
                xe = (self.tens['cp']['accdis'] / np.mean(np.diff(self.tens['cp']['accdis']) / seg['TAB'][i, :])).reshape(-1, 1)
                seg['TA'][i, :] = xe.flatten()
                seg['TAvar'][i, :] = np.zeros_like(seg['TA'][i, :])

                # Matrices have been pre-defined 
                B = self.tens['kf']['B']
                F = self.tens['kf']['F']
                Q = self.tens['kf']['Q']
                H = self.tens['kf']['H']
                R = self.tens['kf']['R']
                P = self.tens['kf']['P']
            else:
                # Define the measurements z consisting of TAB, and speed constraints based on accelerometer locations
                z = np.concatenate((seg['TAB'][i, :], np.zeros((nacc - 1, 1))), axis=0)
                # Define the inputs u, change in wave arrival at accelerometers between successive taps
                u = seg['dTA'][i, :].reshape(-1, 1)

                # Q is the process noise covariance. It represents the amount of uncertainty in the model
                np.fill_diagonal(Q, np.diag(seg['dTAvar'][i, :]))

                # R is the measurement noise covariance. It represents the amount of uncertainty in the measurements
                np.fill_diagonal(R, np.diag(seg['TABvar'][i, :]))
                np.fill_diagonal(R[nacc - 1:, nacc - 1:], (self.tens['pp']['accdis_sd'] * np.mean(xe.flatten()) ** 2))

                # Kalman filter update
                xe, P = self.kalman_update(xe, P, B, F, Q, H, R, u, z)

                # Save the results
                seg['TA'][i, :] = xe.flatten()
                seg['TAvar'][i, :] = np.diag(P).flatten()

            if ii == 1:
                weights = np.ones_like(seg['TAvar'][i, :])
            else:
                weights = 1. / seg['TAvar'][i, :]
            u = seg['TA'][i, :]
            v = self.tens['cp']['accdis']
            sl_int = np.linalg.lstsq(np.column_stack((np.ones_like(u), u)), v, rcond=None)
            seg['wsk'][i] = sl_int[0][1]
        
        return seg

    def wsPlot(self, tens, acc, adfreq):
        # Plots time lage and wave speed results
        #
        # Darryl Thelen - 09/15/21

        # Define some things up front
        npts = acc.shape[0]
        ntaps = len(tens['cp']['taps'])
        nacc = len(tens['cp']['accch'])
        tacc = 1000 * np.arange(1, npts + 1) / adfreq

        # Generate the tap signal time series
        taps = np.full((npts, 1), np.nan)
        for i in range(ntaps - 2):
            taps[tens['cp']['taps'][i] + np.arange(tens['cp']['tapdur'] + 2)] = np.array([0] + [10] * tens['cp']['tapdur'] + [0])

        # FIRST FIGURE
        fig1, axs1 = plt.subplots(2 * (nacc - 1) + nacc, 1, figsize=(10, 6 * (2 * (nacc - 1) + nacc)))

        # Create the acceleration signals subplot
        kp = 0
        for i in range(nacc - 1):
            # Generate the template block time series
            templates = np.full((npts, 1), np.nan)
            templates2 = np.full((npts, 1), np.nan)
            for j in range(ntaps - 1):
                templates[tens['cp']['taps'][j] + tens['pp']['template'][0] + tens['s'][j][i] - 1] = 0
                templates[tens['cp']['taps'][j] + tens['pp']['template'] + tens['s'][j][i]] = 0
                templates[tens['cp']['taps'][j] + tens['pp']['template'][-1] + tens['s'][j][i] + 1] = 0

                templates2[tens['cp']['taps'][j] + tens['s'][j][i]] = 0

            # Create the raw accelerometer signal plot
            axs1[kp].plot(tacc, taps, color=[0.2, 0.2, 0.8], linewidth=1)
            axs1[kp].plot(tacc, templates, color=[0.7, 0, 0], linewidth=2)
            axs1[kp].plot(tacc, templates2, 'rx', linewidth=4)
            axs1[kp].plot(tacc, acc[:, i], 'k-')
            axs1[kp].plot(tacc, acc[:, i + 1], '-', color=[0, 0.7, 0])
            for k in range(ntaps - 1):
                kk = tens['cp']['taps'] + tens['pp']['template'] + tens['s'][k][i]
                axs1[kp].plot(tacc[kk] + tens['lags']['TAB'][k][i], acc[kk, i], 'k--')
            axs1[kp].set_xlabel('Time (ms)')
            axs1[kp].set_ylabel('Acc (m/s^2)')
            kp += 1

            # Create the TAB signal subplot
            axs1[kp].plot(tacc, tens['lags']['TAB'][:, i], 'r-')
            axs1[kp].set_xlabel('Time (ms)')
            axs1[kp].set_ylabel('TAB (ms)')
            kp += 1

        # Now the dTA signal plots
        for i in range(nacc):
            axs1[kp].plot(tacc, tens['lags']['dTA'][:, i], 'r-')
            axs1[kp].set_xlabel('Time (ms)')
            axs1[kp].set_ylabel('dTA (ms)')
            kp += 1

        plt.tight_layout()
        plt.show()

        # SECOND FIGURE
        fig2, axs2 = plt.subplots(nacc - 1 + 1, 1, figsize=(10, 6 * (nacc - 1 + 1)))

        kp = 0
        for i in range(nacc - 1):
            TAB = tens['lags']['TA'][:, i + 1] - tens['lags']['TA'][:, i]
            TABsd = np.sqrt((tens['lags']['TAvar'][:, i + 1] - tens['lags']['TAvar'][:, i]) / 2)
            axs2[kp].plot(1000 * tens['time'], tens['lags']['TAB'][:, i], 'r-')
            axs2[kp].plot(1000 * tens['time'], TAB, 'b-')
            axs2[kp].set_xlabel('Time (ms)')
            axs2[kp].set_ylabel('TAB (ms)')
            kp += 1

        axs2[kp].plot(1000 * tens['time'], tens['ws'][:, 0], 'r-')
        axs2[kp].plot(1000 * tens['time'], tens['ws'], 'r-')
        axs2[kp].plot(1000 * tens['time'], tens['wsa'][:, 0], 'g-')
        axs2[kp].plot(1000 * tens['time'], tens['wsa'], 'g-')
        axs2[kp].plot(1000 * tens['time'], tens['wsk'][:, 0], 'b-')
        axs2[kp].plot(1000 * tens['time'], tens['wsk'], 'b-')
        axs2[kp].set_xlabel('Time (ms)')
        axs2[kp].set_ylabel('Wave Speed (m/s)')
        axs2[kp].legend(['Cross Corr A-B', 'Auto Corr A-A', 'Kalman Filter'])

        plt.tight_layout()
        plt.show()

    def fsID(self, ad):
        # Define some things up front
        raw = ad.data[:, self.fs.accch]
        t_raw = np.arange(len(raw))
        ref = self.tens.wsk
        t_ref = np.arange(len(ref))

        # set up the FS collection plot
        # Code for plotting skipped as it varies depending on the plotting library used in Python

        # user input for start and end of stride
        print("Select start and ~75% of instrumented foot stride")
        # Code for getting user input using matplotlib or other libraries

        pkIndexStart = round(pkIndexStart)
        pkIndexEnd = round(pkIndexEnd)

        # cross correlate template to FS signal
        template = raw[pkIndexStart:pkIndexEnd + 1]
        rSig = normxcorr1(raw, template, mode='valid')

        # Find peaks in smoothed cross-correlation r-values
        # locs, pks = find_peaks(rSig, distance=pkIndexEnd - pkIndexStart)

        # align everything based on the selected stride 1
        startStride = np.argmax(rSig)
        startLocsIndex = np.argmin(np.abs(locs - startStride))
        # locs_acc = locs[startLocsIndex:]

        t_raw = np.linspace(0, len(raw) / ad.freq, len(raw))
        t_ref = np.linspace(0, len(ref) * self.tens.cp.tapms / self.tens.cp.tapdc / 1000, len(ref))
        fsidx = []
        # for loc_acc in locs_acc:
        #     fsidx.append(np.argmin(np.abs(t_ref - t_raw[loc_acc])))

        # self.fs.fsidx = np.array(fsidx)
        # self.fs.fsidx_acc = locs_acc


def normxcorr1(A, B):
    # Length of template and search
    na = len(A)
    nb = len(B)
    # Process the mean and sum of squares of the template signal
    Am = np.sum(A) / na
    sumAA = np.dot(A, A) / na
    # Now pre-compute some quantities for the search signal
    ii = np.arange(na)
    Bm = np.sum(B[ii]) / na
    BB = B * B / na
    sumBB = np.sum(BB[ii])
    # Convolution to do the cross correlation of A and B over all potential shifts
    ABconv = np.convolve(B, A[::-1], 'valid') / na
    # initialize r and compute the first r value
    r = np.zeros(nb - na + 1)
    sumAB = ABconv[0]
    r[0] = (sumAB - Am * Bm) / np.sqrt((sumAA - Am**2) * (sumBB - Bm**2))
    # Now incrementally go through and compute r for the other shifts
    for i in range(nb - na):
        Bm = Bm - B[i] / na + B[na + i] / na
        sumAB = ABconv[i + 1]
        sumBB = sumBB - BB[i] + BB[na + i]
        r[i + 1] = (sumAB - Am * Bm) / np.sqrt((sumAA - Am**2) * (sumBB - Bm**2))
    return r






