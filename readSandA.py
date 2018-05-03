# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 15:49:29 2017

@author: saeed
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import math

plt.close("all")



################### Sport and Daily Activity dataset ####################
##### dataset path ######
fn = '/Users/mac/Desktop/Element_Project/Networ Sience SSL/'

### activity labels
acts = ['a01', 'a02', 'a03', 'a04', 'a05','a06', 'a07','a08','a09','a10', 'a11','a12','a13','a14','a15','a16', 'a17', 'a18', 'a19']

### segment labels
segs = ['s01', 's02', 's03', 's04', 's05','s06', 's07','s08','s09','s10', 's11','s12','s13','s14','s15','s16', 's17', 's18', 's19', 's20',
        's21', 's22', 's23', 's24', 's25','s26', 's27','s28','s29','s30', 's31','s32','s33','s34','s35','s36', 's37', 's38', 's39', 's40',
        's41', 's42', 's43', 's44', 's45','s46', 's47','s48','s49','s50', 's51','s52','s53','s54','s55','s56', 's57', 's58', 's59', 's60',]

### sensor names
sensors = ['acc','gyro', 'magnet']

#### the axis index in each file for different sensor locations
locs = {"T": 0, "RA" : 9, "LA" : 18,"RL" : 27,"LL":36}

### user ids in the dataset format
users = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']

###############################################################


def selected_locations(nodes, locs, num_sensors):
    start = 1
    for n in nodes:
        if start == 1:
            start = 0
            idx = np.arange(locs[n], locs[n] + 9)
        else:
            idx = np.concatenate((idx, np.arange(locs[n], locs[n] + 3 * num_sensors)))
    return idx

def extract_features(x):
    numrows = len(x)    # 3 rows in your example
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    var = np.var(x, axis = 0)
    median = np.median(x, axis = 0)
    xmax = np.amax(x, axis = 0)
    xmin =np.amax(x, axis = 0)
    p2p = xmax - xmin
    amp = xmax - mean
    s2e = x[numrows-1,:] - x[0,:]
    features = np.concatenate([mean, std, var, median, xmax, xmin, p2p, amp, s2e])#, morph]);
    return features





def data_generate(users_idx, acts_idx, segs_idx, fn, nodes, num_sensors):
    start = 0
    idx = selected_locations(nodes, locs, num_sensors)

    for u in users_idx:
        for s in segs_idx:
            for a in acts_idx:
                filename = fn + acts[a] + '/' + users[u] + '/' + segs[s] + '.txt'
                raw_data = np.genfromtxt(filename, delimiter=',', skip_header=0)
                raw_data = raw_data[:,idx]
                f = extract_features(raw_data)
                if start == 0:
                    fsp = f
                    label = a + 1
                    start = 1
                else: 
                    fsp = np.vstack((fsp, f))
                    label = np.vstack((label, a+1))
    return scale(fsp, axis = 0), label



##################### REAL WORLD DATASET ####################

fn1 = 'D:/Python Scripts/ManifoldLearning/realworld_data_selected/S'
def feature_space_real_world_data(sub, sensor, act, loc, skip_columns, fn, seg_s, ol):
    start = 0
    fspace = []
    for l in sub:
        for i in sensor:
            for j in act:
                for k in loc:
                    file_name = fn + str(l) + '/' + sensors[i] + '_' + acts[j] + '_' + locs[k] + '.csv'
                    raw_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
                    raw_data = raw_data[:, skip_columns:]
                    idxx = int(math.floor(raw_data.shape[0] / (seg_s - ol)))

                    for idx in range(idxx):
                        seg = raw_data[idx * (seg_s - ol):idx * (seg_s - ol) + seg_s, :]
                        features = extract_features(seg)
                        features = np.append(features, j)
                        if start == 0:
                            fspace = features
                            start = 1
                        else:
                            fspace = np.vstack((fspace, features))
    return fspace;
