#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
import datetime
from math import radians, cos, sin, asin, sqrt
# import pandas as pd
from scipy.sparse.linalg import eigs


def standardization(data):
    # mu = np.mean(data, axis=0)
    # sigma = np.std(data, axis=0)
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def haversine(lon1, lat1, lon2, lat2): # 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 

    return c * r * 1000

def str2time_T(timestr):
    [date, time_of_date] = timestr.split('T')
    [year, month, day] = date.split('-')
    [hour, min, sec] = time_of_date.split(':')
    date_time = datetime.datetime(int(year), int(month), int(day), int(hour), int(min), int(sec))
    return date_time

def str2time(timestr):
    [date, time_of_date] = timestr.split(' ')
    [year, month, day] = date.split('-')
    [hour, min, sec] = time_of_date.split(':')
    date_time = datetime.datetime(int(year), int(month), int(day), int(hour), int(min), int(sec))
    return date_time

def time2ts(date_time):
    return (date_time.hour*3600 + date_time.minute*60 + date_time.second) / (15*60)

def plot_cdf(x):
    ecdf = sm.distributions.ECDF(x)
    # print(ecdf)
    x = np.linspace(min(x), max(x))
    y = ecdf(x)
    # print(x)
    # print(y)

    # plt.step(x, y)
    # plt.show()

    # return x, y

def checkzeromatrix(matrix):
    if np.where(matrix!=0)[0].shape[0] == 0:
        print('is a zeros matrix')
    else:
        print('is not a zeros matrix')

# def checkzeromatrix(matrix):
#     return np.where(matrix!=0)[0].shape[0]

# print(haversine(113.6793565, 22.2573262, 113.6793565, 22.8644043))

def cheb_polynomial(L_sub, L_bike, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_sub.shape[0]

    L = np.hstack((L_bike, L_sub))

    iden = np.hstack((np.identity(N), np.identity(N)))

    cheb_polynomials = [iden, L.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def get_batches(data, batch_size, seq_len):
    row_total = data.shape[0]
    row_sequence = np.random.choice(row_total-seq_len, batch_size, replace=False, p=None)
    data_list = [data[np.newaxis, :, :, x:x+seq_len] for x in row_sequence]
    data1 = np.concatenate(data_list, axis=0)
    return data1

    # d[1,2,3] -> d[1,2,3,1] d = [:,:,:,np.new]