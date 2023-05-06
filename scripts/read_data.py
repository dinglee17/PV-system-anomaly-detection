# -*- coding: utf-8 -*-
"""
@author: DingLi
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os


def read_single_data_noScale(filename="site_30100005_1.csv", label=False):
    root = "../data/"
    data_path = root+filename
    series_data = pd.read_csv(data_path, sep=',')
    ra_series = series_data["instantaneous_global_radiation"].values
    pow_series = series_data["instantaneous_output_power"].values
    tprt_series = series_data["clearsky_Tamb"].values
    if label == True:
        anomaly_mask = series_data["label"].values
        return ra_series, pow_series, tprt_series, anomaly_mask

    return ra_series, pow_series, tprt_series


def read_single_anomaly_case(filename="spike_anomaly_data_of_20sites.csv", test=False, include_normal=True):

    ra_series_1, pow_series_1, tprt_series_1, anomaly_mask1 = read_single_data_noScale(
        filename, True)

    if include_normal == True:

        if test == False:
            ra_series_0, pow_series_0, tprt_series_0, anomaly_mask0 = read_single_data_noScale(
                "enhance_12times_smooth_intense_normal_data_of_20trainsites_clean.csv", True)
        else:
            ra_series_0, pow_series_0, tprt_series_0, anomaly_mask0 = read_single_data_noScale(
                "enhance_12times_smooth_intense_normal_data_of_10testsites_clean.csv", True)

        ra_series_whole = np.concatenate([ra_series_0[:], ra_series_1])
        pow_series_whole = np.concatenate([pow_series_0[:], pow_series_1])
        tprt_series_whole = np.concatenate([tprt_series_0[:], tprt_series_1])
        anomaly_mask_whole = np.concatenate([anomaly_mask0[:], anomaly_mask1])

    else:
        ra_series_whole = ra_series_1
        pow_series_whole = pow_series_1
        tprt_series_whole = tprt_series_1
        anomaly_mask_whole = anomaly_mask1

    return ra_series_whole, pow_series_whole, tprt_series_whole, anomaly_mask_whole


def read_multi_anomaly_data(test=False):

    root = ""
    if test == False:
        ra_series_0, pow_series_0, tprt_series_0, anomaly_mask0 = read_single_data_noScale(
            root+"enhance_12times_smooth_intense_normal_data_of_20trainsites_clean.csv", True)
        ra_series_1, pow_series_1, tprt_series_1, anomaly_mask1 = read_single_data_noScale(
            root+"enhance_12times_spike_anomaly_data_of_20trainsites.csv", True)
        ra_series_2, pow_series_2, tprt_series_2, anomaly_mask2 = read_single_data_noScale(
            root+"enhance_12times_inverter_fault_anomaly_data_of_20trainsites.csv", True)
        ra_series_3, pow_series_3, tprt_series_3, anomaly_mask3 = read_single_data_noScale(
            root+"enhance_12times_snowy_anomaly_data_of_20trainsites.csv", True)
        ra_series_4, pow_series_4, tprt_series_4, anomaly_mask4 = read_single_data_noScale(
            root+"enhance_12times_cloudy_anomaly_data_of_20trainsites.csv", True)
        ra_series_5, pow_series_5, tprt_series_5, anomaly_mask5 = read_single_data_noScale(
            root+"enhance_12times_lowValue_anomaly_data_of_20trainsites.csv", True)
        ra_series_6, pow_series_6, tprt_series_6, anomaly_mask6 = read_single_data_noScale(
            root+"enhance_12times_shading_anomaly_data_of_20trainsites.csv", True)

    else:
        ra_series_0, pow_series_0, tprt_series_0, anomaly_mask0 = read_single_data_noScale(
            root+"enhance_12times_smooth_intense_normal_data_of_10testsites_clean.csv", True)
        ra_series_1, pow_series_1, tprt_series_1, anomaly_mask1 = read_single_data_noScale(
            root+"enhance_12times_spike_anomaly_data_of_10testsites.csv", True)
        ra_series_2, pow_series_2, tprt_series_2, anomaly_mask2 = read_single_data_noScale(
            root+"enhance_12times_inverter_fault_anomaly_data_of_10testsites.csv", True)
        ra_series_3, pow_series_3, tprt_series_3, anomaly_mask3 = read_single_data_noScale(
            root+"enhance_12times_snowy_anomaly_data_of_10testsites.csv", True)
        ra_series_4, pow_series_4, tprt_series_4, anomaly_mask4 = read_single_data_noScale(
            root+"enhance_12times_cloudy_anomaly_data_of_10testsites.csv", True)
        ra_series_5, pow_series_5, tprt_series_5, anomaly_mask5 = read_single_data_noScale(
            root+"enhance_12times_lowValue_anomaly_data_of_10testsites.csv", True)
        ra_series_6, pow_series_6, tprt_series_6, anomaly_mask6 = read_single_data_noScale(
            root+"enhance_12times_shading_anomaly_data_of_10testsites.csv", True)

    ra_series_whole = np.concatenate(
        [ra_series_0, ra_series_1, ra_series_2, ra_series_3, ra_series_4, ra_series_5, ra_series_6])
    pow_series_whole = np.concatenate(
        [pow_series_0, pow_series_1, pow_series_2, pow_series_3, pow_series_4, pow_series_5, pow_series_6])
    tprt_series_whole = np.concatenate(
        [tprt_series_0, tprt_series_1, tprt_series_2, tprt_series_3, tprt_series_4, tprt_series_5, tprt_series_6])
    anomaly_mask_whole = np.concatenate(
        [anomaly_mask0, anomaly_mask1, anomaly_mask2, anomaly_mask3, anomaly_mask4, anomaly_mask5, anomaly_mask6])

    return ra_series_whole, pow_series_whole, tprt_series_whole, anomaly_mask_whole
