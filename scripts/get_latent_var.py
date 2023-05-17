# -*- coding: utf-8 -*-



import os
import sys
import time
import datetime
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from read_data import read_single_anomaly_case

from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
#sys.path.append(sys.path[0]+"/../")
sys.path.insert(0, sys.path[0]+"/../")
from model.SCVAE_model import SCVAE


def get_hidden_z(model, dataloader, case="", mode="post", site_type="trainsites"):
    print("*"*20+"get hidden variable")
    model.eval()
    with torch.no_grad():
        re_zs = []
        for batch_idx, (data, y_data) in enumerate(dataloader):
            # to (T,batch_size,feature_dim,input_dim)
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            model(data, y_data)
            # print(mu.shape) #(1,72,3)
            if mode == "post":
                re_z = np.array([item.cpu().numpy() for item in model.Z_mean])
            elif mode == "prior":
                re_z = np.array([item.cpu().numpy() for item in model.pZ_mean])
            elif mode == "predict":
                re_z = np.array([item.cpu().numpy()
                                 for item in model.Z_mean_predict])
            re_z = np.transpose(re_z, (1, 0, 2))
            print(re_z.shape)
            re_zs.append(re_z)

        re_zs = np.concatenate(re_zs, axis=0)

        print("re_zs.shape", re_zs.shape)
        save_dir = "re_z"
        os.makedirs(save_dir, exist_ok=True)
        np.save("re_z/re_"+mode+"_zs_"+site_type+"_"+case, re_zs)
        return re_zs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saved_model/SCVAE_savedmodel.pth",
                        help="saved model path")
    parser.add_argument("--reg", type=float, default=0,
                        help='smooth canonical intensity')
    parser.add_argument("--batch_size", type=int,
                        default=256, help="batch size")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="device, e.g. cpu, cuda:0, cuda:1")
    parser.add_argument("--learning_rate", type=float,
                        default=0.000001, help='Adam optimizer learning rate')
    parser.add_argument("--print_every", type=int, default=1,
                        help='The number of iterations in which the result is printed')
    parser.add_argument("--n_epochs", type=int, default=10000,
                        help='maximum number of iterations')
    parser.add_argument("--h_dim", type=int, default=512,
                        help='dimension in Neural network hidden layers ')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='dimensions of latent variable')
    parser.add_argument("--seq_shape", type=tuple, default=(-1, 12, 6),
                        help="the sequence shape, e.g. (-1,12,6)")
    parser.add_argument("--test_ratio", type=float, default=1,
                        help="the test ratio in data_set")
    parser.add_argument("--normal_ratio", type=int, default=90,
                        help="the nomal_ratio % in dataset")
    parser.add_argument("--mode", type=int, default=2,
                        help="the mode when train")
    parser.add_argument("--is_predict", type=bool,
                        default=False, help="whether use predict result ot not")
    parser.add_argument("--is_prior", type=bool, default=False,
                        help="whether use prior to reconstruct ot not")
    parser.add_argument("--is_scale_data", type=bool,
                        default=False, help="whether get scale data")
    parser.add_argument("--is_simulate_data", type=bool, default=True,
                        help="whether use simulate data or predict data")
    parser.add_argument("--is_trainsites", type=bool, default=True,
                        help="whether use trainsites or testsites")
    parser.add_argument("--anomaly_type_detected", type=str,
                        default='all', help="The anomaly type to be detected")
    opt = parser.parse_args()

    is_predict = opt.is_predict
    is_prior = opt.is_prior
    is_simulate = opt.is_simulate_data
    mode = opt.mode
    is_scale = opt.is_scale_data
    is_trainsites = opt.is_trainsites

    seq_shape = opt.seq_shape
    input_dim = seq_shape[-1]
    seq_len = seq_shape[-2]

    device = torch.device(opt.device)
    print(device)
    anomaly_mask = np.array([])
    date_series = np.array([])
    site_num_series = np.array([])
    ra_series_mean = np.array([])
    pow_series_mean = np.array([])
    tprt_series_mean = np.array([])
    ra_series_std = np.array([])
    pow_series_std = np.array([])
    tprt_series_std = np.array([])

    pow_series_simulate = np.array([])
    pow_series_simulate_mean = np.array([])
    pow_series_simulate_std = np.array([])

    case = ["normal", "spike", "inverterFault",
            "snowy", "cloudy", "lowValue", "shading"]

    for case_index in range(len(case)):

        if is_trainsites:
            File_list = ["enhance_12times_smooth_intense_normal_data_of_20trainsites_clean.csv", "enhance_12times_spike_anomaly_data_of_20trainsites.csv", "enhance_12times_inverter_fault_anomaly_data_of_20trainsites.csv",
                         "enhance_12times_snowy_anomaly_data_of_20trainsites.csv", "enhance_12times_cloudy_anomaly_data_of_20trainsites.csv", "enhance_12times_lowValue_anomaly_data_of_20trainsites.csv", "enhance_12times_shading_anomaly_data_of_20trainsites.csv"]
        else:
            File_list = ["enhance_12times_smooth_intense_normal_data_of_10testsites_clean.csv", "enhance_12times_spike_anomaly_data_of_10testsites.csv", "enhance_12times_inverter_fault_anomaly_data_of_10testsites.csv",
                         "enhance_12times_snowy_anomaly_data_of_10testsites.csv", "enhance_12times_cloudy_anomaly_data_of_10testsites.csv", "enhance_12times_lowValue_anomaly_data_of_10testsites.csv", "enhance_12times_shading_anomaly_data_of_10testsites.csv"]

        ra_series_scale, pow_series_scale, tprt_series_scale, anomaly_mask = read_single_anomaly_case(
            File_list[case_index], test=False, include_normal=False)
        ra_per_day_seq = ra_series_scale.reshape(seq_shape)
        pow_per_day_seq = pow_series_scale.reshape(seq_shape)
        tprt_per_day_seq = tprt_series_scale.reshape(seq_shape)

        if len(list(anomaly_mask)) != 0:
            print("anomaly_mask not empty")
            anomaly_mask = anomaly_mask.reshape(-1, seq_len*input_dim)

        if opt.test_ratio != 1:
            x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(
                ra_per_day_seq, tprt_per_day_seq, pow_per_day_seq, test_size=opt.test_ratio, random_state=42)
            if len(list(anomaly_mask)) != 0:
                anomaly_mask_train, anomaly_mask_test = train_test_split(
                    anomaly_mask, test_size=opt.test_ratio, random_state=42)
        else:
            x_train1 = x_test1 = ra_per_day_seq
            x_train2 = x_test2 = tprt_per_day_seq
            y_train = y_test = pow_per_day_seq
            if len(list(anomaly_mask)) != 0:
                anomaly_mask_train = anomaly_mask_test = anomaly_mask

        multi_x_train = np.stack([x_train1, x_train2], axis=3)
        multi_x_test = np.stack([x_test1, x_test2], axis=3)
        multi_y_train = np.stack([y_train], axis=3)
        multi_y_test = np.stack([y_test], axis=3)

        x_dim = multi_x_train.shape[-1]
        y_dim = multi_y_train.shape[-1]
        h_dim = opt.h_dim
        z_dim = opt.z_dim
        n_epochs = opt.n_epochs
        learning_rate = opt.learning_rate
        print_every = opt.print_every
        chunk_torch = torch.FloatTensor(multi_x_train)  # numpy to tensor
        test_torch = torch.FloatTensor(multi_x_test)
        batch_size = opt.batch_size
        import torch.utils.data

        train_loader_ordered = torch.utils.data.DataLoader(TensorDataset(
            chunk_torch, torch.FloatTensor(multi_y_train)), batch_size=batch_size, shuffle=False)
        test_loader_ordered = torch.utils.data.DataLoader(TensorDataset(
            test_torch, torch.FloatTensor(multi_y_test)), batch_size=batch_size, shuffle=False)

        model = SCVAE(x_dim, y_dim, h_dim, z_dim, input_dim, 1,
                      device=device, is_prior=False).to(device)

        torch.cuda.empty_cache()

        reg = opt.reg

        # model_path = "./saved_model/SCVAE_savedmodel"  # mode2
        model_path = opt.model_path
        model.load_state_dict(torch.load(model_path, map_location=opt.device))
        model.eval()
        show_dataloader = test_loader_ordered
        show_x = multi_x_test
        show_y = y_test
        if len(list(anomaly_mask)) != 0:
            anomaly_mask = anomaly_mask_test

        site_type = "trainsites" if is_trainsites else "testsites"
        for mode in ["prior", "predict", "post"]:
            re_zs = get_hidden_z(model, show_dataloader,
                                 case=case[case_index], mode=mode, site_type=site_type)
