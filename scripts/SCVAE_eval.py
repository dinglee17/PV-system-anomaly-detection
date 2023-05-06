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
from read_data import read_multi_anomaly_data

from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append("..")
from SCVAE_model.SCVAE_model import SCVAE


def mean_relative_error(y_true, y_pred):
    relative_error = np.average(np.abs(y_true - y_pred), axis=0)
    relative_error = relative_error / \
        (np.max(y_true, axis=0)-np.min(y_true, axis=0)+0.1)
    return relative_error


def local_relative_error(y_true, y_pred, local_points=1):

    local_relative_error = np.average(
        np.sort(np.abs(y_true - y_pred), axis=0)[-local_points:], axis=0)
    local_relative_error = local_relative_error / \
        (np.max(y_true, axis=0)-np.min(y_true, axis=0)+0.1)
    return local_relative_error


def max_absolute_error(y_true, y_pred):
    max_absolute_error = np.max(np.abs(y_true - y_pred), axis=0)
    return max_absolute_error


def reconstruct(model, dataloader, is_prior=False):
    print("*"*20+"reconstruct result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data, y_data) in enumerate(dataloader):
            # to (T,batch_size,feature_dim,input_dim)
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            mu, std, score = model.reconstruct(data, y_data, is_prior=is_prior)
            mu = np.transpose(mu, (1, 0, 2))  # to (batch_size,T,M)
            std = np.transpose(std, (1, 0, 2))
            score = np.transpose(score, (1, 0, 2))

            mus.append(mu)
            stds.append(std)
            scores.append(score)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        scores = np.concatenate(scores, axis=0)
        print("scores.shape", scores.shape)
        return mus, stds, scores


def predict_withLabel(model, dataloader):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            mu, std, score = model.predict_withLabel(data, y_data)
            mu = np.transpose(mu, (1, 0, 2))
            std = np.transpose(std, (1, 0, 2))
            score = np.transpose(score, (1, 0, 2))

            mus.append(mu)
            stds.append(std)
            scores.append(score)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        scores = np.concatenate(scores, axis=0)
        print("scores.shape", scores.shape)

        return mus, stds, scores


def predict(model, dataloader):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            mu, std = model.predict(data, y_data, is_prior=False)
            mu = np.transpose(mu, (1, 0, 2))
            std = np.transpose(std, (1, 0, 2))
            mus.append(mu)
            stds.append(std)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        print("scores.shape", scores.shape)
        return mus, stds


def get_hidden_z(model, dataloader, case="", mode="post"):
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
        np.save("re_z/re_"+mode+"_zs_enhance_testsites_12times_"+case, re_zs)
        return re_zs


def draw_anomal(ra, tprt, pow_, predict, label, index, y_label1, y_label2, savefig=None):
    fig = plt.figure(index, figsize=(20, 8))

    ax = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 1)
    ax4 = fig.add_subplot(1, 3, 3)

    x = np.arange(pow_.shape[0])

    ax.scatter(x, pow_, s=10, c='r', alpha=1)
    color = 'r'
    ax.plot(x, pow_, c=color, lw=1.5)

    ax2.scatter(x, ra, s=10, c='b', alpha=1)
    color = 'b'
    ax2.plot(x, ra, c=color, lw=1.5)

    color = 'g' if label == 0 else 'r'
    ax3.scatter(x, predict, s=10, c=color, alpha=1)
    ax3.plot(x, predict, c=color, lw=1.5)

    ax4.scatter(x, tprt, s=10, c='b', alpha=1)
    color = 'b'
    ax4.plot(x, tprt, c=color, lw=1.5)

    ax.set_xlabel('time')
    ax.set_ylabel(y_label1)
    title = "data per 15 minute"
    ax.set_title(title)

    ax2.set_xlabel('time')
    ax2.set_ylabel(y_label2)
    title = "data per 15 minute"
    ax2.set_title(title)
    plt.tight_layout()
    plt.show()


def draw_anomaly_withMask(ra, tprt, pow_, predict, label, index, y_label1, y_label2, anomaly_mask, savefig=None):

    fig = plt.figure(index, figsize=(18, 8))
    ax = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    #ax3 = fig.add_subplot(1, 3, 1)
    ax4 = fig.add_subplot(1, 3, 3)

    s_r = 50
    l_w = 2.5
    blue = "#0343DF"
    red = '#C82423'
    # red="#943c39"

    anomaly_map = {0: "normal", 1: "spike", 2: "no_output",
                   3: "snowy", 4: "cloudy", 5: "low_value", 6: "shading"}
    detected_map = {0: "normal", 1: "anomaly"}

    x = np.arange(pow_.shape[0])

    color = blue
    ax.scatter(x, pow_, s=s_r, c=color, alpha=1,
               label=f"Label:{anomaly_map[anomaly_mask[0]]}")
    ax.plot(x, pow_, c=color, lw=l_w)

    ax2.scatter(x, ra, s=s_r, c='#0343DF', alpha=1)
    color = blue
    ax2.plot(x, ra, c=color, lw=l_w)

    color = red if label == 1 else 'g'
    ax.scatter(x, predict, s=s_r, c=color, alpha=1,
               label=f"Detected:{detected_map[label]}")
    ax.plot(x, predict, c=color, lw=l_w)

    ax4.scatter(x, tprt, s=s_r, c='#0343DF', alpha=1)
    color = blue
    ax4.plot(x, tprt, c=color, lw=l_w)

    # ax.set_xticks(fontsize=20)
    # ax.set_yticks(fontsize=20)
    # 03:45-21:30
    ax.set_xticks([0, 36, 72])
    ax.set_xticklabels(['3:45', "12:37", '21:30'])
    ax2.set_xticks([0, 36, 72])
    ax2.set_xticklabels(['3:45', "12:37", '21:30'])
    ax4.set_xticks([0, 36, 72])
    ax4.set_xticklabels(['3:45', "12:37", '21:30'])

    # ax.set_xlabel('time',fontsize=30)
    ax.set_ylabel("value(scaled)", fontsize=30)
    ax.set_title("power", y=1.05, fontsize=30)

    ax2.set_xlabel('time', labelpad=20, fontsize=30)
    # ax2.set_ylabel("value(scaled)",fontsize=20)
    ax2.set_title("irradiation", y=1.05, fontsize=30)

    # ax4.set_xlabel('time',fontsize=30)
    # ax2.set_ylabel("value(scaled)",fontsize=20)
    ax4.set_title("temperature", y=1.05, fontsize=30)

    fontsize = 28
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    ax4.tick_params(labelsize=fontsize)

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0)
    plt.show()
    # fig.savefig(f"SCVAE_{anomaly_map[anomaly_mask[0]]}.pdf")
    if savefig is not None:
        fig.savefig(f"SCVAE_{anomaly_map[anomaly_mask[0]]}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../saved_model/SCVAE_savedmodel.pth",
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
    parser.add_argument("--whether_get_z", type=bool,
                        default=True, help="whether get hidden variables")
    parser.add_argument("--is_scale_data", type=bool,
                        default=False, help="whether get scale data")
    parser.add_argument("--is_simulate_data", type=bool, default=True,
                        help="whether use simulate data or predict data")
    parser.add_argument("--anomaly_type_detected", type=str,
                        default='all', help="The anomaly type to be detected")
    opt = parser.parse_args()

    whether_get_z = opt.whether_get_z
    is_predict = opt.is_predict
    is_prior = opt.is_prior
    is_simulate = opt.is_simulate_data
    mode = opt.mode
    is_scale = opt.is_scale_data
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
    case_index = 2
    ra_series_scale, pow_series_scale, tprt_series_scale, anomaly_mask = read_multi_anomaly_data(
        test=True)
    # ra_series_scale,pow_series_scale,tprt_series_scale=read_whole_enhance_data(nums=21)

    ra_per_day_seq = ra_series_scale.reshape(seq_shape)
    pow_per_day_seq = pow_series_scale.reshape(seq_shape)
    tprt_per_day_seq = tprt_series_scale.reshape(seq_shape)

    # print("pow_series_mean",pow_series_mean)

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

    print(f"data shape：{multi_x_train.shape}")
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

    # model_path = "./saved_model/SCVAE_savedmodel.pth"  # mode2
    model_path = opt.model_path
    model.load_state_dict(torch.load(model_path, map_location=opt.device))
    model.eval()
    show_dataloader = test_loader_ordered
    show_x = multi_x_test
    show_y = y_test
    if len(list(anomaly_mask)) != 0:
        anomaly_mask = anomaly_mask_test

    if whether_get_z == True:
        for mode in ["prior", "predict", "post"]:
            re_zs = get_hidden_z(model, show_dataloader,
                                 case=case[case_index], mode=mode)

    if not is_predict:
        mus, stds, scores = reconstruct(
            model, show_dataloader, is_prior)  # mode0 or mode1
    else:
        mus, stds, scores = predict_withLabel(model, show_dataloader)

    predict_ = np.squeeze(mus).reshape(-1, input_dim *
                                       seq_len)  # predicted value
    pow_ = np.squeeze(show_y).reshape(-1, input_dim*seq_len)  # real value

    mse = mean_squared_error(pow_, predict_)
    print("mse", mse)

    detect_anomaly = 0
    nomal_ratio = opt.normal_ratio
    predict_transpose = predict_.T
    pow_transpose = pow_.T
    #print(predict_transpose.shape, pow_transpose.shape)

    # anomal_score=mean_relative_error(pow_transpose,predict_transpose)
    anomal_score = local_relative_error(pow_transpose, predict_transpose)
    # anomal_score=max_absolute__error(pow_transpose,predict_transpose) #absolute_max_error

    print("anomal_score.shape", anomal_score.shape)

    if len(list(anomaly_mask)) != 0:
        label = np.sum(anomaly_mask, axis=1)

        detect_type = opt.anomaly_type_detected
        if detect_type == "spike":
            index = np.where((label == 1*72) | (label == 0))[0]
        elif detect_type == "no_output":
            index = np.where((label == 2*72) | (label == 0))[0]
        elif detect_type == "snowy":
            index = np.where((label == 3*72) | (label == 0))[0]
        elif detect_type == "cloudy":
            index = np.where((label == 4*72) | (label == 0))[0]
        elif detect_type == "low_value":
            index = np.where((label == 5*72) | (label == 0))[0]
        elif detect_type == "shading":
            index = np.where((label == 6*72) | (label == 0))[0]
        elif detect_type == "all":
            index = np.where((label >= 0))[0]
        else:
            raise RuntimeError('AnomalyType Undifined')

        label_part = label[index]
        print(label_part.shape)
        anomal_score = anomal_score[index]
        mus = mus[index]
        show_x = show_x[index]
        show_y = show_y[index]
        anomaly_mask = anomaly_mask[index]

        sum_num = np.sum(label)
        label = label_part
        label[label > 0] = 1
        num = np.sum(label)
        roc = roc_auc_score(label, anomal_score)
        prc = average_precision_score(label, anomal_score)
        print("roc:", roc)
        print("prc:", prc)

    print("***************average anomaly score*****************",
          np.mean(anomal_score))
    percentile_1 = np.percentile(anomal_score, nomal_ratio)
    label = anomal_score.copy()
    label[label > percentile_1] = 1
    label[label <= percentile_1] = 0

    # anomaly samples
    anomal_inst = [i for i, j in enumerate(label) if j == 1]
    print("len(anomal_inst)", len(anomal_inst))
    for i, anomal in enumerate(anomal_inst):
        model.eval()
        with torch.no_grad():

            predict_ = np.squeeze(mus[anomal]).reshape((72))
            loss = anomal_score[anomal]
            print("anomaly score", loss)

        ra = np.squeeze(show_x[anomal, :, :, 0]).reshape((72))
        tprt = np.squeeze(show_x[anomal, :, :, 1]).reshape((72))
        pow_ = show_y[anomal].reshape((72))

        if len(list(anomaly_mask)) != 0:
            anomaly_mask_ = anomaly_mask[anomal].reshape((72))
            if anomaly_mask_.sum() != 0:
                detect_anomaly += 1
            draw_anomaly_withMask(ra, tprt, pow_, predict_,
                                  1, 2*i, 'pow', 'ra', anomaly_mask_)

        else:
            draw_anomal(ra, tprt, pow_, predict_, 1, 2*i, 'pow', 'ra')

    print(detect_anomaly)
    percentile_2 = np.percentile(-anomal_score, nomal_ratio)
    label = -anomal_score.copy()
    label[label <= percentile_2] = 0
    label[label > percentile_2] = 1
    nomal_inst = [i for i, j in enumerate(label) if j == 1]
    print("len(nomal_inst)", len(nomal_inst))
    for i, nomal in enumerate(nomal_inst):
        model.eval()
        with torch.no_grad():
            predict_ = np.squeeze(mus[nomal]).reshape((72))
            loss = anomal_score[nomal]
            print("anomaly score", loss)

        ra = np.squeeze(show_x[nomal, :, :, 0]).reshape((72))
        tprt = np.squeeze(show_x[nomal, :, :, 1]).reshape((72))
        pow_ = show_y[nomal].reshape((72))

        if len(list(anomaly_mask)) != 0:
            anomaly_mask_ = anomaly_mask[nomal].reshape((72))
            draw_anomaly_withMask(
                ra, tprt, pow_, predict_, 0, 2*i, 'pow', 'ra', anomaly_mask_)
        else:
            draw_anomal(ra, tprt, pow_, predict_, 0, 2*i, 'pow', 'ra')
