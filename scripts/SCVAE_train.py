# -*- coding: utf-8 -*-

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
from read_data import read_single_data_noScale

from scipy.stats import t as studentT
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
import torch
import os
import sys
import time
import datetime
import argparse
from scipy import stats

import sys
sys.path.insert(0, sys.path[0]+"/../")
from model.SCVAE_model import SCVAE

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test_one_epoch(dataloader, model, reg, mode):
    num_batches = len(dataloader)
    # model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            model(data, y_data)
            if mode == 0:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss  # mode0
            elif mode == 1:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                    model.nll_loss_prior + 0 * model.smooth_loss_prior  # mode1
            elif mode == 2:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                    model.kld_loss_predict + model.nll_loss_predict
            test_loss += loss.item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss


def predict_one_epoch(dataloader, model, reg):
    num_batches = len(dataloader)
    # model.eval()
    predict_loss = 0
    reconstruct_loss = 0
    prior_loss = 0
    with torch.no_grad():
        # print(model.prior_flag)
        for batch_idx, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            model(data, y_data)
            loss_predict = model.nll_loss_predict
            loss_reconstruct = model.nll_loss
            loss_prior = model.nll_loss_prior

            predict_loss += loss_predict.item()
            reconstruct_loss += loss_reconstruct.item()
            prior_loss += loss_prior.item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        predict_loss /= num_batches*batch_size
        reconstruct_loss /= num_batches*batch_size
        prior_loss /= num_batches*batch_size

    #correct /= size
    print(
        f"predict Error: \n , Avg loss: {predict_loss:>8f} \n reconstruct Error: \n , Avg loss: {reconstruct_loss:>8f} \n prior Error: \n , Avg loss: {prior_loss:>8f} \n")


def train_one_epoch(dataloader, model, optimizer, reg, mode):
    train_loss = 0
    for batch_idx, (data, y_data) in enumerate(dataloader):
        # to (T,batch_size,feature_dim,input_dim)
        data = data.permute(1, 0, 3, 2).to(device)
        y_data = y_data.permute(1, 0, 3, 2).to(device)
        optimizer.zero_grad()
        model(data, y_data)
        if mode == 0:
            loss = model.kld_loss + model.nll_loss + \
                reg * model.smooth_loss  # mode0 reconstruct
        elif mode == 1:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                model.nll_loss_prior + 0 * model.smooth_loss_prior  # mode1  prior regularizer
        if mode == 2:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                model.kld_loss_predict + model.nll_loss_predict  # mode2  predict regularizer
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            size = len(data)
            loss, current = loss.item(), batch_idx * size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train(train_loader, test_loader, model,  optimizer, reg, mode,model_path):
    epochs = 40000
    best_test_loss = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train()
        train_one_epoch(train_loader, model, optimizer, reg, mode)
        model.eval()
        test_loss = test_one_epoch(test_loader, model, reg, mode)
        print("test predict")
        predict_one_epoch(test_loader, model, reg)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_path)
            print("Saved PyTorch Model State to model.pth")
    print("Done!")


def reconstruct(model, dataloader):
    print("*"*20+"reconstruct results")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            mu, std, score = model.reconstruct(data, y_data, is_prior=False)
            mu = np.transpose(mu, (1, 0, 2))  # transpose to （batch_size,T,M）
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
                        default=1e-5, help='learning_rate')
    parser.add_argument("--print_every", type=int, default=1,
                        help='the number of iterations between printing the results')
    parser.add_argument("--n_epochs", type=int, default=100000,
                        help='Maximum number of iterations')
    parser.add_argument("--h_dim", type=int, default=512,
                        help='Neural network hidden layer dimension')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='hidden variable dimension of VAEs')
    parser.add_argument("--seq_shape", type=tuple, default=(-1, 12, 6),
                        help="the sequence shape, e.g. (-1,12,6)")
    parser.add_argument("--mode", type=int, default=2,
                        help="the mode when train")
    parser.add_argument("--test_ratio", type=float,
                        default=0.25, help="the test ratio in data_set")
    parser.add_argument("--normal_ratio", type=int, default=99.5,
                        help="The percentage of normal samples assumed in the data set")
    opt = parser.parse_args()

    seq_shape = opt.seq_shape
    input_dim = seq_shape[-1]
    seq_len = seq_shape[-2]
    mode = opt.mode
    model_path=opt.model_path

    device = torch.device(opt.device)
    anomaly_mask = np.array([])

    ra_series_scale, pow_series_scale, tprt_series_scale = read_single_data_noScale(
        "enhance_12times_smooth_intense_normal_data_of_20trainsites_clean.csv", False)

    print(ra_series_scale.shape)
    ra_per_day_seq = ra_series_scale.reshape(seq_shape)
    pow_per_day_seq = pow_series_scale.reshape(seq_shape)
    tprt_per_day_seq = tprt_series_scale.reshape(seq_shape)

    print(pow_per_day_seq.shape)

    if len(list(anomaly_mask)) != 0:
        print("anomaly_mask not empty")
        anomaly_mask = anomaly_mask.reshape(-1, seq_len*input_dim)

    if opt.test_ratio != 1:
        x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(
            ra_per_day_seq, tprt_per_day_seq, pow_per_day_seq, test_size=opt.test_ratio, random_state=42)
        if len(list(anomaly_mask)) != 0:
            anomaly_mask_train, anomaly_mask_test = train_test_split(
                anomaly_mask, test_size=0.25, random_state=42)
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

    chunk_torch = torch.FloatTensor(multi_x_train)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    reg = opt.reg

    # Train
    train(train_loader_ordered, test_loader_ordered, model, optimizer, reg, mode, model_path)
