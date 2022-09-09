# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import argparse
from scipy import stats

import torch
from torch import nn, optim
from torch.distributions import Normal,kl_divergence,StudentT
from torch.autograd import Variable
import numpy as np 
import matplotlib
#matplotlib.use('agg') #非交互模式，可以减小内存占用
import matplotlib.pyplot as plt
#device = torch.device("cpu")
from scipy.stats import t as studentT
#device = torch.device("cuda")
from SCVAE_model2 import  Normal_VRNN
from read_data import read_whole_data,read_single_data,read_simulate_data,read_anomaly_data,read_multi_anomaly_data,read_whole_enhance_data_1,read_single_anomaly_case,read_multi_site_data,read_multi_site_data_with_simulate

from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #计算均方误差
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def mean_relative_error(y_true, y_pred):
    #print("np.max(y_true)",np.max(y_true,axis=0).shape)
    relative_error = np.average(np.abs(y_true - y_pred), axis=0)
    #print(relative_error.shape)
    #print(relative_error[:5],(np.max(y_true,axis=0)-np.min(y_true,axis=0))[:5])
    relative_error=relative_error/(np.max(y_true,axis=0)-np.min(y_true,axis=0))
    #print(relative_error[:5])
    return relative_error

def test_one_epoch(dataloader, model, reg, mode):
    num_batches = len(dataloader)
    #model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data,y_data) in enumerate(dataloader):
            data = data.permute(1,0,3,2).to(device)
            y_data = y_data.permute(1,0,3,2).to(device)
            model(data,y_data)
            if mode==1:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + model.nll_loss_prior + 0 * model.smooth_loss_prior #mode1
            elif mode==0:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss  #mode0
            elif mode==2:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss+ model.kld_loss_predict+ model.nll_loss_predict
            test_loss += loss.item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    
def predict_one_epoch(dataloader, model, reg):
    num_batches = len(dataloader)
    #model.eval()
    predict_loss = 0
    reconstruct_loss = 0
    with torch.no_grad():
        #print(model.prior_flag)
        for batch_idx, (data,y_data) in enumerate(dataloader):
            data = data.permute(1,0,3,2).to(device)
            y_data = y_data.permute(1,0,3,2).to(device)
            model(data,y_data)
            loss_predict = model.nll_loss_predict
            loss_reconstruct=model.nll_loss
            predict_loss += loss_predict.item()
            reconstruct_loss += loss_reconstruct.item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        predict_loss /= num_batches
        reconstruct_loss /= num_batches
    #correct /= size
    print(f"predict Error: \n , Avg loss: {predict_loss:>8f} \n reconstruct Error: \n , Avg loss: {reconstruct_loss:>8f} \n")
    
#self.kld_loss,self.nll_loss,self.smooth_loss,self.kld_loss_predict,self.nll_loss_prior,self.nll_loss_predict,self.smooth_loss_prior
def train_one_epoch(dataloader, model,optimizer,reg,mode):
    train_loss = 0
    for batch_idx, (data,y_data) in enumerate(dataloader):
        data = data.permute(1,0,3,2).to(device)  #转置成T,batch_size,feature_dim,input_dim
        #print("np.min(y_data)",y_data.shape,np.min(y_data.detach().cpu().numpy()))
        y_data = y_data.permute(1,0,3,2).to(device)
        optimizer.zero_grad()
        model(data,y_data)
        if mode==1:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + model.nll_loss_prior + 0 * model.smooth_loss_prior #mode1  prior
        if mode==0:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss  #mode0 reconstruct
        if mode==2:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss+ model.kld_loss_predict+ model.nll_loss_predict    #mode2 predict
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            size= len(data)
            loss, current = loss.item(), batch_idx * size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def train(train_loader, test_loader, model,  optimizer,reg,mode):
    epochs = 20000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train()  #设置训练和推理模式，防止两种模式之间弄混
        train_one_epoch(train_loader,model ,optimizer,reg,mode)
        model.eval()
        test_one_epoch(test_loader,model,reg,mode)
        #print("train predict")
        #predict_one_epoch(train_loader,model,reg)
        print("test predict")
        predict_one_epoch(test_loader,model,reg)
        if (t+1)%500==0:
            torch.save(model.state_dict(),"saved_model/2mode1_revise_CVAE_Model_SISVAE_guangfu_z128_h512_epoch%d"%(t))
            print("Saved PyTorch Model State to model.pth")
    print("Done!")
    
def reconstruct(model,dataloader,is_prior=False):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data,y_data) in enumerate(dataloader):
            data = data.permute(1,0,3,2).to(device) #转置成T,batch_size,feature_dim,input_dim
            y_data=y_data.permute(1,0,3,2).to(device)
            mu, std, score = model.reconstruct(data,y_data,is_prior=is_prior)
            #print(mu.shape) #(1,72,3)
            
            mu=np.transpose(mu,(1,0,2))  #变为 batch_size,T,M
            #print(mu)
            std=np.transpose(std,(1,0,2))
            score=np.transpose(score,(1,0,2))
            
            mus.append(mu)
            stds.append(std)
            scores.append(score)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        scores = np.concatenate(scores, axis=0)
        print("scores.shape",scores.shape)

        return mus,stds,scores
    
def predict_withLabel(model,dataloader):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data,y_data) in enumerate(dataloader):
            data = data.permute(1,0,3,2).to(device) #转置成T,batch_size,feature_dim,input_dim
            y_data=y_data.permute(1,0,3,2).to(device)
            mu, std, score = model.predict_withLabel(data,y_data)
            #print(mu.shape) #(1,72,3)
            
            mu=np.transpose(mu,(1,0,2))  #变为 batch_size,T,M
            #print(mu)
            std=np.transpose(std,(1,0,2))
            score=np.transpose(score,(1,0,2))
            
            mus.append(mu)
            stds.append(std)
            scores.append(score)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        scores = np.concatenate(scores, axis=0)
        print("scores.shape",scores.shape)

        return mus,stds,scores
    
def predict(model,dataloader):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for batch_idx, (data,y_data) in enumerate(dataloader):
            data = data.permute(1,0,3,2).to(device) #转置成T,batch_size,feature_dim,input_dim
            y_data=y_data.permute(1,0,3,2).to(device)
            mu, std = model.predict(data,y_data,is_prior=False)
            #print(mu.shape) #(1,72,3)
            
            mu=np.transpose(mu,(1,0,2))  #变为 batch_size,T,M
            #print(mu)
            std=np.transpose(std,(1,0,2))

            mus.append(mu)
            stds.append(std)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        print("scores.shape",scores.shape)

        return mus,stds

def get_hidden_z(model,dataloader,case=""):
    print("*"*20+"get hidden variable z")
    model.eval()
    with torch.no_grad():
        re_zs = []
        for batch_idx, (data,y_data) in enumerate(dataloader):
            data = data.permute(1,0,3,2).to(device) #转置成T,batch_size,feature_dim,input_dim
            y_data=y_data.permute(1,0,3,2).to(device)
            model(data,y_data)
            #print(mu.shape) #(1,72,3)
            re_z=np.array([item.cpu().numpy() for item in model.Z_mean])
            re_z=np.transpose(re_z,(1,0,2))
            print(re_z.shape)
            re_zs.append(re_z)
        
        re_zs = np.concatenate(re_zs, axis=0)

        print("re_zs.shape",re_zs.shape)
        np.save("re_z/re_zs_"+case,re_zs)
     
        return re_zs




#绘制异常的功率曲线          
def draw_anomal(ra,tprt,pow_,predict,label,index,y_label1,y_label2,savefig=None):
    fig = plt.figure(index,figsize=(20, 8))
    
    ax =fig.add_subplot(1,3,1) 
    ax2 = fig.add_subplot(1,3,2)
    ax3=fig.add_subplot(1,3,1)
    ax4=fig.add_subplot(1,3,3) #用来绘制温度图
    

    x=np.arange(pow_.shape[0])
    
    ax.scatter(x,pow_,s=10,c='b',alpha=1) #绘制散点图
    color= 'b' if label==0 else 'r'
    ax.plot(x, pow_, c=color,lw=1.5)  #绘制折线图

    ax2.scatter(x,ra,s=10,c='b',alpha=1) #绘制散点图
    color= 'b'
    ax2.plot(x, ra, c=color,lw=1.5)  #绘制折线图
    
    ax3.scatter(x,predict,s=10,c='b',alpha=1) #绘制散点图
    color= 'g'
    ax3.plot(x, predict, c=color,lw=1.5)  #绘制折线图
    
    ax4.scatter(x,tprt,s=10,c='b',alpha=1) #绘制散点图
    color= 'b'
    ax4.plot(x, tprt, c=color,lw=1.5)  #绘制折线图

    #for i in range(len(label)):
        #if label[i]==1:
            #ax.scatter(x[i], series[i], color='', marker='o', edgecolors='r', s=200)
    ax.set_xlabel('time')
    ax.set_ylabel(y_label1)
    title="data per 15 minute"
    ax.set_title(title)
    
    ax2.set_xlabel('time')
    ax2.set_ylabel(y_label2)
    title="data per 15 minute"
    ax2.set_title(title)
    plt.tight_layout()
    plt.show()
    if savefig is not None:
        fig.savefig(savefig)
        
def draw_anomaly_withMask(ra,tprt,pow_,predict,label,index,y_label1,y_label2,anomaly_mask,savefig=None):
    fig = plt.figure(index,figsize=(20, 8))
    
    ax =fig.add_subplot(1,3,1) 
    ax2 = fig.add_subplot(1,3,2)
    ax3=fig.add_subplot(1,3,1)
    ax4=fig.add_subplot(1,3,3) #用来绘制温度图
    

    x=np.arange(pow_.shape[0])
    
    ax.scatter(x,pow_,s=10,c='b',alpha=1) #绘制散点图
    color= 'b' if label==0 else 'r'
    ax.plot(x, pow_, c=color,lw=1.5)  #绘制折线图

    ax2.scatter(x,ra,s=10,c='b',alpha=1) #绘制散点图
    color= 'b'
    ax2.plot(x, ra, c=color,lw=1.5)  #绘制折线图
    
    ax3.scatter(x,predict,s=10,c='b',alpha=1) #绘制散点图
    color= 'g'
    ax3.plot(x, predict, c=color,lw=1.5)  #绘制折线图
    
    ax4.scatter(x,tprt,s=10,c='b',alpha=1) #绘制散点图
    color= 'b'
    ax4.plot(x, tprt, c=color,lw=1.5)  #绘制折线图

    for i in range(len(anomaly_mask)):
        if anomaly_mask[i]>0:
            ax.scatter(x[i], pow_[i], color='', marker='o', edgecolors='r', s=150)
    #for i in range(len(label)):
        #if label[i]==1:
            #ax.scatter(x[i], series[i], color='', marker='o', edgecolors='r', s=200)
    ax.set_xlabel('time')
    ax.set_ylabel(y_label1)
    title="data per 15 minute"
    ax.set_title(title)
    
    ax2.set_xlabel('time')
    ax2.set_ylabel(y_label2)
    title="data per 15 minute"
    ax2.set_title(title)
    plt.tight_layout()
    plt.show()
    if savefig is not None:
        fig.savefig(savefig)
        
#绘制具有真实信息的曲线
def draw_anomaly_infor(ra,tprt,pow_,predict,label,index,y_label1="power/cap",y_label2="radiation",y_label3="temperature",datetime=None,sitenum=None,city=None,savefig=None):
    
    site_infor=pd.read_csv("../../data/extra_data/site_list.csv")
    fig = plt.figure(index,figsize=(20, 8),clear=True) #新的fig会覆盖旧的
    
    ax =fig.add_subplot(1,3,1) 
    ax2 = fig.add_subplot(1,3,2)
    ax3=fig.add_subplot(1,3,1)
    ax4=fig.add_subplot(1,3,3) #用来绘制温度图
    

    x=np.arange(pow_.shape[0])
    
    ax.scatter(x,pow_,s=10,c='b',alpha=1) #绘制散点图
    color= 'b' if label==0 else 'r'
    tag="normal realistic pow" if label==0 else 'anomaly realistic pow'
    ax.plot(x, pow_, c=color,lw=1.5,label=tag)  #绘制折线图
    ax.legend(loc="best")

    ax2.scatter(x,ra,s=10,c='b',alpha=1) #绘制散点图
    color= 'b'
    ax2.plot(x, ra, c=color,lw=1.5,label="radiation")  #绘制折线图
    
    ax3.scatter(x,predict,s=10,c='b',alpha=1) #绘制散点图
    color= 'g'
    ax3.plot(x, predict, c=color,lw=1.5,label="predict pow")  #绘制折线图
    ax3.legend(loc="best")
    
    ax4.scatter(x,tprt,s=10,c='b',alpha=1) #绘制散点图
    color= 'b'
    ax4.plot(x, tprt, c=color,lw=1.5,label="temperature")  #绘制折线图
    
    if datetime!=None:
        ax4.annotate(datetime,(50,(np.max(tprt)/2+np.min(tprt)/2)),fontsize=14,horizontalalignment='center')
    #assert sitenum!=None
    #print(type(sitenum))
    if sitenum!=None:
        site_longitude=site_infor[site_infor["proj_id"].astype('string')==sitenum]["longitude"].values[0]
        site_lattitude=site_infor[site_infor["proj_id"].astype('string')==sitenum]["lattitude"].values[0]
        #print(site_name)
        
        ax4.annotate("site_num: "+str(sitenum),(50,(2*np.max(tprt)/3+np.min(tprt)/3)),fontsize=14,horizontalalignment='center')
        ax4.annotate("longitude: "+str(round(site_longitude,4))+" lattitude: "+str(round(site_lattitude,4)),(50,(np.max(tprt)/3+2*np.min(tprt)/3)),fontsize=14,horizontalalignment='center')
        #ax4.annotate("lattitude: "+str(site_lattitude),(50,(np.max(tprt)/4+3*np.min(tprt)/4)),fontsize=14,horizontalalignment='center')
        ax4.annotate("city: "+str(city),(50,(np.max(tprt)/5+4*np.min(tprt)/5)),fontsize=14,horizontalalignment='center')

    #for i in range(len(label)):
        #if label[i]==1:
            #ax.scatter(x[i], series[i], color='', marker='o', edgecolors='r', s=200)
    ax.set_xlabel('time')
    ax.set_ylabel(y_label1)
    title="data per 15 minute"
    ax.set_title(title)
    
    ax2.set_xlabel('time')
    ax2.set_ylabel(y_label2)
    title="data per 15 minute"
    ax2.set_title(title)
    ax4.set_title(title)
    ax4.set_xlabel('time')
    ax4.set_ylabel(y_label3)
    plt.tight_layout()
    plt.show()
    if savefig is not None:
        fig.savefig("D:/Guangfu_Test_site/"+sitenum+"_"+savefig)
    plt.clf() #每次清空内存
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='data/site_30100005_series.npz', help='data file path')
    parser.add_argument("--reg", type=float, default=0, help='Smooth canonical intensity')
    parser.add_argument("--batch_size", type=int, default = 256, help = "batch size")
    parser.add_argument("--device", type=str, default='cuda:0', help="device, e.g. cpu, cuda:0, cuda:1")
    parser.add_argument("--learning_rate", type=float, default=0.000001, help='Adam optimizer learning rate')
    parser.add_argument("--print_every", type=int, default = 1, help='The number of iterations in which the result is printed')
    parser.add_argument("--n_epochs", type=int, default = 10000, help='maximum number of iterations')
    parser.add_argument("--h_dim", type=int, default = 512, help='dimension in Neural network hidden layers ')
    parser.add_argument('--z_dim', type=int, default = 128, help = 'dimensions of latent variable')
    parser.add_argument("--test_ratio", type=float, default=1, help="the test ratio in data_set")
    parser.add_argument("--normal_ratio", type=int, default=99.9, help="the nomal_ratio % in dataset")
    parser.add_argument("--mode", type=int, default=2, help="the mode when train")
    parser.add_argument("--is_predict", type=bool, default=False, help="whether predict ot not")
    parser.add_argument("--is_prior", type=bool, default=False, help="whether prior reconstruct ot not")
    parser.add_argument("--whether_get_z", type=bool, default=False, help="whether get hidden variable")
    parser.add_argument("--is_scale_data", type=bool, default=False, help="whether get scale data")
    parser.add_argument("--is_simulate_data", type=bool, default=True, help="whether use simulate data or predict data")
    opt = parser.parse_args()
    
    
    whether_get_z=opt.whether_get_z
    is_predict=opt.is_predict
    is_prior=opt.is_prior
    is_simulate=opt.is_simulate_data
    mode=opt.mode
    is_scale=opt.is_scale_data
    seq_shape=(-1,12,6)
    input_dim=seq_shape[-1]
    seq_len=seq_shape[-2]

    device = torch.device(opt.device)
    print(device)
    save_dir = 'guangfu'
    os.makedirs(save_dir, exist_ok=True)
    anomaly_mask=np.array([])
    date_series=np.array([])
    site_num_series=np.array([])
    ra_series_noscale_simulate=np.array([])
    pow_series_noscale_simulate=np.array([])
    tprt_series_noscale_simulate=np.array([])
    date_series_simulate=np.array([])
    site_num_series_simulate=np.array([])
    
    #ra_series_scale,pow_series_scale,tprt_series_scale,ra_series_mean,pow_series_mean,tprt_series_mean,ra_series_std,pow_series_std,tprt_series_std,date_series,site_num_series,City_dict=read_multi_site_data(start=21,num=31,include_intenseNormal=False,is_scale=is_scale)
    #ra_series_scale,pow_series_scale,tprt_series_scale,ra_series_mean,pow_series_mean,tprt_series_mean,ra_series_std,pow_series_std,tprt_series_std,date_series,site_num_series,City_dict,\
    #ra_series_noscale_simulate,pow_series_noscale_simulate,tprt_series_noscale_simulate,date_series_simulate,site_num_series_simulate=read_multi_site_data_with_simulate(start=21,num=31,include_intenseNormal=False,is_scale=is_scale)
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask=read_single_anomaly_case("test_spike_anomaly_data_of_10sites.csv",test="True")
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask=read_single_anomaly_case("lowValue_anomaly_data_of_20sites.csv",include_normal=False) #"smooth_intense_normal_data_of_20sites.csv"
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask=read_single_anomaly_case("smooth_intense_normal_data_of_20sites.csv",include_normal=False)
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask=read_multi_anomaly_data(test=True)
    #ra_series_scale,pow_series_scale,tprt_series_scale=read_whole_data()
    #ra_series_scale,pow_series_scale,tprt_series_scale=read_single_data()
    #ra_series_scale,pow_series_scale,tprt_series_scale=read_simulate_data()
    ra_series_scale,pow_series_scale,tprt_series_scale=read_whole_enhance_data_1()
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask,tprt_mean,tprt_std=read_anomaly_data("site_30100005_simulate_72_swapAnomaly_labeled_small.csv")
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask,tprt_mean,tprt_std=read_anomaly_data("site_30100008_simulate_72_swapAnomaly_labeled_small.csv")
    
    print(ra_series_scale.shape)
    ra_per_day_seq=ra_series_scale.reshape(seq_shape)  #72个数据点
    pow_per_day_seq=pow_series_scale.reshape(seq_shape)   
    print(pow_per_day_seq[pow_per_day_seq<0])
    #pow_per_day_seq=-pow_per_day_seq #强制取负数看看模型的重构能力
    print("np.min(pow_per_day_seq)",np.min(pow_per_day_seq))
    tprt_per_day_seq=tprt_series_scale.reshape(seq_shape)
    
    #print("pow_series_mean",pow_series_mean)
    '''
    if not is_scale:
        ra_per_day_seq_mean=ra_series_noscale  #72个数据点
        pow_per_day_seq_mean=pow_series_noscale 
        tprt_per_day_seq_mean=tprt_series_noscale
    '''
    
    
    if len(list(date_series))!=0:
        print("data_series not empty")
        date_series=date_series.reshape(-1,seq_len*input_dim)
        print(date_series.shape)
        
    if len(list(site_num_series))!=0:
        print("site_num_series not empty")
        site_num_series=site_num_series.reshape(-1,seq_len*input_dim)
    
    if len(list(anomaly_mask))!=0:
        print("anomaly_mask not empty")
        anomaly_mask=anomaly_mask.reshape(-1,seq_len*input_dim)
        
    if len(list(date_series_simulate))!=0:
        print("data_series_simulate not empty")
        date_series_simulate=date_series_simulate.reshape(-1,seq_len*input_dim)
        print(date_series_simulate.shape)
        
    if len(list(site_num_series_simulate))!=0:
        print("site_num_series_simulate not empty")
        site_num_series_simulate=site_num_series_simulate.reshape(-1,seq_len*input_dim)
    
    if len(list(ra_series_noscale_simulate))!=0:
        print("ra_series_noscale_simulate not empty")
        ra_series_noscale_simulate=ra_series_noscale_simulate.reshape(-1,seq_len*input_dim)

    if len(list(pow_series_noscale_simulate))!=0:
        print("pow_series_noscale_simulate not empty")
        pow_series_noscale_simulate=pow_series_noscale_simulate.reshape(-1,seq_len*input_dim)
        
    if len(list(tprt_series_noscale_simulate))!=0:
        print("ra_series_noscale_simulate not empty")
        tprt_series_noscale_simulate=tprt_series_noscale_simulate.reshape(-1,seq_len*input_dim)
        

    print(ra_per_day_seq.shape)
    print(pow_per_day_seq.shape)
    print(tprt_per_day_seq.shape)

    #x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(ra_per_day_seq,tprt_per_day_seq,pow_per_day_seq,test_size=0.25, random_state=42) #由于已经shuffle过了后面的dataloader就无需shuffle了
    if opt.test_ratio!=1:
        x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(ra_per_day_seq,tprt_per_day_seq,pow_per_day_seq,test_size=opt.test_ratio, random_state=42) #random_state一样，获得的结果就是一样的
        if len(list(anomaly_mask))!=0:
            anomaly_mask_train,anomaly_mask_test = train_test_split(anomaly_mask,test_size=opt.test_ratio, random_state=42)
    else:
        x_train1=x_test1=ra_per_day_seq
        x_train2=x_test2=tprt_per_day_seq
        y_train=y_test=pow_per_day_seq
        if len(list(anomaly_mask))!=0:
            anomaly_mask_train=anomaly_mask_test=anomaly_mask

    print(x_train1.shape)
    print(y_train.shape)
    
    multi_x_train=np.stack([x_train1,x_train2],axis=3)
    multi_x_test=np.stack([x_test1,x_test2],axis=3)
    multi_y_train=np.stack([y_train],axis=3)
    print("np.min(multi_y_train)",np.min(multi_y_train))
    multi_y_test=np.stack([y_test],axis=3)
    
    print(multi_x_train.shape)
    print(multi_y_train.shape)
    
    
    #训练数据要将numpy数组转化为tensor

    train_chunks= multi_x_train
    test_chunks=multi_x_test

    print(f"data_shape：{train_chunks.shape}")
    
    x_dim = train_chunks.shape[-1]
    y_dim = multi_y_train.shape[-1]
    h_dim = opt.h_dim
    z_dim = opt.z_dim
    n_epochs = opt.n_epochs
    learning_rate = opt.learning_rate
    print_every = opt.print_every
    # model = StudentT_VRNN(x_dim, h_dim, z_dim,1,device=device).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    chunk_torch = torch.FloatTensor(train_chunks) #将numpy数组转化为tensor
    test_torch=torch.FloatTensor(test_chunks)
    batch_size = opt.batch_size
    import torch.utils.data

   
    #train_loader_ordered = torch.utils.data.DataLoader(chunk_torch, batch_size=batch_size, shuffle=False)
    #test_loader_ordered=torch.utils.data.DataLoader(test_torch, batch_size=batch_size, shuffle=False)

    train_loader_ordered = torch.utils.data.DataLoader(TensorDataset(chunk_torch,torch.FloatTensor(multi_y_train)), batch_size=batch_size, shuffle=False)
    test_loader_ordered= torch.utils.data.DataLoader(TensorDataset(test_torch,torch.FloatTensor(multi_y_test)), batch_size=batch_size, shuffle=False)

    #x_dim, label_dim, h_dim, z_dim, input_dim,bias=False, device=None, is_prior=False
    model = Normal_VRNN(x_dim, y_dim, h_dim, z_dim, input_dim, 1, device = device,is_prior=False).to(device)

    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    reg = opt.reg
    
    #train model
    train(train_loader_ordered, test_loader_ordered, model, optimizer,reg,mode=mode)
'''
# 模型输出

    #model_path="./saved_model/revise_CVAE_Model_SISVAE_guangfu_72_epoch399"
    #model_path="./saved_model/revise_CVAE_Model_SISVAE_guangfu_z64_epoch499"
    #model_path="./saved_model/revise_CVAE/2mode2_revise_CVAE_Model_SISVAE_guangfu_z128_h512_epoch10999"
    #model_path="./saved_model/revise_CVAE/3mode0_revise_CVAE_Model_SISVAE_guangfu_z128_h512_epoch5999"
    #model_path="./saved_model/revise_CVAE/4mode1_revise_CVAE_Model3_SISVAE_guangfu_z128_h512_epoch7999"           #mode2
    #model_path="./saved_model/revise_CVAE/5mode1_revise_CVAE_Model2_SISVAE_guangfu_z128_h512_epoch8999"           #best perform mode1
    #model_path="./saved_model/revise_CVAE/20site_mode1_revise_CVAE_Model2_SISVAE_guangfu_z128_h512_lr_10-6_reg0_epoch19999"
    model_path="./saved_model/revise_CVAE/20site_mode2_revise_CVAE_Model2_SISVAE_guangfu_z128_h512_lr_10-6_reg0_epoch30499"
    #model_path="./saved_model/revise_CVAE/20site_mode0_revise_CVAE_Model2_SISVAE_guangfu_z128_h512_lr_10-6_reg0_epoch7999"
    #model_path="./saved_model/revise_CVAE/20site_2mode1_revise_CVAE_Model2_SISVAE_guangfu_z128_h512_lr_10-6_reg0_epoch27999"
    model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    model.eval()
    #need to change 用测试集还是训练集来验证
    show_dataloader=test_loader_ordered
    show_x=multi_x_test
    show_y=y_test
    if len(list(anomaly_mask))!=0:
        anomaly_mask=anomaly_mask_test
        
        
    if whether_get_z==True:
        re_zs=get_hidden_z(model,show_dataloader,case="normal")
        
     
    if not is_predict:
        mus,stds,scores=reconstruct(model,show_dataloader,is_prior) #mode0或mode1
    else:
        mus,stds,scores=predict_withLabel(model,show_dataloader)
    #可能需要考虑anomaly_mask
    #if anomaly_mask!=None:
        #print(anomaly_mask.shape)
        #swap_list=[]
        #for i,item in enumerate(anomaly_mask):
            #if item.sum()!=0:
                #swap_list.append(i)
        #print("swap_list",swap_list,len(swap_list))

    
    print(mus.shape)
    print(show_y.shape)
    predict_=np.squeeze(mus).reshape(-1,input_dim*seq_len) #预测值
    pow_=np.squeeze(show_y).reshape(-1,input_dim*seq_len) #真实值
    
    
    #用来验证score的真实性
    
    #score_y=show_y.transpose(1,0,2)
    #score_chain = np.zeros(shape=(12,show_y.shape[0],6))
    #print(score_chain.shape)
    #score_mean=mus.transpose(1,0,2)
    #score_std=stds.transpose(1,0,2)
    #for t in range(score_y.shape[0]):
        #score_y_t=score_y[t]
        #score_chain[t,:,:]=-stats.norm.pdf(score_y_t,
                    #score_mean[t],
                    #score_std[t])
    #score_chain=score_chain.transpose(1,0,2)
    #print(score_chain.shape)
    #print(scores[0][0],score_chain[0][0])
    #assert (score_chain==scores).all()
    #scores=score_chain
    
    
    mse=mean_squared_error(pow_,predict_)
    print("mse",mse)
    
    detect_anomaly=0
  
    #NEED TO CHANGE 需要anomaly_mask时使用
    
    #print(series1.shape)


    nomal_ratio=opt.normal_ratio #要写99不能写0.99
    
    
    #计算recall precision时也要用mse作为异常分值
    predict_transpose=predict_.T
    pow_transpose=pow_.T
    print(predict_transpose.shape,pow_transpose.shape)
    
    #anomal_score=mean_squared_error(pow_transpose,predict_transpose,multioutput='raw_values')

    anomal_score=mean_relative_error(pow_transpose,predict_transpose)
    #anomal_score=np.array([1])
    
    #print(scores.shape)
    #anomal_score=scores[:,:,:].reshape((scores.shape[0],-1))
    #print("anomal_score.shape",anomal_score.shape)
    #anomal_score=np.sum(anomal_score,axis=-1) #将每天每个点的异常值相加  
    #anomal_score=np.mean(anomal_score,axis=-1) #将每天每个点的异常值相加  
    
    print("anomal_score.shape",anomal_score.shape)
    
    if len(list(anomaly_mask))!=0:
        print("anomal_mask.shape",anomaly_mask.shape)
        label=np.sum(anomaly_mask,axis=1)
        sum_num=np.sum(label)
        label[label>0]=1
        num=np.sum(label)
        print("average_swap",sum_num/num)
        print(label.shape)
        print(np.sum(label))
        print(anomal_score.shape)
    
        roc=roc_auc_score(label,anomal_score)
        prc=average_precision_score(label,anomal_score)
        print("roc:",roc)
        print("prc:",prc)

    
    #print(anomal_score)
    print("***************anomal_score*****************",np.mean(anomal_score))
    percentile_1=np.percentile(anomal_score,nomal_ratio) #假设有5%的点是异常点计算98分位点
    print("percentile_1",percentile_1)
    label=anomal_score.copy()
    label[label>percentile_1]=1
    label[label<=percentile_1]=0
    
    normal_pattern=0 #0为选取强正常,1为选取所有正常
    if normal_pattern==1:

        nomal_inst=[i for i,j in enumerate(label) if j == 0]
        print(nomal_inst)
        print(len(nomal_inst))
        for i,nomal in enumerate(nomal_inst):
            model.eval()
            with torch.no_grad():
                predict_=np.squeeze(mus[nomal]).reshape((72))
                loss = anomal_score[nomal]
                print(loss)
    
            ra=np.squeeze(show_x[nomal,:,:,0]).reshape((72))
            #pow_=series1[anomal].reshape((72))
            tprt=np.squeeze(show_x[nomal,:,:,1]).reshape((72))
            pow_=show_y[nomal].reshape((72))
            #tprt=tprt*tprt_std+tprt_mean
            #draw_anomal(ra,tprt,pow_,predict,0,2*i,'pow','ra')
            #数据中如果有anomal_mask的话 NEED TO CHANGE
            #anomaly_mask_=anomaly_mask[nomal].reshape((72))
            if len(list(anomaly_mask))!=0:
                anomaly_mask_=anomaly_mask[nomal].reshape((72))
                draw_anomaly_withMask(ra,tprt,pow_,predict_,0,2*i,'pow','ra',anomaly_mask_)
            elif len(list(date_series))!=0:
                date_=date_series[nomal][0]
                site_num_=site_num_series[nomal][0]
                
                ra_mean=ra_series_mean[site_num_]
                ra_std=ra_series_std[site_num_]
                pow_mean=pow_series_mean[site_num_]
                pow_std=pow_series_std[site_num_]
                tprt_mean=tprt_series_mean[site_num_]
                tprt_std=tprt_series_std[site_num_]
                
                ra=ra*ra_std+ra_mean
                pow_=pow_*pow_std+pow_mean
                tprt=tprt*tprt_std+tprt_mean
                predict_=predict_*pow_std+pow_mean
                
                draw_anomaly_infor(ra,tprt,pow_,predict_,0,2*i,datetime=date_,sitenum=site_num_,city=City_dict[site_num_])
            else:
                draw_anomal(ra,tprt,pow_,predict_,0,2*i,'pow','ra')
            #draw_anomal(ra,tprt,pow_,predict,0,2*i+1,'pow','ra')    
    

    anomal_inst=[i for i,j in enumerate(label) if j == 1]
    print(anomal_inst)
    print(len(anomal_inst))
    for i,anomal in enumerate(anomal_inst):
        model.eval()
        with torch.no_grad():
            
            predict_=np.squeeze(mus[anomal]).reshape((72))
            loss = anomal_score[anomal]
            print(loss)

        ra=np.squeeze(show_x[anomal,:,:,0]).reshape((72))
        #pow_=series1[anomal].reshape((72))
        tprt=np.squeeze(show_x[anomal,:,:,1]).reshape((72))
        pow_=show_y[anomal].reshape((72))
        #tprt=tprt*tprt_std+tprt_mean
        #数据中如果有anomal_mask的话 NEED TO CHANGE

        if len(list(anomaly_mask))!=0:
            anomaly_mask_=anomaly_mask[anomal].reshape((72))
            if anomaly_mask_.sum()!=0:
                detect_anomaly+=1
            draw_anomaly_withMask(ra,tprt,pow_,predict_,1,2*i,'pow','ra',anomaly_mask_)
        
        elif len(list(date_series))!=0:
            date_=date_series[anomal][0]
            site_num_=site_num_series[anomal][0]
            
            ra_mean=ra_series_mean[site_num_]
            ra_std=ra_series_std[site_num_]
            pow_mean=pow_series_mean[site_num_]
            pow_std=pow_series_std[site_num_]
            tprt_mean=tprt_series_mean[site_num_]
            tprt_std=tprt_series_std[site_num_]
            
            ra=ra*ra_std+ra_mean
            pow_=pow_*pow_std+pow_mean
            tprt=tprt*tprt_std+tprt_mean
            predict_=predict_*pow_std+pow_mean
            
            draw_anomaly_infor(ra,tprt,pow_,predict_,1,2*i,datetime=date_,sitenum=site_num_,city=City_dict[site_num_],savefig="anomal_index_"+str(anomal)+".png")
        else:
            draw_anomal(ra,tprt,pow_,predict_,1,2*i,'pow','ra')
    print(detect_anomaly)

    
    if normal_pattern==0:
        percentile_2=np.percentile(-anomal_score,nomal_ratio) #获取正常点
        print(percentile_2)
        label=-anomal_score.copy()
        label[label<=percentile_2]=0
        label[label>percentile_2]=1
        nomal_inst=[i for i,j in enumerate(label) if j == 1]
        print(nomal_inst)
        for i,nomal in enumerate(nomal_inst):
            model.eval()
            with torch.no_grad():
                predict_=np.squeeze(mus[nomal]).reshape((72))
                loss = anomal_score[nomal]
                print(loss)
    
            ra=np.squeeze(show_x[nomal,:,:,0]).reshape((72))
            #pow_=series1[anomal].reshape((72))
            tprt=np.squeeze(show_x[nomal,:,:,1]).reshape((72))
            pow_=show_y[nomal].reshape((72))
            #tprt=tprt*tprt_std+tprt_mean
            #draw_anomal(ra,tprt,pow_,predict,0,2*i,'pow','ra')
            #数据中如果有anomal_mask的话 NEED TO CHANGE
            #anomaly_mask_=anomaly_mask[nomal].reshape((72))
            if len(list(anomaly_mask))!=0:
                anomaly_mask_=anomaly_mask[nomal].reshape((72))
                draw_anomaly_withMask(ra,tprt,pow_,predict_,0,2*i,'pow','ra',anomaly_mask_)
            elif len(list(date_series))!=0:
                date_=date_series[nomal][0]
                site_num_=site_num_series[nomal][0]
                
                ra_mean=ra_series_mean[site_num_]
                ra_std=ra_series_std[site_num_]
                pow_mean=pow_series_mean[site_num_]
                pow_std=pow_series_std[site_num_]
                tprt_mean=tprt_series_mean[site_num_]
                tprt_std=tprt_series_std[site_num_]
                
                ra=ra*ra_std+ra_mean
                pow_=pow_*pow_std+pow_mean
                tprt=tprt*tprt_std+tprt_mean
                predict_=predict_*pow_std+pow_mean
                
                draw_anomaly_infor(ra,tprt,pow_,predict_,0,2*i,datetime=date_,sitenum=site_num_,city=City_dict[site_num_])
            else:
                draw_anomal(ra,tprt,pow_,predict_,0,2*i,'pow','ra')
            #draw_anomal(ra,tprt,pow_,predict,0,2*i+1,'pow','ra')
'''
    
