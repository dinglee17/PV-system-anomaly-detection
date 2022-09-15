# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:56:38 2021

@author: 李鼎
SIS的数据读取文件
"""

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import os

def read_whole_data():
    #先读取真实数据
    root="../../data/"
    data_path=root+"site_30100005_72_nonzero.csv"     
    series_data=pd.read_csv(data_path,sep=',')
    ra_series=series_data["instantaneous_global_radiation"]
    pow_series=series_data["instantaneous_output_power"]
    tprt_series=series_data["clearsky_Tamb"]

    enhance_length=5
    simulate_length=10
    for i in range(1,enhance_length+1):
        series_data=pd.read_csv(root+f"site_30100005_nonzero_72_dataEnhance_50_{i}.csv",sep=',')
        ra_series=ra_series.append(series_data["instantaneous_global_radiation"],ignore_index=True)
        pow_series=pow_series.append(series_data["instantaneous_output_power"],ignore_index=True)
        tprt_series=tprt_series.append(series_data["clearsky_Tamb"],ignore_index=True)
    
    ra_series=ra_series.values
    pow_series=pow_series.values
    tprt_series=tprt_series.values
        
    ra_series_scale = preprocessing.scale(ra_series.reshape((-1,1))).reshape((-1)) #正太分布scacle
    pow_series_scale = preprocessing.scale(pow_series.reshape((-1,1))).reshape((-1))
    tprt_series_scale=preprocessing.scale(tprt_series.reshape((-1,1))).reshape((-1))
    
    #再读取模拟数据
    series_data=pd.read_csv(root+"site_30100005_simulate_1.csv",sep=',')
    ra_series=series_data["instantaneous_global_radiation"]
    pow_series=series_data["modeled_power"]
    tprt_series=series_data["clearsky_Tamb"]
    
    
    for i in range(1,simulate_length+1):
        series_data=pd.read_csv(root+f"site_30100005_simulate_72_dataEnhance_50_{i}.csv",sep=',')
        ra_series=ra_series.append(series_data["instantaneous_global_radiation"],ignore_index=True)
        pow_series=pow_series.append(series_data["modeled_power"],ignore_index=True)
        tprt_series=tprt_series.append(series_data["clearsky_Tamb"],ignore_index=True)

    ra_series=ra_series.values
    pow_series=pow_series.values
    tprt_series=tprt_series.values
    
    ra_series_scale_simulate = preprocessing.scale(ra_series.reshape((-1,1))).reshape((-1)) #正太分布scacle
    pow_series_scale_simulate = preprocessing.scale(pow_series.reshape((-1,1))).reshape((-1))
    tprt_series_scale_simulate = preprocessing.scale(tprt_series.reshape((-1,1))).reshape((-1))
    
    ra_series_scale=np.concatenate([ra_series_scale,ra_series_scale_simulate])
    pow_series_scale=np.concatenate([pow_series_scale,pow_series_scale_simulate])
    tprt_series_scale=np.concatenate([tprt_series_scale,tprt_series_scale_simulate])
    print(ra_series_scale.shape)
    print(pow_series_scale.shape)
    print(tprt_series_scale.shape)

    return ra_series_scale,pow_series_scale,tprt_series_scale

def read_single_data_noScale(filename="site_30100005_1.csv",label=False):
    root="./data/"
    data_path=root+filename
    series_data=pd.read_csv(data_path,sep=',')
    ra_series=series_data["instantaneous_global_radiation"].values
    pow_series=series_data["instantaneous_output_power"].values
    tprt_series=series_data["clearsky_Tamb"].values
    if label==True:
        anomaly_mask=series_data["label"].values
        return ra_series,pow_series,tprt_series,anomaly_mask
    
    return ra_series,pow_series,tprt_series

def read_single_anomaly_case(filename="spike_anomaly_data_of_20sites.csv",test=False,include_normal=True):

    ra_series_1,pow_series_1,tprt_series_1,anomaly_mask1=read_single_data_noScale(filename,True)
    
    if include_normal==True:
        
        if test==False:
            ra_series_0,pow_series_0,tprt_series_0,anomaly_mask0=read_single_data_noScale("smooth_intense_normal_data_of_20sites.csv",True)
        else:
            ra_series_0,pow_series_0,tprt_series_0,anomaly_mask0=read_single_data_noScale("2test_smooth_intense_normal_data_of_10sites.csv",True)
        
        ra_series_whole=np.concatenate([ra_series_0[:],ra_series_1])
        pow_series_whole=np.concatenate([pow_series_0[:],pow_series_1])
        tprt_series_whole=np.concatenate([tprt_series_0[:],tprt_series_1])
        anomaly_mask_whole=np.concatenate([anomaly_mask0[:],anomaly_mask1])
        
    else:
        ra_series_whole=ra_series_1
        pow_series_whole=pow_series_1
        tprt_series_whole=tprt_series_1
        anomaly_mask_whole=anomaly_mask1
    
    return ra_series_whole,pow_series_whole,tprt_series_whole,anomaly_mask_whole

def read_multi_anomaly_data(test=False):
    
    if test==False:
        ra_series_0,pow_series_0,tprt_series_0,anomaly_mask0=read_single_data_noScale("2smooth_intense_normal_data_of_20sites.csv",True)
        ra_series_1,pow_series_1,tprt_series_1,anomaly_mask1=read_single_data_noScale("2spike_anomaly_data_of_20sites.csv",True)
        ra_series_2,pow_series_2,tprt_series_2,anomaly_mask2=read_single_data_noScale("2inverter_fault_anomaly_data_of_20sites.csv",True)
        ra_series_3,pow_series_3,tprt_series_3,anomaly_mask3=read_single_data_noScale("2snowy_anomaly_data_of_20sites.csv",True)
        ra_series_4,pow_series_4,tprt_series_4,anomaly_mask4=read_single_data_noScale("2cloudy_anomaly_data_of_20sites.csv",True)
        ra_series_5,pow_series_5,tprt_series_5,anomaly_mask5=read_single_data_noScale("2lowValue_anomaly_data_of_20sites.csv",True)
        ra_series_6,pow_series_6,tprt_series_6,anomaly_mask6=read_single_data_noScale("2shading_anomaly_data_of_20sites.csv",True)
    else:
        ra_series_0,pow_series_0,tprt_series_0,anomaly_mask0=read_single_data_noScale("3test_smooth_intense_normal_data_of_10sites.csv",True)
        ra_series_1,pow_series_1,tprt_series_1,anomaly_mask1=read_single_data_noScale("3test_spike_anomaly_data_of_10sites.csv",True)
        ra_series_2,pow_series_2,tprt_series_2,anomaly_mask2=read_single_data_noScale("3test_inverter_fault_anomaly_data_of_10sites.csv",True)
        ra_series_3,pow_series_3,tprt_series_3,anomaly_mask3=read_single_data_noScale("3test_snowy_anomaly_data_of_10sites.csv",True)
        ra_series_4,pow_series_4,tprt_series_4,anomaly_mask4=read_single_data_noScale("3test_cloudy_anomaly_data_of_10sites.csv",True)
        ra_series_5,pow_series_5,tprt_series_5,anomaly_mask5=read_single_data_noScale("3test_lowValue_anomaly_data_of_10sites.csv",True)
        ra_series_6,pow_series_6,tprt_series_6,anomaly_mask6=read_single_data_noScale("3test_shading_anomaly_data_of_10sites.csv",True)
    
    ra_series_whole=np.concatenate([ra_series_0,ra_series_1,ra_series_2,ra_series_3,ra_series_4,ra_series_5,ra_series_6])
    pow_series_whole=np.concatenate([pow_series_0,pow_series_1,pow_series_2,pow_series_3,pow_series_4,pow_series_5,pow_series_6])
    tprt_series_whole=np.concatenate([tprt_series_0,tprt_series_1,tprt_series_2,tprt_series_3,tprt_series_4,tprt_series_5,tprt_series_6])
    anomaly_mask_whole=np.concatenate([anomaly_mask0,anomaly_mask1,anomaly_mask2,anomaly_mask3,anomaly_mask4,anomaly_mask5,anomaly_mask6])
    
    return ra_series_whole,pow_series_whole,tprt_series_whole,anomaly_mask_whole

def read_single_data(filename="site_30100005_1.csv"):
    root="./data/"
    data_path=root+filename
    series_data=pd.read_csv(data_path,sep=',')
    ra_series=series_data["instantaneous_global_radiation"].values
    pow_series=series_data["instantaneous_output_power"].values
    tprt_series=series_data["clearsky_Tamb"].values
    
    ra_series_scale = preprocessing.scale(ra_series.reshape((-1,1))).reshape((-1)) #正太分布scacle
    pow_series_scale = preprocessing.scale(pow_series.reshape((-1,1))).reshape((-1))
    tprt_series_scale=preprocessing.scale(tprt_series.reshape((-1,1))).reshape((-1))
    return ra_series_scale,pow_series_scale,tprt_series_scale

def read_simulate_data(filename="site_30100005_simulate_1.csv"):
    root="../../data/"
    data_path=root+filename     #真实数据集 
    series_data=pd.read_csv(data_path,sep=',')
    ra_series=series_data["instantaneous_global_radiation"].values
    pow_series=series_data["modeled_power"].values
    tprt_series=series_data["clearsky_Tamb"].values
    
    ra_series_scale = preprocessing.scale(ra_series.reshape((-1,1))).reshape((-1)) #正太分布scacle
    pow_series_scale = preprocessing.scale(pow_series.reshape((-1,1))).reshape((-1))
    tprt_series_scale=preprocessing.scale(tprt_series.reshape((-1,1))).reshape((-1))
    return ra_series_scale,pow_series_scale,tprt_series_scale

def read_anomaly_data(filename="site_30100005_simulate_72_swapAnomaly_20.csv"):
    root="../../data/"
    data_path=root+filename #    污染数据集  
    series_data=pd.read_csv(data_path,sep=',')
    ra_series=series_data["instantaneous_global_radiation"].values
    pow_series=series_data["modeled_power"].values
    tprt_series=series_data["clearsky_Tamb"].values
    
    true_data_path=root+"site_30100005_simulate_1.csv"
    true_series_data=pd.read_csv(true_data_path,sep=',')
    true_pow_series=true_series_data["modeled_power"].values
    scaler = preprocessing.StandardScaler().fit(true_pow_series.reshape((-1,1))) #用标准均值和方差转化
    anomaly_mask=series_data["swap_mask"].values

    ra_series_scale = preprocessing.scale(ra_series.reshape((-1,1))).reshape((-1)) 
    #pow_series_scale = preprocessing.scale(pow_series.reshape((-1,1)))
    pow_series_scale=scaler.transform(pow_series.reshape((-1,1))).reshape((-1))   #用标准均值方差 mask过的 scale 否则会出现不匹配的问题
    tprt_series_scale=preprocessing.scale(tprt_series.reshape((-1,1))).reshape((-1)) 
    tprt_mean=tprt_series.mean()
    print(tprt_mean)
    tprt_std=tprt_series.std()
    print(tprt_std)
    return ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask,tprt_mean,tprt_std


def read_multi_site_data(start=0,num=31,include_intenseNormal=True,is_scale=True):
    
    #考虑装机容量，在除以装机容量后再进行标准化
    
    root="../../data/"
    pow_path='20200601after/site_output'
    sites_info_path="20200601after/site_list.csv"
    sites_info=pd.read_csv(root+sites_info_path,sep=',')
    City_dict={"30100005":"Shanxi","30100007":"Guiyang","30100008":"Chaohu","30100009":"Guiyang","30100010":"Guangyuan",
               "30100012":"Chengdu","30100013":"Gaobei","30100014":"Shenyang","30100015":"Weifang","30100020":"Xianyang",
               "30100035":"Xixiang","30100045":"Nanyang","30100046":"Kunming","30100059":"Jiangxi","30100060":"Jining",
               "30100061":"Chaohu","30100064":"Xianyang","30100094":"Shenyang","30100107":"Xishuangbanna","30100109":"Nanjing",
               "30100113":"Xining","30100115":"Haerbing","30100119":"Shanxi","30100137":"Yichang","30100132":"Haerbing",
               "30100139":"Hefei","30100142":"Heilongjiang","30100144":"Hubei","30100143":"Guangdong","30100146":"Hefei","30100263":"Zibo"}
    
    ra_series_scale_whole=np.empty(shape=(0))
    pow_series_scale_whole=np.empty(shape=(0))
    tprt_series_scale_whole=np.empty(shape=(0))
    
    ra_series_mean_whole={}
    pow_series_mean_whole={}
    tprt_series_mean_whole={}
    ra_series_std_whole={}
    pow_series_std_whole={}
    tprt_series_std_whole={}
    
    date_series_whole=np.empty(shape=(0))
    site_num_series_whole=np.empty(shape=(0))
    
    if include_intenseNormal==True:
        keyWord="nonzero"
    else:
        keyWord="unnormal"
    
    
    print(root+pow_path)
    for (_,_,files) in os.walk(root+pow_path):
    #ra_path=args.ra_path
        
        for file in list(files)[start:start+num]:
            
            site_num=(file.split('_')[-1].strip('.csv'))
            #print(site_num)
            if site_num=="30100132":
                continue
            print(site_num)
            cacp=sites_info.loc[sites_info["proj_id"].astype('str')==site_num,'cacp'].values[0]
            #print(cacp)
            ref_filename=f"siteId_{site_num}/site_{site_num}_{keyWord}.csv"
            ref_data_path=root+ref_filename
            ref_series_data=pd.read_csv(ref_data_path,sep=',')
            #print(ref_series_data["instantaneous_output_power"].head(5))
            #pow_data[datetime]=pd.to_datetime(pow_data[pow_datetime])
            ref_series_data["timestamp"]=pd.to_datetime(ref_series_data["timestamp"])
            date_series=ref_series_data["timestamp"].dt.date.values
            site_num_series=np.array([site_num]*len(date_series))
            #print(date_series)
            
            ref_pow_series=ref_series_data["instantaneous_output_power"].values/cacp
            #print(ref_pow_series[:5])
            
            pow_scaler = preprocessing.StandardScaler().fit(ref_pow_series.reshape((-1,1))) #用标准均值和方差转化
            ref_pow_series_scale=pow_scaler.transform(ref_pow_series.reshape((-1,1))).reshape((-1))
            if not is_scale:
                ref_pow_series_mean=ref_pow_series.mean()
                ref_pow_series_std=ref_pow_series.std()
            
            ref_ra_series=ref_series_data["instantaneous_global_radiation"].values
            
            ra_scaler = preprocessing.StandardScaler().fit(ref_ra_series.reshape((-1,1))) #用标准均值和方差转化
            ref_ra_series_scale=ra_scaler.transform(ref_ra_series.reshape((-1,1))).reshape((-1))
            if not is_scale:
                ref_ra_series_mean=ref_ra_series.mean()
                #print(ref_ra_series_mean)
                ref_ra_series_std=ref_ra_series.std()
            
            ref_tprt_series=ref_series_data["clearsky_Tamb"].values
            ref_tprt_series_scale=preprocessing.scale(ref_tprt_series.reshape((-1,1))).reshape((-1))
            if not is_scale:
                ref_tprt_series_mean=ref_tprt_series.mean()
                ref_tprt_series_std=ref_tprt_series.std()
 
            #数据聚合
            date_series_whole=np.concatenate([date_series_whole,date_series])
            site_num_series_whole=np.concatenate([site_num_series_whole,site_num_series])
            
            ra_series_scale_whole=np.concatenate([ra_series_scale_whole,ref_ra_series_scale])
            pow_series_scale_whole=np.concatenate([pow_series_scale_whole,ref_pow_series_scale])
            tprt_series_scale_whole=np.concatenate([tprt_series_scale_whole,ref_tprt_series_scale])
            #print("ra_series_scale_whole.shape",ra_series_scale_whole.shape)
            #return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,date_series_whole,site_num_series_whole
            if  not is_scale:
                ra_series_mean_whole[site_num]=ref_ra_series_mean
                pow_series_mean_whole[site_num]=ref_pow_series_mean
                tprt_series_mean_whole[site_num]=ref_tprt_series_mean
                ra_series_std_whole[site_num]=ref_ra_series_std
                pow_series_std_whole[site_num]=ref_pow_series_std
                tprt_series_std_whole[site_num]=ref_tprt_series_std
                #print("ra_series_noscale_whole.shape",ra_series_noscale_whole.shape)
                #return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,ra_series_noscale_whole,pow_series_noscale_whole,tprt_series_noscale_whole,date_series_whole,site_num_series_whole
    if  is_scale:
        return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,date_series_whole,site_num_series_whole
    else:
        return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,ra_series_mean_whole,pow_series_mean_whole,tprt_series_mean_whole,ra_series_std_whole,pow_series_std_whole,tprt_series_std_whole,date_series_whole,site_num_series_whole,City_dict
        
def read_multi_site_data_with_simulate(start=0,num=31,include_intenseNormal=True,is_scale=True):
    
    #考虑装机容量，在除以装机容量后再进行标准化
    
    root="./data/"
    pow_path='20200601after/site_output'
    sites_info_path="20200601after/site_list.csv"
    sites_info=pd.read_csv(root+sites_info_path,sep=',')
    City_dict={"30100005":"Shanxi","30100007":"Guiyang","30100008":"Chaohu","30100009":"Guiyang","30100010":"Guangyuan",
               "30100012":"Chengdu","30100013":"Gaobei","30100014":"Shenyang","30100015":"Weifang","30100020":"Xianyang",
               "30100035":"Xixiang","30100045":"Nanyang","30100046":"Kunming","30100059":"Jiangxi","30100060":"Jining",
               "30100061":"Chaohu","30100064":"Xianyang","30100094":"Shenyang","30100107":"Xishuangbanna","30100109":"Nanjing",
               "30100113":"Xining","30100115":"Haerbing","30100119":"Shanxi","30100137":"Yichang","30100132":"Haerbing",
               "30100139":"Hefei","30100142":"Heilongjiang","30100144":"Hubei","30100143":"Guangdong","30100146":"Hefei","30100263":"Zibo"}
    
    ra_series_scale_whole=np.empty(shape=(0))
    pow_series_scale_whole=np.empty(shape=(0))
    tprt_series_scale_whole=np.empty(shape=(0))
    
    ra_series_noscale_simulate_whole=np.empty(shape=(0))
    pow_series_noscale_simulate_whole=np.empty(shape=(0))
    tprt_series_noscale_simulate_whole=np.empty(shape=(0))
    
    ra_series_mean_whole={}
    pow_series_mean_whole={}
    tprt_series_mean_whole={}
    ra_series_std_whole={}
    pow_series_std_whole={}
    tprt_series_std_whole={}
    
    date_series_whole=np.empty(shape=(0))
    site_num_series_whole=np.empty(shape=(0))
    date_series_simulate_whole=np.empty(shape=(0))
    site_num_series_simulate_whole=np.empty(shape=(0))
    
    if include_intenseNormal==True:
        keyWord="nonzero"
    else:
        keyWord="unnormal"
    
    
    print(root+pow_path)
    for (_,_,files) in os.walk(root+pow_path):
    #ra_path=args.ra_path
        
        for file in list(files)[start:start+num]:
            
            site_num=(file.split('_')[-1].strip('.csv'))
            #print(site_num)
            if site_num=="30100132":
                continue
            print(site_num)
            cacp=sites_info.loc[sites_info["proj_id"].astype('str')==site_num,'cacp'].values[0]
            #print(cacp)
            ref_filename=f"siteId_{site_num}/site_{site_num}_{keyWord}.csv"
        
            ref_data_path=root+ref_filename
            ref_series_data=pd.read_csv(ref_data_path,sep=',')
            #print(ref_series_data["instantaneous_output_power"].head(5))
            #pow_data[datetime]=pd.to_datetime(pow_data[pow_datetime])
            ref_series_data["timestamp"]=pd.to_datetime(ref_series_data["timestamp"])
            date_series=ref_series_data["timestamp"].dt.date.values
            site_num_series=np.array([site_num]*len(date_series))
            #print(date_series)
            
            ref_pow_series=ref_series_data["instantaneous_output_power"].values/(cacp*1000)
            #print(ref_pow_series[:5])
            
            pow_scaler = preprocessing.StandardScaler().fit(ref_pow_series.reshape((-1,1))) #用标准均值和方差转化
            ref_pow_series_scale=pow_scaler.transform(ref_pow_series.reshape((-1,1))).reshape((-1))
            if not is_scale:
                ref_pow_series_mean=ref_pow_series.mean()
                ref_pow_series_std=ref_pow_series.std()
            
            ref_ra_series=ref_series_data["instantaneous_global_radiation"].values
            
            ra_scaler = preprocessing.StandardScaler().fit(ref_ra_series.reshape((-1,1))) #用标准均值和方差转化
            ref_ra_series_scale=ra_scaler.transform(ref_ra_series.reshape((-1,1))).reshape((-1))
            if not is_scale:
                ref_ra_series_mean=ref_ra_series.mean()
                #print(ref_ra_series_mean)
                ref_ra_series_std=ref_ra_series.std()
            
            ref_tprt_series=ref_series_data["clearsky_Tamb"].values
            ref_tprt_series_scale=preprocessing.scale(ref_tprt_series.reshape((-1,1))).reshape((-1))
            if not is_scale:
                ref_tprt_series_mean=ref_tprt_series.mean()
                ref_tprt_series_std=ref_tprt_series.std()
                
            #读取模拟数据
            series_data_simulate=pd.read_csv(root+f"siteId_{site_num}/site_{site_num}_simulate.csv",sep=',')
            ra_series_simulate=series_data_simulate["instantaneous_global_radiation"].values
            pow_series_simulate=series_data_simulate["modeled_power"].values/cacp
            tprt_series_simulate=series_data_simulate["clearsky_Tamb"].values
            series_data_simulate["timestamp"]=pd.to_datetime(series_data_simulate["timestamp"])
            date_series_simulate=series_data_simulate["timestamp"].dt.date.values
            site_num_series_simulate=np.array([site_num]*len(date_series_simulate))
            
            
            ra_series_scale_simulate = preprocessing.scale(ra_series_simulate.reshape((-1,1))).reshape((-1)) #正太分布scacle
            pow_series_scale_simulate = preprocessing.scale(pow_series_simulate.reshape((-1,1))).reshape((-1))
            tprt_series_scale_simulate = preprocessing.scale(tprt_series_simulate.reshape((-1,1))).reshape((-1))
 
            #数据聚合
            date_series_whole=np.concatenate([date_series_whole,date_series])
            site_num_series_whole=np.concatenate([site_num_series_whole,site_num_series])
            
            ra_series_scale_whole=np.concatenate([ra_series_scale_whole,ref_ra_series_scale])
            pow_series_scale_whole=np.concatenate([pow_series_scale_whole,ref_pow_series_scale])
            tprt_series_scale_whole=np.concatenate([tprt_series_scale_whole,ref_tprt_series_scale])
            
            #模拟数据
            date_series_simulate_whole=np.concatenate([date_series_simulate_whole,date_series_simulate])
            site_num_series_simulate_whole=np.concatenate([site_num_series_simulate_whole,site_num_series_simulate])
            
            ra_series_noscale_simulate_whole=np.concatenate([ra_series_noscale_simulate_whole,ra_series_scale_simulate])
            pow_series_noscale_simulate_whole=np.concatenate([pow_series_noscale_simulate_whole,pow_series_scale_simulate])
            tprt_series_noscale_simulate_whole=np.concatenate([tprt_series_noscale_simulate_whole,tprt_series_scale_simulate])
            
            
            
            #print("ra_series_scale_whole.shape",ra_series_scale_whole.shape)
            #return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,date_series_whole,site_num_series_whole
            if  not is_scale:
                ra_series_mean_whole[site_num]=ref_ra_series_mean
                pow_series_mean_whole[site_num]=ref_pow_series_mean
                tprt_series_mean_whole[site_num]=ref_tprt_series_mean
                ra_series_std_whole[site_num]=ref_ra_series_std
                pow_series_std_whole[site_num]=ref_pow_series_std
                tprt_series_std_whole[site_num]=ref_tprt_series_std
                #print("ra_series_noscale_whole.shape",ra_series_noscale_whole.shape)
                #return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,ra_series_noscale_whole,pow_series_noscale_whole,tprt_series_noscale_whole,date_series_whole,site_num_series_whole
    if  is_scale:
        return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,date_series_whole,site_num_series_whole
    else:
        return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole,ra_series_mean_whole,pow_series_mean_whole,tprt_series_mean_whole,ra_series_std_whole,pow_series_std_whole,tprt_series_std_whole,date_series_whole,site_num_series_whole,City_dict,\
    ra_series_noscale_simulate_whole,pow_series_noscale_simulate_whole,tprt_series_noscale_simulate_whole,date_series_simulate_whole,site_num_series_simulate_whole
            
#两种读取方式
    #1.分别处理每天的数据然后concat
    #2.将所有的真实数据放在一起concat,模拟数据放在一起concat
def read_whole_enhance_data_1(num=21):
    root="./data/"
    pow_path='20200601after/site_output'
    
    ra_series_scale_whole=np.empty(shape=(0))
    pow_series_scale_whole=np.empty(shape=(0))
    tprt_series_scale_whole=np.empty(shape=(0))
    
    print(root+pow_path)
    for (_,_,files) in os.walk(root+pow_path):
    #ra_path=args.ra_path
        
        for file in list(files)[:num]:
            
            site_num=(file.split('_')[-1].strip('.csv'))
            print(site_num)
            if site_num=="30100132":
                continue
            print(site_num)
            ref_filename=f"siteId_{site_num}/site_{site_num}_instense_normal.csv"
            
            ref_data_path=root+ref_filename
            ref_series_data=pd.read_csv(ref_data_path,sep=',')
            
            ref_pow_series=ref_series_data["instantaneous_output_power"].values
            pow_scaler = preprocessing.StandardScaler().fit(ref_pow_series.reshape((-1,1))) #用标准均值和方差转化
            ref_pow_series_scale=pow_scaler.transform(ref_pow_series.reshape((-1,1))).reshape((-1))
            
            ref_ra_series=ref_series_data["instantaneous_global_radiation"].values
            ra_scaler = preprocessing.StandardScaler().fit(ref_ra_series.reshape((-1,1))) #用标准均值和方差转化
            ref_ra_series_scale=ra_scaler.transform(ref_ra_series.reshape((-1,1))).reshape((-1))
            
            ref_tprt_series=ref_series_data["clearsky_Tamb"].values

            ref_tprt_series_scale=preprocessing.scale(ref_tprt_series.reshape((-1,1))).reshape((-1))
            
            
            filename=f"siteId_{site_num}/site_{site_num}_dataEnhance_instense_normal.csv"
            data_path=root+filename
            
            series_data=pd.read_csv(data_path,sep=',')
            ra_series=series_data["instantaneous_global_radiation"].values
            pow_series=series_data["instantaneous_output_power"].values
            tprt_series=series_data["clearsky_Tamb"].values
            
            pow_series_scale=pow_scaler.transform(pow_series.reshape((-1,1))).reshape((-1))
            ra_series_scale=ra_scaler.transform(ra_series.reshape((-1,1))).reshape((-1))
            
            #ra_series_scale = preprocessing.scale(ra_series.reshape((-1,1))).reshape((-1)) #正太分布scacle
            #pow_series_scale = preprocessing.scale(pow_series.reshape((-1,1))).reshape((-1))
            tprt_series_scale=preprocessing.scale(tprt_series.reshape((-1,1))).reshape((-1))
            
            
            #读取模拟数据
            series_data_simulate=pd.read_csv(root+f"siteId_{site_num}/site_{site_num}_simulate.csv",sep=',')
            ra_series_simulate=series_data_simulate["instantaneous_global_radiation"].values
            pow_series_simulate=series_data_simulate["modeled_power"].values
            tprt_series_simulate=series_data_simulate["clearsky_Tamb"].values
            
            
            ra_series_scale_simulate = preprocessing.scale(ra_series_simulate.reshape((-1,1))).reshape((-1)) #正太分布scacle
            pow_series_scale_simulate = preprocessing.scale(pow_series_simulate.reshape((-1,1))).reshape((-1))
            tprt_series_scale_simulate = preprocessing.scale(tprt_series_simulate.reshape((-1,1))).reshape((-1))
            
            
            #数据聚合
            ra_series_scale_whole=np.concatenate([ra_series_scale_whole,ref_ra_series_scale,ra_series_scale,ra_series_scale_simulate])
            pow_series_scale_whole=np.concatenate([pow_series_scale_whole,ref_pow_series_scale,pow_series_scale,pow_series_scale_simulate])
            tprt_series_scale_whole=np.concatenate([tprt_series_scale_whole,ref_tprt_series_scale,tprt_series_scale,tprt_series_scale_simulate])
            
            
            print(ra_series_scale_whole.shape)
            
    return ra_series_scale_whole,pow_series_scale_whole,tprt_series_scale_whole

if __name__ == '__main__':
    #ra_series_scale,pow_series_scale,tprt_series_scale=read_whole_enhance_data_1()
    #ra_series_scale,pow_series_scale,tprt_series_scale,anomaly_mask=read_multi_anomaly_data()
    #print(ra_series_scale.shape,pow_series_scale.shape,tprt_series_scale.shape)
    #print(anomaly_mask[100000:])
    
    ra_series_scale,pow_series_scale,tprt_series_scale,date_series,site_num_series=read_multi_site_data(start=21,num=10,include_intenseNormal=False,is_scale=False)
    #print(date_series[:100])
    #print(ra_series_scale[:30])
    #print(site_num_series)
    