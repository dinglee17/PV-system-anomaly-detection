# -*- coding: utf-8 -*-
"""
@author: 李鼎
"""
import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from read_data import read_single_anomaly_case
import xgboost as xgb
import sklearn
from sklearn.model_selection import GridSearchCV
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def plot_embedding(data, labels, title, num_perclass, agg_flag=False, equ_ano_flag=False, wea_ano_flag=False, savefig=None):

    if agg_flag == True:
        label_list = ["equip_anomaly", "weather_anomaly"]
        num_perclass *= 3

    elif equ_ano_flag == True:
        label_list = ["inverter_fault_z", "spike_z", "lowValue_z"]

    elif wea_ano_flag == True:
        label_list = ["snowy_z", "cloudy_z", "shading_z"]

    else:
        label_list = ["inverter_fault_z", "spike_z",
                      "snowy_z", "cloudy_z", "lowValue_z", "shading_z"]

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(111, projection='3d')
    for i in range(data.shape[0]):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], s=20,
                   color=plt.cm.Set1(labels[i] / 7)
                   )
    # set up for handles declaration
    patches = [mpatches.Patch(color=plt.cm.Set1(
        labels[i] / 7), label=label_list[int(labels[i])]) for i in range(0, len(labels), num_perclass)]
    print(patches)
    angle1 = 0
    angle2 = 60
    ax.view_init(angle1, angle2)  # 初始化视角 angle1沿着y轴旋转,angle2沿着z轴旋转
    ax.legend(handles=patches, loc='best', fontsize=20)
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    # plt.zticks([])
    # plt.title(title)
    if savefig is not None:
        fig.savefig(savefig)
    return fig


def plot_embedding_2d(data, labels, title, num_perclass, agg_flag=False, equ_ano_flag=False, wea_ano_flag=False, savefig=None):

    label_dict = {0: "inverter_fault_z", 1: "spike_z",
                  2: "snowy_z", 3: "cloudy_z", 4: "lowValue_z", 5: "shading_z"}

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        ax.scatter(data[i, 0], data[i, 1], s=15, alpha=0.6,
                   color=plt.cm.Set1(labels[i] / 7)
                   )
    patches = [mpatches.Patch(color=plt.cm.Set1(
        labels[i] / 7), label=label_dict[int(labels[i])]) for i in range(0, len(labels), num_perclass)]
    ax.legend(handles=patches, loc='best', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(title)
    if savefig is not None:
        fig.savefig(savefig)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="device, e.g. cpu, cuda:0, cuda:1")
    parser.add_argument("--latent_var_path", type=str, default="re_z",
                        help="saved model path")
    parser.add_argument("--model_path", type=str, default="saved_model/SCVAE_savedmodel.pth",
                        help="saved model path")
    parser.add_argument("--latent_mode", type=str, choices=['post', 'prior', 'predict', 'diff'], default='diff',
                        help="the mode of using latent variable")
    parser.add_argument("--is_trainsites", type=bool, default=True,
                        help="whether use trainsites or testsites")
    parser.add_argument("--anomaly_type", type=str, choices=["equipment_anomaly", "weather_equipment_2class", "whole_anomaly", "weather_anomaly"],
                        default="equipment_anomaly", help="The anomaly type to clustering or classification")
    parser.add_argument("--task", type=str, choices=['clustering', 'classification'], default='clustering',
                        help="clustering or classification")
    parser.add_argument("--classifier", type=str, choices=['SVM', 'XGB', 'MLP'], default='XGB',
                        help="classification methods")
    opt = parser.parse_args()

    num = 4
    end = 8

    train_flag = opt.is_trainsites

    # File_list=["enhance_smooth_intense_normal_data_of_20trainsites.csv","enhance_spike_anomaly_data_of_20trainsites.csv","enhance_inverter_fault_anomaly_data_of_20trainsites.csv","enhance_snowy_anomaly_data_of_20trainsites.csv","enhance_cloudy_anomaly_data_of_20trainsites.csv","enhance_lowValue_anomaly_data_of_20trainsites.csv","enhance_shading_anomaly_data_of_20trainsites.csv"]

    case = ["normal", "spike", "inverterFault",
            "snowy", "cloudy", "lowValue", "shading"]

    classifier = ""
    task = opt.task
    mode = opt.latent_mode
    latent_var_path = opt.latent_var_path

    TSNE_flag = False
    if task == "clustering":
        TSNE_flag = True
    else:
        classifier = opt.classifier

    detect_mode_list = ["equipment_anomaly", "weather_equipment_2class",
                        "whole_anomaly", "weather_anomaly"]
    detect_mode_index = detect_mode_list.index(opt.anomaly_type)

    if detect_mode_index == 0:
        weather_equipment_2anomaly_flag = False
        equipment_anomaly_detect_flag = True
        weather_anomaly_detect_flag = False
    elif detect_mode_index == 1:
        weather_equipment_2anomaly_flag = True
        equipment_anomaly_detect_flag = False
        weather_anomaly_detect_flag = False
    elif detect_mode_index == 2:
        weather_equipment_2anomaly_flag = False
        equipment_anomaly_detect_flag = False
        weather_anomaly_detect_flag = False
    elif detect_mode_index == 3:
        weather_equipment_2anomaly_flag = False
        equipment_anomaly_detect_flag = False
        weather_anomaly_detect_flag = True

    if train_flag:
        dataset = "trainsites"
    else:
        dataset = "testsites"

    normal_z = np.load(latent_var_path+"/re_post_zs_"+dataset+"_normal.npy")
    # normal_z=normal_z[:,num:end,:]
    # normal_z=normal_z.reshape(normal_z.shape[0],-1)
    average_normal_z = normal_z.mean(axis=0)
    print("average_normal_z.shape", average_normal_z.shape)

    if mode == "post":
        normal_z = np.load(latent_var_path+"/re_post_zs_" +
                           dataset+"_normal.npy")
    elif mode == "prior":
        normal_z = np.load(latent_var_path+"/re_prior_zs_" +
                           dataset+"_normal.npy")
    elif mode == "predict":
        normal_z = np.load(
            latent_var_path+"/re_predict_zs_"+dataset+"_normal.npy")
    elif mode == "diff":
        normal_z1 = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_normal.npy")
        normal_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_normal.npy")
        # normal_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_normal.npy")
        normal_z = normal_z1-normal_z2

    normal_z = normal_z[:, num:end, :]
    normal_z = normal_z.reshape(normal_z.shape[0], -1)
    label_normal = np.zeros((normal_z.shape[0]))

    if mode == "post":
        inverter_fault_z = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_inverterFault.npy")
    elif mode == "prior":
        inverter_fault_z = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_inverterFault.npy")
    elif mode == "predict":
        inverter_fault_z = np.load(
            latent_var_path+"/re_predict_zs_"+dataset+"_inverterFault.npy")
    elif mode == "diff":
        inverter_fault_z1 = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_inverterFault.npy")
        inverter_fault_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_inverterFault.npy")
        # inverter_fault_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_inverterFault.npy")
        inverter_fault_z = inverter_fault_z1-inverter_fault_z2

    inverter_fault_z = inverter_fault_z[:, num:end, :]
    # print(inverter_fault_z.shape)
    inverter_fault_z = inverter_fault_z.reshape(
        inverter_fault_z.shape[0], -1)
    print(inverter_fault_z.shape)
    num_perclass = inverter_fault_z.shape[0]
    label_inverter_fault = np.ones((inverter_fault_z.shape[0]))

    if mode == "post":
        spike_z = np.load(latent_var_path+"/re_post_zs_"+dataset+"_spike.npy")
    elif mode == "prior":
        spike_z = np.load(latent_var_path+"/re_prior_zs_"+dataset+"_spike.npy")
    elif mode == "predict":
        spike_z = np.load(latent_var_path+"/re_predict_zs_" +
                          dataset+"_spike.npy")
    elif mode == "diff":
        spike_z1 = np.load(latent_var_path+"/re_post_zs_"+dataset+"_spike.npy")
        spike_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_spike.npy")
        # spike_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_spike.npy")
        spike_z = spike_z1-spike_z2

    spike_z = spike_z[:, num:end, :]
    spike_z = spike_z.reshape(spike_z.shape[0], -1)
    # print(spike_z)

    if weather_equipment_2anomaly_flag == False:
        label_spike = np.ones((spike_z.shape[0]))*2
    else:
        label_spike = np.ones((spike_z.shape[0]))

    if mode == "post":
        snowy_z = np.load(latent_var_path+"/re_post_zs_"+dataset+"_snowy.npy")
    elif mode == "prior":
        snowy_z = np.load(latent_var_path+"/re_prior_zs_"+dataset+"_snowy.npy")
    elif mode == "predict":
        snowy_z = np.load(latent_var_path+"/re_predict_zs_" +
                          dataset+"_snowy.npy")
    elif mode == "diff":
        snowy_z1 = np.load(latent_var_path+"/re_post_zs_"+dataset+"_snowy.npy")
        snowy_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_snowy.npy")
        # snowy_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_snowy.npy")
        snowy_z = snowy_z1-snowy_z2

    snowy_z = snowy_z[:, num:end, :]
    snowy_z = snowy_z.reshape(snowy_z.shape[0], -1)
    if weather_equipment_2anomaly_flag == False:
        label_snowy = np.ones((snowy_z.shape[0]))*3
    else:
        label_snowy = np.ones((spike_z.shape[0]))*2

    if mode == "post":
        cloudy_z = np.load(latent_var_path+"/re_post_zs_" +
                           dataset+"_cloudy.npy")
    elif mode == "prior":
        cloudy_z = np.load(latent_var_path+"/re_prior_zs_" +
                           dataset+"_cloudy.npy")
    elif mode == "predict":
        cloudy_z = np.load(
            latent_var_path+"/re_predict_zs_"+dataset+"_cloudy.npy")
    elif mode == "diff":
        cloudy_z1 = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_cloudy.npy")
        cloudy_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_cloudy.npy")
        # cloudy_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_cloudy.npy")
        cloudy_z = cloudy_z1-cloudy_z2

    cloudy_z = cloudy_z[:, num:end, :]
    cloudy_z = cloudy_z.reshape(cloudy_z.shape[0], -1)
    # print(cloudy_z)
    if weather_equipment_2anomaly_flag == False:
        label_cloudy = np.ones((cloudy_z.shape[0]))*4
    else:
        label_cloudy = np.ones((spike_z.shape[0]))*2

    if mode == "post":
        lowValue_z = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_lowValue.npy")
    elif mode == "prior":
        lowValue_z = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_lowValue.npy")
    elif mode == "predict":
        lowValue_z = np.load(
            latent_var_path+"/re_predict_zs_"+dataset+"_lowValue.npy")
    elif mode == "diff":
        lowValue_z1 = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_lowValue.npy")
        lowValue_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_lowValue.npy")
        # lowValue_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_lowValue.npy")
        lowValue_z = lowValue_z1-lowValue_z2

    lowValue_z = lowValue_z[:, num:end, :]
    lowValue_z = lowValue_z.reshape(lowValue_z.shape[0], -1)
    # print(lowValue_z)
    if weather_equipment_2anomaly_flag == False:
        label_lowValue = np.ones((lowValue_z.shape[0]))*5
    else:
        label_lowValue = np.ones((spike_z.shape[0]))*1

    if mode == "post":
        shading_z = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_shading.npy")
    elif mode == "prior":
        shading_z = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_shading.npy")
    elif mode == "predict":
        shading_z = np.load(
            latent_var_path+"/re_predict_zs_"+dataset+"_shading.npy")
    elif mode == "diff":
        shading_z1 = np.load(
            latent_var_path+"/re_post_zs_"+dataset+"_shading.npy")
        shading_z2 = np.load(
            latent_var_path+"/re_prior_zs_"+dataset+"_shading.npy")
        # shading_z2=np.load(latent_var_path+"/re_predict_zs_"+dataset+"_shading.npy")
        shading_z = shading_z1-shading_z2

    shading_z = shading_z[:, num:end, :]
    shading_z = shading_z.reshape(shading_z.shape[0], -1)
    # print(shading_z)
    if weather_equipment_2anomaly_flag == False:
        label_shading = np.ones((shading_z.shape[0]))*6
    else:
        label_shading = np.ones((spike_z.shape[0]))*2

    classify_2 = False
    if classify_2:
        data = np.concatenate([normal_z, shading_z], axis=0)
        label = np.concatenate([label_normal, label_shading], axis=0)
    elif weather_equipment_2anomaly_flag:
        data = np.concatenate(
            [inverter_fault_z, spike_z, snowy_z, cloudy_z, lowValue_z, shading_z], axis=0)
        # label=np.concatenate([label_inverter_fault-1,label_spike-1,label_snowy-1,label_cloudy-1,label_lowValue-1,label_shading-1],axis=0)
        label = np.concatenate([label_inverter_fault, label_spike, label_snowy,
                                label_cloudy, label_lowValue, label_shading], axis=0)
        label = label-1
    elif equipment_anomaly_detect_flag:
        data = np.concatenate([inverter_fault_z, spike_z, lowValue_z], axis=0)
        # label=np.concatenate([label_inverter_fault-1,label_spike-1,label_lowValue-3],axis=0)
        label = np.concatenate(
            [label_inverter_fault, label_spike, label_lowValue], axis=0)
        label = label-1
    elif weather_anomaly_detect_flag:
        data = np.concatenate([snowy_z, cloudy_z, shading_z], axis=0)
        # label=np.concatenate([label_snowy-3,label_cloudy-3,label_shading-4],axis=0)
        label = np.concatenate(
            [label_snowy, label_cloudy, label_shading], axis=0)
        label = label-1

    else:
        # data=np.concatenate([normal_z,inverter_fault_z,spike_z,snowy_z,cloudy_z,lowValue_z,shading_z],axis=0)
        # label=np.concatenate([label_normal,label_inverter_fault,label_spike,label_snowy,label_cloudy,label_lowValue,label_shading],axis=0)
        data = np.concatenate(
            [inverter_fault_z, spike_z, snowy_z, cloudy_z, lowValue_z, shading_z], axis=0)
        # label=np.concatenate([label_inverter_fault-1,label_spike-1,label_snowy-1,label_cloudy-1,label_lowValue-1,label_shading-1],axis=0)
        label = np.concatenate([label_inverter_fault, label_spike, label_snowy,
                                label_cloudy, label_lowValue, label_shading], axis=0)
        label = label-1

    print(data.shape)
    print(label.shape)
    print(label)

    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.25, random_state=42, stratify=label)

    # TSNE

    if train_flag:
        site = "trainsite"
    else:
        site = "testsite"

    if TSNE_flag == True:
        tsne = TSNE(n_components=3, init='pca', random_state=0)
        t0 = time()
        result = tsne.fit_transform(data)

        print(equipment_anomaly_detect_flag)
        fig = plot_embedding_2d(result, label,
                                't-SNE embedding of the latent variables (time %.2fs)'
                                % (time() - t0), num_perclass, weather_equipment_2anomaly_flag, equipment_anomaly_detect_flag, weather_anomaly_detect_flag)  # f"imgs/{detect_mode_list[detect_mode_index]}_{mode}_{site}.png"
        plt.show()

    # print(clf.score(data_train, label_train)) #0.76 如果分为设备异常和天气异常 0.8858
    # print(clf.score(data_test, label_test)) #0.75 0.8717

    # svm二分类
    if classifier == "SVM":

        # svm 多分类结果

        clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
        clf.fit(data_train, label_train)
        pred_y_train = clf.predict(data_train)
        pred_y_test = clf.predict(data_test)
        print("train set accuracy", sklearn.metrics.accuracy_score(
            label_train, pred_y_train))
        print("test set accuracy", sklearn.metrics.accuracy_score(
            label_test, pred_y_test))

    elif classifier == "XGB":

        #data_train,data_val,label_train,label_val = train_test_split(data_train,label_train,test_size=0.2, random_state=42,stratify=label_train)

        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',   # 回归任务设置为：'objective': 'reg:gamma',
            # 'objective': 'reg:gamma',
            'num_class': 6,      # 回归任务没有这个参数
            'gamma': 1,
            'max_depth': 5,
            'lambda': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'silent': 0,
            'eta': 0.1,
            'seed': 42,
            'nthread': 4,
            'n_estimators': 200,
            'learning_rate': 0.1
        }

        # print(type(params.items()))
        plst = list(params.items())

        #xlf = xgb.XGBClassifier(*plst)

        # grid_search
        #gsearch = GridSearchCV(xlf, param_grid=grid_search_params, scoring='accuracy', cv=3)
        #gsearch.fit(data_train, label_train)
        #print("Best score: %0.3f" % gsearch.best_score_)
        #print("Best parameters set:")
        #best_parameters = gsearch.best_estimator_.get_params()
        # for param_name in sorted(grid_search_params.keys()):
        #print("\t%s: %r" % (param_name, best_parameters[param_name]))

        num_round = 300
        dtrain = xgb.DMatrix(data_train, label_train)
        model = xgb.train(plst, dtrain, num_round)
        dtest = xgb.DMatrix(data_train)
        pred_y_train = model.predict(dtest)
        dtest = xgb.DMatrix(data_test)
        pred_y_test = model.predict(dtest)

        print("train set accuracy", sklearn.metrics.accuracy_score(
            label_train, pred_y_train))
        print("test set accuracy", sklearn.metrics.accuracy_score(
            label_test, pred_y_test))

    # MLP

    elif classifier == "MLP":

        device = torch.device(opt.device)

        class MLP(nn.Module):

            def __init__(self, input_dim, hidden_dim, class_num):
                super(MLP, self).__init__()
                self.inputLayer = nn.Linear(input_dim, hidden_dim)
                self.BatchNorm1 = torch.nn.BatchNorm1d(hidden_dim)
                self.hiddenLayer = nn.Linear(hidden_dim, hidden_dim)
                self.BatchNorm2 = torch.nn.BatchNorm1d(hidden_dim)
                self.outputLayer = nn.Linear(hidden_dim, class_num)
                self.relu = nn.ReLU()

            def forward(self, X):

                X = self.relu(self.inputLayer(X))
                X = self.BatchNorm1(X)
                X = self.relu(self.hiddenLayer(X))
                X = self.BatchNorm2(X)
                logits = self.relu(self.outputLayer(X))

                return logits

        def train_one_epoch(train_loader, model, loss_fn, optimizer):
            size = len(train_loader.dataset)
            model.train()
            for batch, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        def test_one_epoch(test_loader, model, loss_fn):
            size = len(test_loader.dataset)
            batches = len(test_loader)
            model.eval()
            loss, acc = 0, 0
            for batch, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss += loss_fn(logits, y)
                acc += (logits.argmax(1) == y).type(torch.float).sum().item()

            loss /= batches
            acc /= size

            print(f"Test acc:{acc:>8f} \n , Avg loss: {loss:>8f} \n")
            return acc

        def train(train_loader, val_loader, test_loader, model, loss_fn, optimizer, epochs=100):
            best_acc = 0.90
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                model.train()  # 设置训练和推理模式，防止两种模式之间弄混
                train_one_epoch(train_loader, model, loss_fn, optimizer)
                model.eval()
                val_acc = test_one_epoch(val_loader, model, loss_fn)
                if val_acc > best_acc:
                    print(val_acc)
                    best_acc = val_acc
                    os.makedirs("MLP_classifier", exist_ok=True)
                    torch.save(model.state_dict(),
                               "MLP_classifier/MLP_Classifier.pth")
                    print("Saved PyTorch Model State to model.pth")

            model.load_state_dict(torch.load(
                "MLP_classifier/MLP_Classifier.pth"))
            model.eval()
            test_acc = test_one_epoch(test_loader, model, loss_fn)
            val_acc = test_one_epoch(val_loader, model, loss_fn)
            print("val_acc", val_acc)
            print("test_acc", test_acc)

            print("Done!")

        data_train, data_val, label_train, label_val = train_test_split(
            data_train, label_train, test_size=0.2, random_state=42, stratify=label_train)

        input_dim = data_train.shape[1]
        data_train = torch.tensor(data_train, dtype=torch.float32)
        data_val = torch.tensor(data_val, dtype=torch.float32)
        data_test = torch.tensor(data_test, dtype=torch.float32)
        label_train = torch.tensor(label_train, dtype=torch.long)
        label_val = torch.tensor(label_val, dtype=torch.long)
        label_test = torch.tensor(label_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(
            data_train, label_train), batch_size=1024, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            data_val, label_val), batch_size=1024, shuffle=True)
        test_loader = DataLoader(TensorDataset(
            data_test, label_test), batch_size=1024, shuffle=True)

        model = MLP(input_dim=512, hidden_dim=1024, class_num=6).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)

        train(train_loader, val_loader, test_loader,
              model, loss_fn, optimizer, 1000)
