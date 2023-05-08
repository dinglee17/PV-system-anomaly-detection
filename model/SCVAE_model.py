# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.distributions import Normal, kl_divergence, StudentT
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as studentT
import scipy.stats as stats


class SCVAE(nn.Module):
    def __init__(self, x_dim, label_dim, h_dim, z_dim, input_dim, bias=False, device=None, is_prior=False):

        super(SCVAE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.is_prior = is_prior

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # ---Network Structure------------
        # feature-extracting transformations
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.ReLU())
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_y = nn.Sequential(
            nn.Linear(label_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Sequential(nn.Linear(h_dim, z_dim),
                                      nn.LeakyReLU(0.1))
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim),
                                        nn.LeakyReLU(0.1))
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # predict
        self.predict_z = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.predict_mean = nn.Sequential(nn.Linear(h_dim, z_dim),
                                          nn.LeakyReLU(0.1))
        self.predict_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        self.dec_prior = nn.Sequential(
            nn.Linear(h_dim + h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        self.dec_predict = nn.Sequential(
            nn.Linear(h_dim + h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        #self.dec_mean = nn.Sequential(nn.Linear(h_dim, label_dim))
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h_dim, input_dim*label_dim),
            nn.LeakyReLU(0.1))

        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, input_dim*label_dim),
            nn.Softplus())

        self.rnn = nn.GRUCell(h_dim*2, h_dim)
        self.rnn2 = nn.GRUCell(h_dim*2, h_dim)

    def reucrrence(self, x_t, y_t, h, h2, recording=True):

        # x shape (seq_len,Batch_size, feature_dim)

        phi_x_t = self.phi_x(x_t)
        phi_y_t = self.phi_y(y_t)

        # encoding
        enc_t = self.enc(torch.cat([phi_x_t, phi_y_t, h], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)
        # prior
        prior_t = self.prior(torch.cat([phi_x_t, h], 1))
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        # prediction model doesn't share the encoder with reconstruction model
        # predict model only considering x
        predict_t = self.predict_z(torch.cat([phi_x_t, h2], 1))
        predict_mean_t = self.predict_mean(predict_t)
        predict_std_t = self.predict_std(predict_t)
        # sampling and reparamerization

        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

        z_t_prior = self._reparameterized_sample(prior_mean_t, prior_std_t)

        z_t_predict = self._reparameterized_sample(
            predict_mean_t, predict_std_t)

        phi_z_t = self.phi_z(z_t)
        #phi_z_t_prior = self.phi_z(z_t_prior)

        phi_z_t_prior = self.phi_z(z_t_prior)
        phi_z_t_predict = self.phi_z(z_t_predict)

        # decoding
        #dec_t = self.dec(torch.cat([phi_z_t, h],1))
        # dec_t condition on x_t and z_t
        dec_t = self.dec(torch.cat([phi_x_t, phi_z_t, h], 1))
        dec_mean_t = self.dec_mean(dec_t)
        # print(dec_mean_t.detach().cpu().numpy())

        dec_std_t = self.dec_std(dec_t)

        # the mean and std decoders are sharing
        # y_t condition on x_t and z_t
        dec_t_prior = self.dec_prior(torch.cat([phi_x_t, phi_z_t_prior, h], 1))
        dec_mean_t_prior = self.dec_mean(dec_t_prior)
        # print(dec_mean_t_prior.detach().cpu().numpy())
        dec_std_t_prior = self.dec_std(dec_t_prior)

        # y_t condition on x_t and z_t
        dec_t_predict = self.dec_predict(
            torch.cat([phi_x_t, phi_z_t_predict, h2], 1))
        dec_mean_t_predict = self.dec_mean(dec_t_predict)
        # print(dec_mean_t_predict.detach().cpu().numpy())
        dec_std_t_predict = self.dec_std(dec_t_predict)

        if recording:

            self.h_chain.append(h)
            self.h2_chain.append(h2)

            self.Z_mean.append(enc_mean_t)
            self.Z_std.append(enc_std_t)
            self.Xr_mean.append(dec_mean_t)
            self.Xr_std.append(dec_std_t)

            self.pZ_mean.append(prior_mean_t)
            self.pZ_std.append(prior_std_t)
            self.Xr_mean_prior.append(dec_mean_t_prior)
            self.Xr_std_prior.append(dec_std_t_prior)

            self.Z_mean_predict.append(predict_mean_t)
            self.Z_std_predict.append(predict_std_t)
            self.Xr_mean_predict.append(dec_mean_t_predict)
            self.Xr_std_predict.append(dec_std_t_predict)

        # h is the hidden state of reconstruction model
        h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), h)
        # h2 is the hidden state of prediction model
        h2 = self.rnn2(torch.cat([phi_x_t, phi_z_t_predict], 1), h2)
        return h, h2

    def predict(self, X, SMC_iter=1):

        # X_shape: (T,batch_size,M,SMC_iter)

        # (Batch_size, seq_len, feature_dim)
        X_emb = torch.squeeze(self.embedding(X), dim=-1)

        mu_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], self.input_dim*self.label_dim, SMC_iter))
        std_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], self.input_dim*self.label_dim, SMC_iter))
        for i in range(SMC_iter):
            h2 = torch.zeros(X.shape[1], self.h_dim).to(self.device)
            for t in range(X.shape[0]):
                x_t = X_emb[t]
                phi_x_t = self.phi_x(x_t)
                # encoding
                predict_t = self.prior(torch.cat([phi_x_t, h2], 1))
                predict_mean_t = self.prior_mean(predict_t)
                predict_std_t = self.prior_std(predict_t)
                # sampling and reparamerization

                z_t = predict_mean_t

                phi_z_t = self.phi_z(z_t)

                # decoding
                dec_t = self.dec(torch.cat([phi_x_t, phi_z_t, h2], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_std_t = self.dec_std(dec_t)
                mu_rec_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_rec_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()

                h2 = self.rnn2(torch.cat([phi_x_t, phi_z_t], 1), h2)
        return mu_rec_chain.mean(axis=3), std_rec_chain.mean(axis=3)

    def predict_withLabel(self, X, Y, SMC_iter=1):

        # X_shape: (T,batch_size,M,SMC_iter)
        # (Batch_size, seq_len, feature_dim)
        X_emb = torch.squeeze(self.embedding(X), dim=-1)
        # (Batch_size, seq_len, feature_dim)
        Y_emb = torch.squeeze(self.embedding(Y), dim=-1)

        Y = Y.view(Y.shape[0], Y.shape[1], self.input_dim*self.label_dim)
        mu_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], self.input_dim*self.label_dim, SMC_iter))
        std_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], self.input_dim*self.label_dim, SMC_iter))
        score_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], self.input_dim*self.label_dim, SMC_iter))
        for i in range(SMC_iter):
            h2 = torch.zeros(X.shape[1], self.h_dim).to(self.device)
            for t in range(X.shape[0]):
                x_t = X_emb[t]
                y_emb_t = Y_emb[t]
                y_t = Y[t]
                phi_x_t = self.phi_x(x_t)
                # encoding
                predict_t = self.prior(torch.cat([phi_x_t, h2], 1))
                predict_mean_t = self.prior_mean(predict_t)
                predict_std_t = self.prior_std(predict_t)
                # sampling and reparamerization

                z_t = predict_mean_t

                phi_z_t = self.phi_z(z_t)

                # decoding
                dec_t = self.dec(torch.cat([phi_x_t, phi_z_t, h2], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_std_t = self.dec_std(dec_t)
                mu_rec_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_rec_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()
                score_chain[t, :, :, i] = -stats.norm.pdf(y_t.cpu().numpy(),
                                                          dec_mean_t.detach().cpu().numpy(),
                                                          dec_std_t.detach().cpu().numpy())
                h2 = self.rnn2(torch.cat([phi_x_t, phi_z_t], 1), h2)
        return mu_rec_chain.mean(axis=3), std_rec_chain.mean(axis=3), score_chain.mean(axis=3)

    def reconstruct(self, X, Y, SMC_iter=1, is_prior=False, is_predict=False):
        # Iuput X shape : (seq_len, input_dim, feature_dim)
        # Input label shape: (Batch_size,seq_len,input_dim)

        # (Batch_size, seq_len, feature_dim)
        X_emb = torch.squeeze(self.embedding(X), dim=-1)

        # (Batch_size, seq_len, feature_dim)
        Y_emb = torch.squeeze(self.embedding(Y), dim=-1)

        Y = Y.view(Y.shape[0], Y.shape[1], self.input_dim*self.label_dim)
        mu_rec_chain = np.zeros(
            shape=(Y.shape[0], Y.shape[1], self.input_dim*self.label_dim, SMC_iter))
        std_rec_chain = np.zeros(
            shape=(Y.shape[0], Y.shape[1], self.input_dim*self.label_dim, SMC_iter))
        score_chain = np.zeros(
            shape=(Y.shape[0], Y.shape[1], self.input_dim*self.label_dim, SMC_iter))
        for i in range(SMC_iter):
            h = torch.zeros(X.shape[1], self.h_dim).to(self.device)
            for t in range(X.shape[0]):
                x_t = X_emb[t]
                y_emb_t = Y_emb[t]
                y_t = Y[t]
                phi_x_t = self.phi_x(x_t)
                phi_y_t = self.phi_y(y_emb_t)
                # encoding
                enc_t = self.enc(torch.cat([phi_x_t, phi_y_t, h], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)
                # prior
                prior_t = self.prior(torch.cat([phi_x_t, h], 1))
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
                # sampling and reparamerization

                if(not is_prior):
                    z_t = enc_mean_t
                else:
                    z_t = prior_mean_t

                phi_z_t = self.phi_z(z_t)

                # decoding
                dec_t = self.dec(torch.cat([phi_x_t, phi_z_t, h], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_std_t = self.dec_std(dec_t)
                mu_rec_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_rec_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()
                score_chain[t, :, :, i] = -stats.norm.pdf(y_t.cpu().numpy(),
                                                          dec_mean_t.detach().cpu().numpy(),
                                                          dec_std_t.detach().cpu().numpy())
                h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), h)
        return mu_rec_chain.mean(axis=3), std_rec_chain.mean(axis=3), score_chain.mean(axis=3)

    def reconstruct_single(self, X, SMC_iter=10):

        SMC_iter = 10
        mu_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], X.shape[2], SMC_iter))
        std_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], X.shape[2], SMC_iter))
        df_rec_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], X.shape[2], SMC_iter))
        score_chain = np.zeros(
            shape=(X.shape[0], X.shape[1], X.shape[2], SMC_iter))
        for i in range(SMC_iter):

            h = torch.zeros(X.shape[1], self.h_dim).to(self.device)
            for t in range(X.shape[0]):
                x_t = X[t]
                phi_x_t = self.phi_x(x_t)
                # encoding
                enc_t = self.enc(torch.cat([phi_x_t, h], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)
                # prior
                prior_t = self.prior(h)
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)
                # sampling and reparamerization
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                phi_z_t = self.phi_z(z_t)

                # decoding
                dec_t = self.dec(torch.cat([phi_z_t, h], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_std_t = self.dec_std(dec_t)

                mu_rec_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_rec_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()
                score_chain[t, :, :, i] = -stats.norm.pdf(x_t.cpu().numpy(),
                                                          dec_mean_t.detach().cpu().numpy(),
                                                          dec_std_t.detach().cpu().numpy())
                h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), h)
        return mu_rec_chain.mean(axis=3), std_rec_chain.mean(axis=3), score_chain.mean(axis=3)

    def forward(self, X, Y):
        # Iuput X shape : ( seq_len, Batch_size, feature_dim, input_dim)
        # (Batch_size, seq_len, feature_dim)
        X_emb = torch.squeeze(self.embedding(X), dim=-1)

        # ( seq_len, Batch_size,feature_dim)
        Y_emb = torch.squeeze(self.embedding(Y), dim=-1)

        self._reset_variables()
        h = torch.zeros(X.shape[1], self.h_dim).to(self.device)
        h2 = torch.zeros(X.shape[1], self.h_dim).to(self.device)

        for t in range(X.shape[0]):
            x_t = X_emb[t]
            y_t = Y_emb[t]
            h, h2 = self.reucrrence(x_t, y_t, h, h2)

        self.kld_loss, self.nll_loss, self.smooth_loss, self.kld_loss_predict, self.nll_loss_prior, self.nll_loss_predict, self.smooth_loss_prior = self.calc_loss(
            Y)

    def _reset_variables(self):
        # defined variables
        self.Z_mean, self.Z_std = [], []
        self.Xr_mean, self.Xr_std, self.Xr_df = [], [], []
        self.pZ_mean, self.pZ_std = [], []
        self.h_chain = []
        self.h2_chain = []

        self.Xr_mean_prior, self.Xr_std_prior, self.Xr_df_prior = [], [], []

        self.Xr_mean_predict, self.Xr_std_predict = [], []
        self.Z_mean_predict, self.Z_std_predict = [], []

        # defined losses
        self.kld_loss = 0
        self.nll_loss = 0
        self.smooth_loss = 0

        self.kld_loss_prior = 0
        self.nll_loss_prior = 0
        self.smooth_loss_prior = 0

    def calc_loss(self, X):
        # seq_len,batch_size,input_dim*feature_dim)
        X = X.view(X.shape[0], X.shape[1], -1)
        kld_loss = 0
        kld_loss_predict = 0
        nll_loss = 0
        nll_loss_prior = 0
        nll_loss_predict = 0
        smooth_loss = torch.FloatTensor([0]).to(self.device)
        smooth_loss_prior = torch.FloatTensor([0]).to(self.device)
        for t in range(len(self.h_chain)):
            normal_t = Normal(self.Xr_mean[t], self.Xr_std[t])
            normal_t_prior = Normal(
                self.Xr_mean_prior[t], self.Xr_std_prior[t])
            normal_t_predict = Normal(
                self.Xr_mean_predict[t], self.Xr_std_predict[t])
            #dec_studentT = StudentT(dec_df,dec_mean,dec_std)
            kld_loss = kld_loss + self._kld_gauss(self.Z_mean[t], self.Z_std[t],
                                                  self.pZ_mean[t], self.pZ_std[t])

            kld_loss_predict = kld_loss_predict + self._kld_gauss(self.Z_mean[t], self.Z_std[t],
                                                                  self.Z_mean_predict[t], self.Z_std_predict[t])

            nll_loss = nll_loss - normal_t.log_prob(X[t]).sum()
            nll_loss_prior = nll_loss_prior - \
                normal_t_prior.log_prob(X[t]).sum()
            nll_loss_predict = nll_loss_predict - \
                normal_t_predict.log_prob(X[t]).sum()

        for t in range(len(self.h_chain)-1):
            smooth_loss = smooth_loss + self._kld_gauss(self.Xr_mean[t], self.Xr_std[t],
                                                        self.Xr_mean[t+1], self.Xr_std[t+1])
            smooth_loss_prior = smooth_loss_prior + self._kld_gauss(self.Xr_mean_prior[t], self.Xr_std_prior[t],
                                                                    self.Xr_mean_prior[t+1], self.Xr_std_prior[t+1])

        return kld_loss, nll_loss, smooth_loss, kld_loss_predict, nll_loss_prior, nll_loss_predict, smooth_loss_prior

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_gauss(self, mean, std, x):

        return torch.sum(
            0.5*torch.log(torch.Tensor([2*np.pi]).to(self.device))
            + 0.5*torch.log(std**2)
            + 0.5 * (mean-x)**2 / std**2)


def windowing(ts, anomaly_mask=None, window_size=50):
    chunks = []
    anomaly_label = []
    for t in range(ts.shape[-1] - window_size+1):
        if anomaly_mask is not None:
            anomaly_label.append(anomaly_mask[:, t:t+window_size])
        chunks.append(ts[:, t:t+window_size])
    if anomaly_mask is not None:
        return np.stack(chunks).swapaxes(2, 1), np.stack(anomaly_label).swapaxes(2, 1)
    return np.stack(chunks).swapaxes(2, 1)


def windowing_gf(ts, anomaly_mask=None, window_size=50):
    chunks = []
    anomaly_label = []
    for t in range(0, ts.shape[-1] - window_size+1, 20):
        if anomaly_mask is not None:
            anomaly_label.append(anomaly_mask[:, t:t+window_size])
        chunks.append(ts[:, t:t+window_size])
    if anomaly_mask is not None:
        return np.stack(chunks).swapaxes(2, 1), np.stack(anomaly_label).swapaxes(2, 1)
    return np.stack(chunks).swapaxes(2, 1)


def windowing_true(ts, anomaly_mask=None, window_size=50):
    chunks = []
    anomaly_label = []
    for t in range(ts.shape[-1] - window_size+1):
        if anomaly_mask is not None:
            anomaly_label.append(anomaly_mask[:, t:t+window_size])
        chunks.append(ts[:, t:t+window_size])
    if anomaly_mask is not None:
        return np.stack(chunks).swapaxes(2, 1).swapaxes(0, 1), np.stack(anomaly_label).swapaxes(2, 1).swapaxes(0, 1)
    return np.stack(chunks).swapaxes(2, 1).swapaxes(0, 1)
