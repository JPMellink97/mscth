import numpy as np
import pysindy as ps

from scipy.integrate import solve_ivp
from pysindy.utils import linear_damped_SHO, cubic_damped_SHO, van_der_pol, lotka

import matplotlib.pyplot as plt

from numpy import genfromtxt

import deepSI
from deepSI import System_data

import torch
from torch import nn
from torch.nn import functional as F

from tqdm.auto import tqdm

import csv
import os

from scipy.io import loadmat

u_train = None
u_test = None

# data dir and file
save_dir = r"/home/joost/mscth/data"

out = loadmat(os.path.join(save_dir,'Silverbox_full_state_low_error.mat'))
x_data = out['xOptTot']

save_dir = r"/home/joost/mscth/data"
out = loadmat(os.path.join(save_dir,'Silverbox_u_upsampled.mat'))
u = out['u']

dt = 1

idx_train = 800000

# trim since batchting not implemented
x_train = x_data
u_train = u

x_train = np.c_[x_train, u_train]

def normalize(y):
  y_mu = np.mean(y)
  y_std = np.std(y)
  y_norm = (y-y_mu)/y_std

  return y_norm, y_std, y_mu

#generate function set
x_train, x_std, x_mu = normalize(x_train)

degree = 3

# train
train_x = x_train[:-1,:]
Theta = torch.as_tensor(np.array(ps.PolynomialLibrary(degree=degree, include_interaction=False).fit(train_x).transform(train_x))).to(torch.float32)

if u_train is not None:
  target_x = torch.as_tensor(x_train[1:,:-1]).to(torch.float32)
else:
  target_x = torch.as_tensor(x_train[1:,:]).to(torch.float32)

class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__()
        
        self.layer = nn.Linear(n_in, n_out, bias=False)

    def forward(self, x):
        out = self.layer(x)
        return out
    
n_in = Theta.shape[-1]
n_out = target_x.shape[-1]

model = MLP(n_in, n_out)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def SINDyLoss2(X_pred, X_true, Theta, Xi, l):
    params = [x.view(-1) for x in Xi]
    l1_params = torch.cat(params)
    Xi = Xi[0]
    reg_loss = 1/X_true.shape[0] *torch.sum(torch.abs(X_true-torch.matmul(Theta,torch.transpose(Xi,0,1)))**2)
    pred_loss = 1/X_true.shape[0] *torch.sum(torch.abs(X_true-X_pred)**2)
    l1_loss = l*torch.norm(l1_params, 1)
    return reg_loss, pred_loss, l1_loss

epochs = 100

lambda_1 = 1e-3

batch_size = 256
n_batches = Theta.shape[0]//batch_size

losses = []

for epo in tqdm(range(epochs)):
    for batch in range(n_batches):
        Theta_b    = Theta[batch*batch_size:batch*batch_size+batch_size,:]
        target_x_b = target_x[batch*batch_size:batch*batch_size+batch_size,:]

        optimizer.zero_grad()
        output = model(Theta_b)

        # loss
        Xi = [*model.parameters()]

        reg, pred, l1 = SINDyLoss2(output, target_x_b, Theta_b, Xi, lambda_1)
        loss = reg+l1+pred
        losses.append(loss.detach().numpy())

        loss.backward()

        optimizer.step()

    if (epo%(epochs//10)==0 or epo==epochs-1) and epo != 0:
        epo_p = epo if epo != epochs-1 else epo+1
        print("Epoch {} train loss: {}".format(epo_p, loss))
        # print("reg loss {:.4f}, pred loss {:.4f}, l1 loss: {:.4f}".format(reg, pred, l1))


torch.save(model.state_dict(), "test_v2")
