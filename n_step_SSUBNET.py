import pysindy as ps

import deepSI
from deepSI.fit_systems import SS_encoder_general
from deepSI.fit_systems.encoders import default_encoder_net, default_state_net, default_output_net

import torch
from torch import nn

import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from scipy.io import loadmat
import os

import deepSI
from deepSI import System_data

from utils import load_data

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


# load data
test_samples = 100000
train_samples = 500000
x_data, u_data, y_data, th_data = load_data(pc=0, set="SILVERSIN")
train, test = System_data(u=u_data[:train_samples,0],y=x_data[:train_samples,:]), System_data(u=u_data[-test_samples:],y=x_data[-test_samples:,:])

class SS_encoder_general_eq(SS_encoder_general):
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, \
        e_net=default_encoder_net, f_net=default_state_net, h_net=default_output_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, na_right=0, nb_right=0, \
        gamma=1e-4):

        super(SS_encoder_general_eq, self).__init__()
        self.nx, self.na, self.nb = nx, na, nb
        self.k0 = max(self.na,self.nb)
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs

        self.f_net = f_net
        self.f_net_kwargs = f_net_kwargs

        self.h_net = h_net
        self.h_net_kwargs = h_net_kwargs

        self.feedthrough = feedthrough
        self.na_right = na_right
        self.nb_right = nb_right
        ######################################
        # args added for feature transform and
        # regurlarization
        self.gamma = gamma
        ######################################

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=(self.nb+nb_right), nu=nu, na=(self.na+na_right), ny=ny, nx=self.nx, **self.e_net_kwargs)
        ######################################
        ###### change fn intialization #######
        self.fn     =      self.f_net(nx=self.nx, nu=nu, **self.f_net_kwargs)
        ######################################
        if self.feedthrough:
            self.hn =      self.h_net(nx=self.nx, ny=ny, nu=nu,                     **self.h_net_kwargs) 
        else:
            self.hn =      self.h_net(nx=self.nx, ny=ny,                            **self.h_net_kwargs) 

    def loss(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            error = nn.functional.mse_loss(y, self.hn(x,u) if self.feedthrough else self.hn(x))
            ##################################
            ## add penalty to weights in fn ##
            # params = [*self.fn.parameters()]
            # weights = [x.view(-1) for x in params][0]
            # error += self.gamma*torch.norm(weights, 1)
            ##################################
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            x = self.fn(x,u) #advance state. 
            
        return torch.mean(torch.stack(errors))
    
class h_identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input
    
class e_identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input):
        output = input[-1]
        output = torch.reshape(output,(output.shape[0], output.shape[-1]))
    
        return output
    
class simple_Linear(torch.nn.Module):
    def __init__(self, nx, nu, **kwargs):
        super(simple_Linear, self).__init__()

        self.nx = nx
        self.nu = kwargs['u']

        self.feature_library = kwargs['feature_library']
        test_sample = torch.rand(1,self.nx+self.nu, requires_grad=True)
        self.nf = (self.feature_library.fit_transform(test_sample)).shape[1]
        
        self.layer = nn.Linear(self.nf, nx, bias=False)
        

    def forward(self, x, u):
        # make sure u is column
        u = torch.reshape(u, (u.shape[-1],1))
        x = torch.hstack((x, u))
        Theta = self.feature_library.fit_transform(x)
        out = self.layer(Theta)
        return out


class feature_library():
    def __init__(
            self,
            functions,
            interaction_only=True
    ):
        self.functions = functions
        self.interaction_only = interaction_only

    def fit_transform(self, X):
        # off set
        out_feature = ((X[:,0])**0).unsqueeze(1)
        if self.interaction_only:
            for f in self.functions:
                out_feature = torch.hstack((out_feature, f(X)))
            return out_feature


def f(x):
  return x

def f2(x):
  return x**2

def sin(x):
  return torch.sin(x)

def f3(x):
  return x**3

# function library
functions = [f, sin]
poly = feature_library(functions=functions)

nx, nu = 2, 1 # state dimension and inputs
na, nb = 1, 0

f_net = simple_Linear
f_net_kwargs = {"feature_library": poly, "u": nu, "nf": 7}

h_net = h_identity
h_net_kwargs = {}

fit_sys = SS_encoder_general_eq(nx=nx, na=na, nb=nb, \
                                f_net=f_net, f_net_kwargs=f_net_kwargs,\
                                e_net=e_identity, e_net_kwargs=f_net_kwargs,\
                                h_net=h_net)

# fit auto_norm False
fit_sys.fit(train, test, epochs=1, batch_size = 8192, optimizer_kwargs={"lr": 1e-3}, loss_kwargs=dict(nf=100), auto_fit_norm=False)

# process results
test_sim_enc = fit_sys.apply_experiment(test)

def NRMS(y_pred, y_true):
    RMS = np.sqrt(np.mean((y_pred-y_true)**2))
    return RMS/np.std(y_true)

plt.plot(test.y[:,0])
plt.plot(test.y[:,0]-test_sim_enc.y[:,0])
plt.title(f'test set simulation SS encoder, NRMS = {NRMS(test_sim_enc.y[:,0],test.y[:,0]):.2%}')
plt.show()
plt.savefig('res_sim_x1.png')
plt.close()

plt.plot(test.y[:,1])
plt.plot(test.y[:,1]-test_sim_enc.y[:,1])
plt.title(f'test set simulation SS encoder, NRMS = {NRMS(test_sim_enc.y[:,1],test.y[:,1]):.2%}')
plt.show()
plt.savefig('res_sim_x2.png')
plt.close()

true = np.array([[0, 1, 1, 0, 0, 0, 0],[0, -0.1, 0.5, 0.1, -0.2, 0, 0]])
found = [*fit_sys.fn.parameters()][0].detach().numpy()


fig, (ax1,ax2) = plt.subplots(2, 1)

x_labels = ["1","x0[k]","x1[k]","u[k]","sin(x0[k])","sin([x1[k]])","sin(u[k])"]

data1 = np.vstack((true[0,:],found[0,:]))
data2 = np.vstack((true[1,:],found[1,:]))
cmap_white = LinearSegmentedColormap.from_list("white", [(1, 1, 1), (1, 1, 1)])

im = ax1.imshow(data1, cmap=cmap_white)

ax1.set_xticks(np.arange(data1.shape[1]), labels=x_labels, rotation=25)
ax1.set_yticks(np.arange(data1.shape[0]), labels=["True", "Found"])

for i in range(data1.shape[0]):
    for j in range(data1.shape[1]):
        text = ax1.text(j, i, round(data1[i, j],3),
                       ha="center", va="center", color="k")
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1)
        ax1.add_patch(rect)

ax1.patch.set_linewidth(2.0)        
ax1.patch.set_edgecolor('black')

# second
im = ax2.imshow(data2, cmap=cmap_white)

ax2.set_xticks(np.arange(data2.shape[1]), labels=x_labels, rotation=25)
ax2.set_yticks(np.arange(data2.shape[0]), labels=["True", "Found"])

for i in range(data2.shape[0]):
    for j in range(data2.shape[1]):
        text = ax2.text(j, i, round(data2[i, j], 3),
                       ha="center", va="center", color="k")
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1)
        ax2.add_patch(rect)

ax2.patch.set_linewidth(2.0)        
ax2.patch.set_edgecolor('black')

plt.show()
plt.savefig('coeffs.png')
plt.close()