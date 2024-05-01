import pysindy as ps

import deepSI
from deepSI.fit_systems import SS_encoder_general
from deepSI.fit_systems.encoders import default_encoder_net, default_state_net, default_output_net

import torch
from torch import nn

import numpy as np

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
            params = [*self.fn.parameters()]
            weights = [x.view(-1) for x in params][0]
            error += self.gamma*torch.norm(weights, 1)
            ##################################
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            x = self.fn(x,u) #advance state. 
            
        return torch.mean(torch.stack(errors))
    
class identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input[:,-1]
    
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
        x = torch.hstack((x, u.unsqueeze(1)))
        Theta = self.feature_library.fit_transform(x)
        mu = torch.mean(Theta)
        std = torch.std(Theta)
        Theta = (Theta-mu)/std
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

def f3(x):
  return x**3

functions = [f, f2, f3]

poly = feature_library(functions=functions)

nx, nu = 2, 1 # state dimension and inputs
na, nb = 5, 5

f_net = simple_Linear
f_net_kwargs= f_net_kwargs={"feature_library": poly, "u": nu, "nf": 10}

h_net = identity
h_net_kwargs = {}

fit_sys = SS_encoder_general_eq(nx=2, na=50, nb=50, \
                                f_net=f_net, f_net_kwargs=f_net_kwargs,\
                                h_net=identity)

train, test = deepSI.datasets.Silverbox()
train, test = train[:1000], test[:1000]

fit_sys.fit(train, test, epochs=20, batch_size = 2, optimizer_kwargs={"lr": 1e-7}, loss_kwargs=dict(nf=100))

print([*fit_sys.fn.parameters()])