import numpy as np
import os

import deepSI
from deepSI import System_data

from torch import nn
import torch
from scipy.io import loadmat


class base_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(base_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)
    
    
class base_state_net(nn.Module):
    def __init__(self, nx, nu, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(base_state_net, self).__init__()
        from deepSI.utils import simple_res_net
        
        nu = 1 if nu==None else nu
        self.nu = nu
        self.nx = nx

        self.net = simple_res_net(n_in=self.nx+self.nu, \
            n_out=self.nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)
    
# H identity to test first run col 1
class identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input[:,-1]
    
save_dir = r"/home/joost/mscth/data"
save_dir = os.path.join(save_dir,'SilverboxFiles')
out = loadmat(os.path.join(save_dir,'SNLS80mV.mat'))

u, y = out['V1'][0], out['V2'][0]

train, test = System_data(u=u[40000:],y=y[40000:]), System_data(u=u[:40000],y=y[:40000])

fit_sys = deepSI.fit_systems.SS_encoder_general(nx=2, na=50, nb=50, \
                                                e_net=base_encoder_net, e_net_kwargs=dict(n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh),\
                                                f_net=base_state_net, f_net_kwargs=dict(n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh),\
                                                h_net=identity)

train, test = deepSI.datasets.Silverbox()
fit_sys.fit(train, test, epochs=500, loss_kwargs=dict(nf=100))

fit_sys.save_system('H_identity')