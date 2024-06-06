import deepSI
from deepSI.fit_systems import SS_encoder_general
from deepSI.fit_systems.encoders import default_encoder_net, default_state_net, default_output_net

import torch
from torch import nn


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
        #####################################################
        if isinstance(Loss_kwargs.get('encoder_off'), bool) and Loss_kwargs.get('encoder_off'):
            # In the case the encoder is removed from the process
            yh_shape = yhist.shape
            yhist = yfuture[:,0,:]  # select current state as initial state
            yhist = torch.reshape(yhist, yh_shape)
        #####################################################
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            error = nn.functional.mse_loss(y, self.hn(x,u) if self.feedthrough else self.hn(x))
            # TODO: add regurlization
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            x = self.fn(x,u) #advance state.

        return torch.mean(torch.stack(errors))
    

class h_identity(nn.Module):
    """ Output network which is just identity passes all input directly to output.

        Attributes:
            args        : To allow working with SUBNET
            kwargs      : To allow working with SUBNET

        Methods:
            __init__(self, *args, **kwargs):
                Constructor method

            forward(self, input):
                Method is neccesary for working with SUBNET as SUBNET calls forward on this network.
    """
    # TODO: add check for output and input dimension and raise exception when not equal. Currently only works with fully state information and output is considered just to be the state.
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input
    

class e_identity(nn.Module):
    """ Identity encoder class. Given full state information passes directly to f network in simplified case.

        Attributes:
            args        : To allow working with SUBNET
            kwargs      : To allow working with SUBNET

        Methods:
            __init__(self, *args, **kwargs):
                Constructor method

            forward(self, input):
                Method is neccesary for working with SUBNET as SUBNET calls forward on this network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input):
        # SUBNET stack the control input and state information as (u,x) therefore only the last input is passed directly to output.
        output = input[-1]
        output = torch.reshape(output,(output.shape[0], output.shape[-1]))
        return output
    

class simple_Linear(nn.Module):
    """ Neural network linear in the parameters e.g. with a single linear layer. The input to the network is the state information and the input(x,u) transformed by the feature library. This is a seperate class that transforms the input with a specified function library.

        Attributes:
            nx          : The state dimension of data passed through the system.
            nu          : Neccesary to work with SUBNET.
            kwargs      : Used to set dimensions of nu and choosing function library.

        Methods:
            __init__(self, nx, nu, **kwargs):
                Constructor method.

            forward(self, x, u):
                forward used by torch.nn.Module which handels forward passes through the network. x is the state information and u is the control input.
    """
    def __init__(self, nx, nu, **kwargs):
        super(simple_Linear, self).__init__()

        self.nx = nx
        self.nu = kwargs['u']

        self.feature_library = kwargs['feature_library']
        self.nf = self.feature_library.feature_number()
        
        self.layer = nn.Linear(self.nf, nx, bias=False)
        

    def forward(self, x, u):
        # make sure u is column
        u = torch.reshape(u, (u.shape[-1],1))
        x = torch.hstack((x, u))
        Theta = self.feature_library.fit_transform(x)
        out = self.layer(Theta)
        return out