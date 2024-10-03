import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader

import deepSI
from deepSI import System_data
from deepSI.fit_systems import SS_encoder_general
from deepSI.fit_systems.encoders import default_encoder_net, default_state_net, default_output_net
from deepSI.fit_systems.fit_system import My_Simple_DataLoader, print_array_byte_size, Tictoctimer

from tqdm.auto import tqdm
import time
from copy import deepcopy
import itertools

import numpy as np
import math

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
    

class simple_Linear(torch.nn.Module):
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
        # set feature library
        self.feature_library = kwargs['feature_library']
        self.nf = self.feature_library.feature_number()
        # single layer
        self.layer = nn.Linear(self.nf, nx, bias=False)
        
    def forward(self, x, u):
        # make sure u is column
        u = torch.reshape(u, (u.shape[-1],1))
        x = torch.hstack((x, u))
        # apply basis functions
        Theta = self.feature_library.fit_transform(x)
        if Theta.shape[0] != 1:
            self.batch_rms = torch.sqrt(torch.mean(Theta**2, dim=0))
        # out = theta*coeff
        return self.layer(Theta)


def CustomPruning(model, threshold=0.1):
    # select correct layer
    layer = model.fn.layer
    name = 'weight'

    if type(threshold) is not float:
        threshold = float(threshold)
        
    mask = threshold_mask(layer.weight, threshold=threshold)
    prune.CustomFromMask.apply(layer, name, mask)
    return


def ApplyMask(model, mask):
    layer = model.fn.layer
    name = 'weight'
    prune.CustomFromMask.apply(layer, name, mask)
    return


def threshold_mask(model, threshold=0.1):
    return torch.abs(model.fn.layer.weight) > threshold


def rms_mask(RMS, threshold=0.1, max_prune=1):
    return torch.abs(RMS) > threshold


def cal_mask(RMS, threshold=0.1, max_prune=1):
    # find vals under threshold
    abs_RMS = torch.abs(RMS)
    below_threshold_mask = abs_RMS < threshold
    below_threshold_values = abs_RMS[below_threshold_mask]
    
    if below_threshold_values.numel() == 0:
        return torch.ones_like(abs_RMS)
    
    # Sort vals
    sorted_values, sorted_indices = torch.sort(below_threshold_values)
    zeros = RMS.numel()-torch.count_nonzero(RMS)
    num_to_zero = min(max_prune+zeros, sorted_values.numel())

    # mask
    mask_below_threshold = torch.ones_like(below_threshold_values)
    mask_below_threshold[sorted_indices[:num_to_zero]] = 0  # Set smallest to 0
    
    mask = torch.ones_like(abs_RMS)
    mask[below_threshold_mask] = mask_below_threshold.float()
    return mask != 0


def _loss_function(y_true, y_pred, p="MSE", avg=True):
    if p=="MSE":
        return nn.functional.mse_loss(y_true, y_pred)
    else:
        assert p is not int, "https://pytorch.org/docs/stable/generated/torch.norm.html"
        
        N = np.max(y_true.shape)
        loss = torch.norm(y_pred-y_true,p=p)
        loss = 1/N*loss if avg else loss
        return loss
    

class SS_encoder_general_eq(SS_encoder_general):
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, \
        e_net=default_encoder_net, f_net=simple_Linear, h_net=default_output_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, na_right=0, nb_right=0):

        super().__init__(nx=nx,na=na,nb=nb,\
                         na_right=na_right, nb_right=nb_right,\
                         e_net=e_net, e_net_kwargs=e_net_kwargs,\
                         f_net=f_net, f_net_kwargs=f_net_kwargs,\
                         h_net=h_net, h_net_kwargs=h_net_kwargs,\
                         feedthrough=feedthrough)
        

    def loss(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
        #####################################################  
            # pnorm loss
            error = self._loss_function(y, self.hn(x,u) if self.feedthrough else self.hn(x), **Loss_kwargs)
        #####################################################
        
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            x = self.fn(x,u) #advance state.
        
        return torch.mean(torch.stack(errors))
    
    
    def _loss_function(self, y_true, y_pred, **Loss_kwargs):
        # selecting the wanted loss function base MSE
        # Lp norm availlable

        # averaged by default
        factor = 1/np.max(y_true.shape)
        if 'avg' in Loss_kwargs and not Loss_kwargs.get('avg'):
            factor = 1

        if "p" in Loss_kwargs:
            p = Loss_kwargs.get('p')
            assert p is not int, "https://pytorch.org/docs/stable/generated/torch.norm.html"
            return factor*torch.norm(y_pred-y_true,p=p)
        else:
            return nn.functional.mse_loss(y_true, y_pred)
    
    def _apply_mask(self, mask):
        layer = self.fn.layer
        name = 'weight'
        prune.CustomFromMask.apply(layer, name, mask)
        return
        
    
    def fit(self, train_sys_data, val_sys_data, epochs=30, batch_size=256, loss_kwargs={}, early_stopping=True, earlystop_epoch=15,\
            auto_fit_norm=True, validation_measure='sim-NRMS', optimizer_kwargs={}, concurrent_val=False, cuda=False, \
            timeout=None, verbose=1, sqrt_train=True, num_workers_data_loader=0, print_full_time_profile=False, scheduler_kwargs={}, load_best=False):
        '''The batch optimization method with parallel validation, 

        Parameters
        ----------
        train_sys_data : System_data or System_data_list
            The system data to be fitted
        val_sys_data : System_data or System_data_list
            The validation system data after each used after each epoch for early stopping. Use the keyword argument validation_measure to specify which measure should be used. 
        epochs : int
        batch_size : int
        loss_kwargs : dict
            The Keyword Arguments to be passed to the self.make_training_data and self.loss of the current fit_system.
        auto_fit_norm : boole
            If true will use self.norm.fit(train_sys_data) which will fit it element wise. 
        validation_measure : str
            Specify which measure should be used for validation, e.g. 'sim-RMS', '10-step-last-RMS', 'sim-NRMS_sys_norm', ect. See self.cal_validation_error for details.
        optimizer_kwargs : dict
            The Keyword Arguments to be passed on to init_optimizer. notes; init_optimizer['optimizer'] is the optimization function used (default torch.Adam)
            and optimizer_kwargs['parameters_optimizer_kwargs'] the learning rates and such for the different elements of the models. see https://pytorch.org/docs/stable/optim.html
        concurrent_val : boole
            If set to true a subprocess will be started which concurrently evaluates the validation method selected.
            Warning: if concurrent_val is set than "if __name__=='__main__'" or import from a file if using self defined method or networks.
        cuda : bool
            if cuda will be used (often slower than not using it, be aware)
        timeout : None or number
            Alternative to epochs to run until a set amount of time has past. 
        verbose : int
            Set to 0 for a silent run
        sqrt_train : boole
            will sqrt the loss while printing
        num_workers_data_loader : int
            see https://pytorch.org/docs/stable/data.html
        print_full_time_profile : boole
            will print the full time profile, useful for debugging and basic process optimization. 
        scheduler_kwargs : dict
            learning rate scheduals are a work in progress.
        
        Notes
        -----
        This method implements a batch optimization method in the following way; each epoch the training data is scrambled and batched where each batch
        is passed to the self.loss method and utilized to optimize the parameters. After each epoch the systems is validated using the evaluation of a 
        simulation or a validation split and a checkpoint will be crated if a new lowest validation loss has been achieved. (or concurrently if concurrent_val=True)
        After training (which can be stopped at any moment using a KeyboardInterrupt) the system is loaded with the lowest validation loss. 

        The default checkpoint location is "C:/Users/USER/AppData/Local/deepSI/checkpoints" for windows and ~/.deepSI/checkpoints/ for unix like.
        These can be loaded manually using sys.load_checkpoint("_best") or "_last". (For this to work the sys.unique_code needs to be set to the correct string)
        '''
        def validation(train_loss=None, time_elapsed_total=None):
            self.eval(); self.cpu()
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
            self.Loss_val.append(Loss_val)
            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                self.checkpoint_save_system()
            if cuda: 
                self.cuda()
            self.train()
            return Loss_val
        
        ########## Initialization ##########
        if self.init_model_done==False:
            if verbose: print('Initilizing the model and optimizer')
            device = 'cuda' if cuda else 'cpu'
            optimizer_kwargs = deepcopy(optimizer_kwargs)
            parameters_optimizer_kwargs = optimizer_kwargs.get('parameters_optimizer_kwargs',{})
            if parameters_optimizer_kwargs:
                del optimizer_kwargs['parameters_optimizer_kwargs']
            self.init_model(sys_data=train_sys_data, device=device, auto_fit_norm=auto_fit_norm, optimizer_kwargs=optimizer_kwargs,\
                    parameters_optimizer_kwargs=parameters_optimizer_kwargs, scheduler_kwargs=scheduler_kwargs)
        else:
            if verbose: print('Model already initilized (init_model_done=True), skipping initilizing of the model, the norm and the creation of the optimizer')
            self._check_and_refresh_optimizer_if_needed() 


        if self.scheduler==False and verbose:
            print('!!!! Your might be continuing from a save which had scheduler but which was removed during saving... check this !!!!!!')
        
        self.dt = train_sys_data.dt
        if cuda: 
            self.cuda()
        self.train()

        self.epoch_counter = 0 if len(self.epoch_id)==0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
        extra_t            = 0 if len(self.time)    ==0 else self.time[-1] #correct timer after restart

        ########## Getting the data ##########
        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        if not isinstance(data_train, Dataset) and verbose: print_array_byte_size(sum([d.nbytes for d in data_train]))

        #### transforming it back to a list to be able to append. ########
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        #### init monitoring values ########
        Loss_acc_val, N_batch_acc_val, val_counter, best_epoch, batch_id_start = 0, 0, 0, 0, self.batch_counter #to print the frequency of the validation step.
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        if verbose>0: 
            print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')
        
        ### convert to dataset ###
        if isinstance(data_train, Dataset):
            persistent_workers = False if num_workers_data_loader==0 else True
            data_train_loader = DataLoader(data_train, batch_size=batch_size, drop_last=True, shuffle=True, \
                                   num_workers=num_workers_data_loader, persistent_workers=persistent_workers)
        else: #add my basic DataLoader
            data_train_loader = My_Simple_DataLoader(data_train, batch_size=batch_size) #is quite a bit faster for low data situations

        if concurrent_val:
            self.remote_start(val_sys_data, validation_measure)
            self.remote_send(float('nan'), extra_t)
        else: #start with the initial validation 
            validation(train_loss=float('nan'), time_elapsed_total=extra_t) #also sets current model to cuda
            if verbose: 
                print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        #####################################
        #####################################
        # saving params
        epo_param0 = self.fn.layer.weight.data.cpu().flatten().numpy()
        
        if isinstance(loss_kwargs.get('save_params'), bool) and loss_kwargs.get('save_params'):
            save_params = loss_kwargs.get('save_params')

            if not hasattr(self, 'coefficients'):
                setattr(self, 'coefficients', [epo_param0])

        else:
            save_params = False


        # regularization setup
        mode = False
        if isinstance(loss_kwargs.get('gamma'), (float,int)):
            gamma = loss_kwargs.get('gamma')
            gamma_vec = gamma*np.ones(epochs)
            if isinstance(loss_kwargs.get('mode'), str):
                mode = loss_kwargs.get('mode')
                if mode=="decaying":
                    # regularization where gamma goes to zero linear
                    gamma_vec = np.linspace(gamma, 0, num=epochs)
                elif mode=="MartiusLampert16":
                    # regularization scheme with 3 phases
                    # phase 1: no regularization
                    # phase 2: gamma regularization
                    # phase 3: no regularization and small weights frozen
                    if "T_idx" in loss_kwargs:
                        T_idx = loss_kwargs.get('T_idx')
                        assert len(T_idx)==2, "Only give 2 indexes"
                        assert T_idx[0]<T_idx[1], "T_idx[0]>T_idx[1]"
                        T12, T23 = T_idx[0], T_idx[1]
                    else:
                        T12, T23 = 0.25, 19/20
                    T12_idx = round(T12*epochs)
                    T23_idx = epochs-round(T23*epochs)
                    T2_idx = epochs-T12_idx-T23_idx
                    gamma_vec = np.r_[np.zeros(T12_idx), gamma*np.ones(T2_idx), np.zeros(T23_idx)]
        else:
            gamma_vec = np.zeros(epochs)

        # pruning setup
        if isinstance(loss_kwargs.get('pruning'), bool) and loss_kwargs.get('pruning'):
            pruning = True
            if isinstance(loss_kwargs.get('pruning_mode'), str):
                pruning_mode = loss_kwargs.get('pruning_mode')
            else:
                pruning_mode = "RMS"
            threshold = loss_kwargs.get('threshold') if "threshold" in loss_kwargs else 1e-1
            epo_idx = int(loss_kwargs.get('epo_idx')) if "epo_idx" in loss_kwargs else 1
        else:
            pruning = False

        mask = torch.ones(self.fn.layer.weight.data.shape)==1

        best_nonnan = 0
        #####################################
        #####################################

        try:
            t = Tictoctimer()
            start_t = time.time() #time keeping
            epochsrange = range(epochs) if timeout is None else itertools.count(start=0)
            if timeout is not None and verbose>0: 
                print(f'Starting indefinite training until {timeout} seconds have passed due to provided timeout')

            for epoch in (tqdm(epochsrange) if verbose>0 else epochsrange):
                bestfit_old = self.bestfit #to check if a new lowest validation loss has been achieved
                Loss_acc_epoch = 0.
                t.start()
                t.tic('data get')
                for train_batch in data_train_loader:
                    if cuda:
                        train_batch = [b.cuda() for b in train_batch]
                    t.toc('data get')
                    def closure(backward=True):
                        # called every update
                        t.toc('optimizer start')
                        t.tic('loss')
                        Loss = self.loss(*train_batch, **loss_kwargs)
                        ##############################################
                        ##############################################
                        # regurlarization
                        fn_W = self.fn.layer.weight.data.flatten()
                        Loss += gamma_vec[epoch]*torch.norm(fn_W, 1)
                        # phase 3 freeze weights
                        if mode=="MartiusLampert16" and epoch==T23_idx:
                            mask = threshold_mask(self, threshold=0.1)
                            self._apply_mask(mask)
                            def freeze_phase2_weight(grad):
                                return grad*mask
                            self.fn.layer.weight.register_hook(freeze_phase2_weight)
                        ##############################################
                        ##############################################
                        t.toc('loss')
                        if backward:
                            t.tic('zero_grad')
                            self.optimizer.zero_grad()
                            t.toc('zero_grad')
                            t.tic('backward')
                            Loss.backward()
                            t.toc('backward')
                        t.tic('stepping')
                        return Loss
                    
                    t.tic('optimizer start')
                    training_loss = self.optimizer.step(closure).item()
                    t.toc('stepping')
                    if self.scheduler:
                        t.tic('scheduler')
                        self.scheduler.step()
                        t.tic('scheduler')
                    Loss_acc_val += training_loss
                    Loss_acc_epoch += training_loss
                    N_batch_acc_val += 1
                    self.batch_counter += 1
                    self.epoch_counter += 1/N_batch_updates_per_epoch

                    ##############################################
                    ##############################################
                    # saving params
                    if save_params:
                        epo_param = self.fn.layer.weight.data.cpu().flatten().numpy()
                        self.coefficients.append(epo_param.copy())
                    ##############################################
                    ##############################################

                    t.tic('val')
                    if concurrent_val and self.remote_recv(): ####### validation #######
                        self.remote_send(Loss_acc_val/N_batch_acc_val, time.time()-start_t+extra_t)
                        Loss_acc_val, N_batch_acc_val, val_counter = 0., 0, val_counter + 1
                    t.toc('val')
                    t.tic('data get')
                t.toc('data get')
                ##############################################
                ##############################################
                # pruning
                if pruning and epoch>0 and (epoch%epo_idx==0):
                    if pruning_mode == "RMS":
                        coeff_batch_rms = torch.mean(torch.tensor(self.coefficients).T[:,-round(epo_idx*N_batch_updates_per_epoch):],axis=1).reshape(self.fn.layer.weight.data.shape)*self.fn.batch_rms
                    elif pruning_mode == "th":
                        coeff_batch_rms = torch.mean(torch.tensor(self.coefficients).T[:,-round(epo_idx*N_batch_updates_per_epoch):],axis=1).reshape(self.fn.layer.weight.data.shape)
                    
                    # throw away maximum of 10% of coefficient at any pruning step
                    mask = cal_mask(coeff_batch_rms, threshold=threshold, max_prune=math.ceil(self.fn.layer.weight.data.shape[-1]*0.1))
                    self._apply_mask(mask)
                    print(mask)

                def freeze_pruned_weight(grad):
                    return grad*mask
                self.fn.layer.weight.register_hook(freeze_pruned_weight)
                ##############################################
                ##############################################

                ########## end of epoch clean up ##########
                train_loss_epoch = Loss_acc_epoch/N_batch_updates_per_epoch
                if np.isnan(train_loss_epoch):
                    if verbose>0: print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at epoch {epoch}, breaking from loop &&&&&&&&&&')
                    break

                t.tic('val')
                if not concurrent_val:
                    validation(train_loss=train_loss_epoch, \
                               time_elapsed_total=time.time()-start_t+extra_t) #updates bestfit and goes back to cpu and back
                t.toc('val')
                t.pause()

                ######### Printing Routine ##########
                if verbose>0:
                    time_elapsed = time.time() - start_t
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {validation_measure} = {self.bestfit}')
                        best_epoch = epoch+1
                        ############################################################
                        ############################################################
                        best_nonnan = best_epoch
                        ############################################################
                        ############################################################
                        
                    if concurrent_val: #if concurrent val than print validation freq
                        val_feq = val_counter/(epoch+1)
                        valfeqstr = f', {val_feq:4.3} vals/epoch' if (val_feq>1 or val_feq==0) else f', {1/val_feq:4.3} epochs/val'
                    else: #else print validation time use
                        valfeqstr = f''
                    trainstr = f'sqrt loss {train_loss_epoch**0.5:7.4}' if sqrt_train and train_loss_epoch>=0 else f'loss {train_loss_epoch:7.4}'
                    Loss_val_now = self.Loss_val[-1] if len(self.Loss_val)!=0 else float('nan')
                    Loss_str = f'Epoch {epoch+1:4}, {trainstr}, Val {validation_measure} {Loss_val_now:6.4}'
                    loss_time = (t.acc_times['loss'] + t.acc_times['optimizer start'] + t.acc_times['zero_grad'] + t.acc_times['backward'] + t.acc_times['stepping'])  /t.time_elapsed
                    time_str = f'Time Loss: {loss_time:.1%}, data: {t.acc_times["data get"]/t.time_elapsed:.1%}, val: {t.acc_times["val"]/t.time_elapsed:.1%}{valfeqstr}'
                    self.batch_feq = (self.batch_counter - batch_id_start)/(time.time() - start_t)
                    batch_str = (f'{self.batch_feq:4.1f} batches/sec' if (self.batch_feq>1 or self.batch_feq==0) else f'{1/self.batch_feq:4.1f} sec/batch')
                    print(f'{Loss_str}, {time_str}, {batch_str}')
                    if print_full_time_profile:
                        print('Time profile:',t.percent())

                ####### Timeout Breaking ##########
                if timeout is not None:
                    if time.time() >= start_t+timeout:
                        break
                ############################################################
                ############################################################
                if np.isnan(self.Loss_val[-1]):
                    best_nonnan = best_nonnan +1

                if early_stopping and (epoch-best_nonnan+1==earlystop_epoch):
                    break
                ############################################################
                ############################################################
                
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')

        self.train(); self.cpu()
        del data_train_loader

        ####### end of training concurrent things #####
        if concurrent_val:
            if verbose: print(f'Waiting for started validation process to finish and one last validation... (receiving = {self.remote.receiving})',end='')
            if self.remote_recv(wait=True):
                if verbose: print('Recv done... ',end='')
                if N_batch_acc_val>0:
                    self.remote_send(Loss_acc_val/N_batch_acc_val, time.time()-start_t+extra_t)
                    self.remote_recv(wait=True)
            self.remote_close()
            if verbose: print('Done!')

        
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        ####################################################
        ####################################################
        if load_best:
            try:
                self.checkpoint_load_system(name='_best')
            except FileNotFoundError:
                print('no best checkpoint found keeping last')
            if verbose: 
                print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_epoch} (epoch_id={self.epoch_id[-1] if len(self.epoch_id)>0 else 0:.2f})')
        ####################################################
        ####################################################