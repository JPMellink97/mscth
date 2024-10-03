import pysindy as ps

import deepSI
from deepSI import System_data
from deepSI.fit_systems import SS_encoder_general
from deepSI.fit_systems.encoders import default_encoder_net, default_state_net, default_output_net
from deepSI.fit_systems.fit_system import My_Simple_DataLoader, print_array_byte_size, Tictoctimer
import torch
from torch import nn

import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from scipy.io import loadmat
import scipy.linalg as lin
import os

from utils import load_data
from data_processing import add_gaussian_noise

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import time
from copy import deepcopy
import itertools

import re

from feature_construction import feature_library
import polynomial as p
import fourier as f

from SI_SUBNET import SS_encoder_general_eq, h_identity, e_identity, simple_Linear
from SI_metrics import plot_coeff_grid, display_equation

import random
import string

from scipy.io import loadmat, savemat
import json
import pickle

from IPython.display import clear_output


def run_SISUBNET(nx, na, nb, na_right, \
                 f_net_kwargs, e_net, e_net_kwargs, h_net, h_net_kwargs,\
                 arrow, train, val, test, data_set_name,\
                 nf, gamma, mode, T_idx, pruning, threshold, epo_idx,\
                 batch_size, pre_train=False, no_reg_epochs=100, nf_epochs=100,\
                 save_data=True, set_name=None, verbose=True):
      
      print("start") 
      
      # SI-SUBNET initialization
      fit_sys = SS_encoder_general_eq(nx=nx, na=na, nb=nb, na_right=na_right,\
                                 f_net_kwargs=f_net_kwargs,\
                                 e_net=e_net, e_net_kwargs=e_net_kwargs,\
                                 h_net=h_net, h_net_kwargs=h_net_kwargs)
      
      # train 1 step ahead first always 50 epoch
      if pre_train:
         loss_kwargs= dict(nf=2, save_params=True, gamma=0, mode=None, T_idx=None, pruning=False, threshold=0, epo_idx=99999)
         fit_sys.fit(train, val, epochs=100, batch_size=batch_size, optimizer_kwargs={"lr": 1e-3}, loss_kwargs=loss_kwargs, auto_fit_norm=False, early_stopping=False, load_best=False, verbose=verbose)

      # train nf step ahead with reg and pruning
      loss_kwargs= dict(nf=nf+1, save_params=True, gamma=gamma, mode=mode, T_idx=T_idx, pruning=pruning, threshold=threshold, epo_idx=epo_idx)
      fit_sys.fit(train, val, epochs=nf_epochs, batch_size = batch_size, optimizer_kwargs={"lr": 1e-3}, loss_kwargs=loss_kwargs, auto_fit_norm=False, early_stopping=False, load_best=False, verbose=verbose)

      # train nf step ahead with no reg or pruning
      loss_kwargs= dict(nf=nf+1, save_params=True, gamma=0, mode=None, T_idx=None, pruning=False, threshold=0, epo_idx=99999)
      fit_sys.fit(train, val, epochs=no_reg_epochs, batch_size = batch_size, optimizer_kwargs={"lr": 1e-3}, loss_kwargs=loss_kwargs, auto_fit_norm=False, early_stopping=False, load_best=False, verbose=verbose)

      Loss_val = fit_sys.Loss_val
      epoch_id = fit_sys.epoch_id
      Loss_train = fit_sys.Loss_train**0.5

      # generating results
      arrow_sim_enc = fit_sys.apply_experiment(arrow)
      print("arrow test complete!")
      NRMS_arrow = arrow_sim_enc.NRMS(arrow)
      test_sim_enc  = fit_sys.apply_experiment(test)
      print("test test complete!")
      NRMS_test = test_sim_enc.NRMS(test)

      coeff = np.array(fit_sys.coefficients).T
      eq_res = display_equation(coeff[:,-1:], f_net_kwargs["feature_library"].feature_names,threshold=0.000001, precision=6, verbose=False)

      # loading best
      fit_sys.checkpoint_load_system(name='_best')
      best_arrow_sim_enc = fit_sys.apply_experiment(arrow)
      print("arrow test complete!")
      best_NRMS_arrow = best_arrow_sim_enc.NRMS(arrow)
      best_test_sim_enc  = fit_sys.apply_experiment(test)
      print("test test complete!")
      best_NRMS_test = best_test_sim_enc.NRMS(test)

      best_coeff = np.array(fit_sys.coefficients).T
      best_eq_res = display_equation(coeff[:,-1:], f_net_kwargs["feature_library"].feature_names,threshold=0.000001, precision=6, verbose=False)

      # saving data
      if save_data:
         if set_name is None:
            set_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

         data_dict = {"nx":nx, "na":na, "nb":nb, "na_right":na_right, \
                        "feature_names":f_net_kwargs["feature_library"].feature_names, "e_net":e_net, "e_net_kwargs":e_net_kwargs, "h_net":h_net, "h_net_kwargs":h_net_kwargs,\
                        "data_set_name":data_set_name,\
                        "nf":nf, "gamma":gamma, "mode":mode, "T_idx":T_idx, "pruning":pruning, "threshold":threshold, "epo_idx":epo_idx,\
                        "batch_size":batch_size, "pre_train":pre_train, "no_reg_epochs":no_reg_epochs, "nf_epochs":nf_epochs,\
                        "epoch_id": epoch_id, "Loss_val":Loss_val, "Loss_train":Loss_train,\
                        "arrow_sim_enc":arrow_sim_enc.y, "NRMS_arrow":NRMS_arrow, "test_sim_enc":test_sim_enc.y, "NRMS_test":NRMS_test,\
                        "coeff":coeff, "eq_res":eq_res, "best_arrow_sim_enc":best_arrow_sim_enc.y, "best_NRMS_arrow":best_NRMS_arrow,\
                        "best_test_sim_enc":best_test_sim_enc.y, "best_NRMS_test":best_NRMS_test,\
                        "best_coeff":best_coeff, "best_eq_res":best_eq_res, "best_weights":fit_sys.fn.layer.weight.data}# 
         
         with open("result_runs\\"+set_name+".pkl", "wb") as file:
            pickle.dump(data_dict, file)

      print("done") 
      clear_output(wait=True)
      return    
