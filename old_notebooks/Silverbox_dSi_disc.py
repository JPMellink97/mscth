import numpy as np
from numpy import genfromtxt
import csv
import os

import deepSI
from deepSI import System_data

import matplotlib.pyplot as plt

DATA_PATH = r"data//"

WIENER = "WienerHammerBenchmark"
SILVER = "SNLS80mV"
# change data set
DATA = SILVER
CSV = ".csv"

PATH = os.path.join(DATA_PATH, DATA+CSV)

# load data
data = genfromtxt(PATH, delimiter=",")

# drop nan's may need to be adjusted between data sets
mask = ~np.isnan(data)
data = data[mask[:,0],:]
data = data[:,:-1]

system_data = System_data(u=data[:,0],y=data[:,1])
train, test = system_data[40000:], system_data[:40000]

fit_sys_ss_enc = deepSI.fit_systems.SS_encoder(nx=2, na=50, nb=50) #state dimention = 6, past outputs = 3, past inputs = 3.
train_enc, val_enc = train.train_test_split(split_fraction=0.2)

#Start fitting
fit_sys_ss_enc.fit(train_sys_data=train_enc, val_sys_data=val_enc, \
                   epochs=500, batch_size=256, loss_kwargs={'nf':100}) #nf is T in paper

print(fit_sys_ss_enc.fn) #state network x_t+1 = f([x,u])
print(fit_sys_ss_enc.hn) #output network y_t = h(x_t)
print(fit_sys_ss_enc.encoder) #encoder network x_t = psi([upast, ypast])


fit_sys_ss_enc.save_system('enc-sys')
