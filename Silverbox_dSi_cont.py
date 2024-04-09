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

fs = 610.35
dt = 1/fs

train.dt = dt
test.dt = dt

tau = train.dt/0.03

fit_sys_ss_enc = deepSI.fit_systems.SS_encoder_deriv_general(nx=2, na=10, nb=10, tau=tau)
train_enc, val_enc = train.train_test_split(split_fraction=0.2)

#Start fitting
fit_sys_ss_enc.fit(train_sys_data=train_enc, val_sys_data=val_enc, \
                   epochs=100, batch_size=256, loss_kwargs={'nf':100}) #nf is T in paper

print(fit_sys_ss_enc.fn) #state network x_t+1 = f([x,u])
print(fit_sys_ss_enc.hn) #output network y_t = h(x_t)
print(fit_sys_ss_enc.encoder) #encoder network x_t = psi([upast, ypast])

# test_sim_enc = fit_sys_ss_enc.apply_experiment(test, save_state=True)
# np.savetxt("ct_subnet_states.csv", test_sim_enc.x, delimiter=",")

# fit_sys_IO.save_system('IO-sys')
# fit_sys_SS.save_system('SS-sys')
# fit_sys_ss_enc.save_system('enc-sys')

plt.plot(test.y)
plt.plot(test.y - test_sim_enc.y)
plt.title(f'test set simulation SS encoder, NRMS = {test_sim_enc.NRMS(test):.2%}')
plt.show()
plt.savefig('test_res_sim.png')
plt.close()

# train_sim_enc = fit_sys_ss_enc.apply_experiment(train)

# plt.plot(train.y)
# plt.plot(train.y - train_sim_enc.y)
# plt.title(f'train set simulation SS encoder, NRMS = {train_sim_enc.NRMS(train):.2%}')
# plt.savefig('train_res_sim.png')
# plt.show()
# plt.close()