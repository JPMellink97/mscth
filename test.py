import numpy as np
from numpy import genfromtxt
import csv
import os

import pysindy as ps

import deepSI

import deepSI
train, test = deepSI.datasets.Silverbox() # Automaticly downloaded (and cashed) the Silverbox system data
                                          # It also splitted the data into two instances of System_data

print(train.shape)


print("Complete")