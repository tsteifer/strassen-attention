# Import packages

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import ModuleList, LayerNorm, Dropout, CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


import numpy as np
import einops
#import tqdm.notebook as tqdm
#from opt_einsum import contract

import math


import random
import time
import copy

#from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
#%matplotlib inline
import plotly.express as px
import plotly.io as pio
#pio.renderers.default='browser'
import plotly.graph_objects as go
from plotly import subplots

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc

import re

# import comet_ml
import itertools

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root=os.path

results=torch.load('TripletSmart3.pth')
print(results['emp_loss'])
#plt.plot(np.array(results['emp_loss'][0]))
#plt.show()
#for o in range(2):
#  plt.plot(results['emp_loss'][o])
#plt.plot(np.arange(0,1000,100),gen_losses,label='nodes=12')
#plt.legend()
#plt.show()