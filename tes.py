import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

import torch
import torch.nn as nn 
from torch.optim import Adam
from models.CA_wyh import CA_base_wyh
from analysis import alpha_plot
from utils import CHARAS_LIST
from models.CA_wyh import VAE_wyh,CA3_wyh
# model=VAE_wyh(hidden_size=5,lr=0.001,omit_char=[],device='cuda')
model=CA3_wyh(hidden_size=5,lr=0.001,omit_char=[],device='cuda')
alpha_plot(model, type='inference', save_dir='imgs_vae')