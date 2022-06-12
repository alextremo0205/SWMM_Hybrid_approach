# Imports
import importlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

#ML imports
import torch
import torch.nn as nn
import torch.optim as optim


## Custom modules
import utils.DynamicEmulator as DE
import utils.head_change_utils as utils
from models.mlp_q_interchange import QInterchangeNN, QRunoffNN 
from models.mlp_q_interchange import count_parameters

# Open the file and load the file
yaml_path = 'config_file.yaml'
yaml_data = utils.load_yaml(yaml_path)

# Directories
simulations_path = yaml_data['simulations_path']
inp_path =  yaml_data['inp_path']
rain_path = simulations_path + '\\block_0' + '\\block_0.dat'
heads_path = simulations_path + '\\block_0' + '\\hydraulic_head.pk'
runoff_path = simulations_path + '\\block_0' + '\\runoff.pk'