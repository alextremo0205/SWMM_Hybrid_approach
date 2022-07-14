# Imports
import os
import pickle
import random
import importlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ipywidgets import FloatProgress
from sklearn.model_selection import train_test_split

#ML imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

## Custom modules
from models.GNNModel import GNNModel
import visualization.Visualization as vis
from libraries.Normalizer import Normalizer
from libraries.trainingPyTorch import train
from libraries.SWMM_Simulation import SWMMSimulation


import libraries.utils as utils

torch.set_printoptions(precision=4, sci_mode = False)
utils.print_current_git_branch()