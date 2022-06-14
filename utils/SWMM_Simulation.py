
import networkx as nx
import pandas as pd

from torch_geometric.utils import from_networkx
# from torch_geometric.data import Data



class SWMMSimulation:
    def __init__(self, G, heads_raw_data, runoff_raw_data):
        
        self.G = G
        self.heads_raw_data = heads_raw_data
        self.runoff_raw_data = runoff_raw_data
        
    def get_window(self, steps_ahead, time):
        
        heads = self.heads_raw_data
        runoff = self.runoff_raw_data
        
        h0 = pd.DataFrame(heads.iloc[time, :])
        ro_timeperiod = runoff.iloc[time:time+steps_ahead,:].transpose()
        
        x_matrix = pd.concat([h0, ro_timeperiod]).reset_index(drop =True).transpose().to_numpy()

        
        
        window = from_networkx(self.G)

        

        return h0, ro_timeperiod
        
        
       