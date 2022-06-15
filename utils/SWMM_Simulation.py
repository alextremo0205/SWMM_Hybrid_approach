
import networkx as nx
import pandas as pd

from torch_geometric.utils import from_networkx

class SWMMSimulation:
    def __init__(self, G, heads_raw_data, runoff_raw_data):
        
        self.G = G
        self.heads_raw_data = heads_raw_data
        self.runoff_raw_data = runoff_raw_data
        self.simulation_length = len(self.heads_raw_data)
        
    def get_window(self, steps_ahead, time):
        
        self.checkOutOfBounds(steps_ahead, time)
        
        h0 =                self.get_h0_for_window(time)
        ro_timeperiod =     self.get_ro_for_window(time, steps_ahead)
        ht_timeperiod =     self.get_ht_for_window(time, steps_ahead)
        
        input_features_dict=    self.get_features_dictionary(h0, ro_timeperiod)
        output_features_dict=   self.get_features_dictionary(ht_timeperiod)
        
        G_for_window =      self.G
        nx.set_node_attributes(G_for_window, input_features_dict, name = 'x')
        nx.set_node_attributes(G_for_window, output_features_dict,name = 'y')
        
        window = from_networkx(G_for_window)

        return window

    def checkOutOfBounds(self, steps_ahead, time):
        max_allowable_time = self.simulation_length - steps_ahead
        if time>max_allowable_time:
            raise ValueError("The window is beyond the time series length")

    def get_h0_for_window(self, time):
        return pd.DataFrame(self.heads_raw_data.iloc[time, :]).transpose()
    def get_ro_for_window(self, time, steps_ahead):
        return self.runoff_raw_data.iloc[time:time+steps_ahead,:]
    
    def get_ht_for_window(self, time, steps_ahead):
        return self.heads_raw_data.iloc[time:time+steps_ahead,:]


    def get_features_dictionary(self, *args):
        features_df = pd.concat(args).reset_index(drop =True).transpose()  
        node_names =        list(features_df.index)
        list_features =     features_df.values.tolist()
        input_features_dict=dict(zip(node_names, list_features))
        return input_features_dict


        
        
       