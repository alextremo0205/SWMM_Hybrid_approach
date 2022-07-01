import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx

class SWMMSimulation:
    def __init__(self, G, heads_raw_data, runoff_raw_data, name_simulation):
        
        self.G                  = G
        self.heads_raw_data     = heads_raw_data
        self.runoff_raw_data    = runoff_raw_data
        self.simulation_length  = len(self.heads_raw_data)
        self.name_simulation    = name_simulation
        
    def get_simulation_in_one_window(self):
        one_window = self.get_all_windows(steps_ahead = self.simulation_length - 2)[0]
        return one_window
        
    def get_all_windows(self, steps_ahead):
        assert steps_ahead>0, "The steps should be greater than 0"
        max_time_allowed = (self.simulation_length - steps_ahead) - 1
        windows_list = []
        
        for time in range(0, max_time_allowed, steps_ahead):
            window = self.get_window(steps_ahead, time)
            windows_list.append(window)

        return windows_list
    
    def get_window(self, steps_ahead, time):
        
        self.checkOutOfBounds(steps_ahead, time)
        
        h0 =                self.get_h0_for_window(time)
        ro_timeperiod =     self.get_ro_for_window(time, steps_ahead)
        ht_timeperiod =     self.get_ht_for_window(time, steps_ahead)
        
        h_x_dict=    self.get_features_dictionary(h0)
        ro_x_dict =  self.get_features_dictionary(ro_timeperiod)
        output_features_dict=   self.get_features_dictionary(ht_timeperiod)
        
        G_for_window =      self.G
        nx.set_node_attributes(G_for_window, h_x_dict, name = 'h_x')
        nx.set_node_attributes(G_for_window, ro_x_dict, name ='runoff')
        nx.set_node_attributes(G_for_window, output_features_dict,name = 'h_y')
        
        window = from_networkx(G_for_window)

        window['steps_ahead'] = steps_ahead
        
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
        return self.heads_raw_data.iloc[time+1:time+steps_ahead+1,:]

    def get_features_dictionary(self, *args):
        features_df         = pd.concat(args).reset_index(drop =True).transpose()  
        node_names          = list(features_df.index)
        list_features       = features_df.values.tolist()
        input_features_dict = dict(zip(node_names, list_features))
        
        return input_features_dict


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name_simulation})'