import torch
import pandas as pd
from torch_geometric.loader import DataLoader

class Normalizer:
    
    def __init__(self, training_windows):
        self.training_windows = training_windows
        self.min_h = self.get_min_h()
        self.max_h = self.get_max_h()
        
        self.max_length = self.use_function_get_value(torch.max, 'length')
        self.min_length = self.use_function_get_value(torch.min, 'length')
        
        self.max_geom_1 = self.use_function_get_value(torch.max, 'geom_1')
        self.min_geom_1 = self.use_function_get_value(torch.min, 'geom_1')
        
        self.max_runoff = self.use_function_get_value(torch.max, 'runoff')
        self.min_runoff = self.use_function_get_value(torch.min, 'runoff')
        
        self.name_nodes = training_windows[0].name_nodes
        
    def get_min_h(self):
        extreme_min_h_x = self.use_function_get_value(torch.min, 'h_x')
        extreme_min_h_y = self.use_function_get_value(torch.min, 'h_y')
        extreme_min_elev = self.use_function_get_value(torch.min, 'elevation')
        
        min_h_water = torch.min(extreme_min_h_x, extreme_min_h_y)
        min_h = torch.min(min_h_water, extreme_min_elev)
        
        return min_h
    
    def get_max_h(self):
        extreme_max_h_x = self.use_function_get_value(torch.max, 'h_x')
        extreme_max_h_y = self.use_function_get_value(torch.max, 'h_y')
        extreme_max_elev = self.use_function_get_value(torch.max, 'elevation')
        
        max_h_water = torch.max(extreme_max_h_x, extreme_max_h_y)
        max_h = torch.max(max_h_water, extreme_max_elev)
        
        return max_h
    
    def use_function_get_value(self, f, attribute):
        window = self.training_windows[0]
        extreme = f(window[attribute])
        
        for window in self.training_windows:
            candidate = f(window[attribute])
            extreme = f(extreme, candidate)
        
        return extreme
    
    def get_list_normalized_training_windows(self):
        list_norm_windows = [self.normalize_window(window) for window in self.training_windows]
        return(list_norm_windows)
    
    def normalize_window(self, window):
        
        window_with_x           = self.add_x_normalized_features(window)
        window_with_y           = self.add_y_normalized_features(window_with_x)
        window_with_elev        = self.add_elev_norm_features(window_with_y)
        window_with_geom_1      = self.add_geom_1_norm_features(window_with_elev)
        window_with_length      = self.add_length_norm_features(window_with_geom_1)
        window_with_in_offset   = self.add_in_offset_norm_features(window_with_length)
        window_with_out_offset  = self.add_out_offset_norm_features(window_with_in_offset)
        
        final_normalized_window = window_with_out_offset
        return final_normalized_window

    def add_x_normalized_features(self, window):
        norm_h_x = self.normalize_h_min_max(window['h_x'])
        norm_runoff = self.normalize_runoff_min_max(window['runoff'])
        
        x = torch.cat((norm_h_x, norm_runoff), dim = 1)
        window['x'] = x
        return window
    
    def add_y_normalized_features(self, window):
        norm_h_y = self.normalize_h_min_max(window['h_y'])
        window['y'] = norm_h_y
        return window
    
    def add_elev_norm_features(self, window):
        norm_elev = self.normalize_h_min_max(window['elevation'])
        norm_elev= norm_elev.reshape(-1,1)
        window['norm_elev'] = norm_elev
        
        return window
    
    def add_in_offset_norm_features(self, window):
        norm_in_offset = self.scale_with_h_min_max(window['in_offset'])
        norm_in_offset = norm_in_offset.reshape(-1,1)
        window['norm_in_offset'] = norm_in_offset
        
        return window
   
    def add_out_offset_norm_features(self, window):
        norm_out_offset = self.scale_with_h_min_max(window['out_offset'])
        norm_out_offset = norm_out_offset.reshape(-1,1)
        window['norm_out_offset'] = norm_out_offset
        return window
   
    
    def add_geom_1_norm_features(self, window):
        norm_geom_1 = self.normalize_geom_1_min_max(window['geom_1'])
        norm_geom_1= norm_geom_1.reshape(-1,1)
        window['norm_geom_1'] = norm_geom_1
        
        return window
    
    def add_length_norm_features(self, window):
        norm_length = self.normalize_length_min_max(window['length'])
        norm_length= norm_length.reshape(-1,1)
        window['norm_length'] = norm_length
        
        return window
    
    def normalize_h_min_max(self, original_h):
        return (original_h-self.min_h)/(self.max_h-self.min_h)
    
    def scale_with_h_min_max(self, relevant_distance):
        return (relevant_distance)/(self.max_h-self.min_h)
    
    def normalize_runoff_min_max(self, original_runoff):
        return (original_runoff-self.min_runoff)/(self.max_runoff-self.min_runoff)
   
    def normalize_length_min_max(self, original_length):
        return (original_length-self.min_length)/(self.max_length-self.min_length)
    
    def normalize_geom_1_min_max(self, original_geom_1):
        return (original_geom_1-self.min_geom_1)/(self.max_geom_1-self.min_geom_1)
    

    def unnormalize_heads(self, normalizedHeads):
        return (normalizedHeads)*(self.max_h-self.min_h) + self.min_h
    
    def get_unnormalized_heads_pd(self, tensor_heads):
        normalized_heads_tensor     = self.unnormalize_heads(tensor_heads)
        normalized_heads_np         = normalized_heads_tensor.detach().numpy()
        normalized_heads_pd         = pd.DataFrame(dict(zip(self.name_nodes, normalized_heads_np)))
        return normalized_heads_pd
        
    def get_dataloader(self, batch_size):
        list_of_windows = self.get_list_normalized_training_windows()
        return DataLoader(list_of_windows, batch_size)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'