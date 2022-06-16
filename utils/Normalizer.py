import torch

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
        
        
    def get_min_h(self):
        extreme_min_h_x = self.use_function_get_value(torch.min, 'h_x')
        extreme_min_h_y = self.use_function_get_value(torch.min, 'h_y')
        
        min_h = torch.min(extreme_min_h_x, extreme_min_h_y)
        
        return min_h
    
    def get_max_h(self):
        extreme_max_h_x = self.use_function_get_value(torch.max, 'h_x')
        extreme_max_h_y = self.use_function_get_value(torch.max, 'h_y')
        
        max_h = torch.max(extreme_max_h_x, extreme_max_h_y)

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
        
        window_with_x =     self.add_x_normalized_features(window)
        normalized_window = self.add_y_normalized_features(window_with_x)
        
        return normalized_window

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
    
    def normalize_h_min_max(self, original_h):
        return (original_h-self.min_h)/(self.max_h-self.min_h)
    
    def normalize_runoff_min_max(self, original_runoff):
        return (original_runoff-self.min_runoff)/(self.max_runoff-self.min_runoff)
    
    
    
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'