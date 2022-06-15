import torch

class Normalizer:
    
    def __init__(self, training_windows):
        self.training_windows = training_windows
        self.min_h = self.get_min_h()
        self.max_h = self.get_max_h()
        self.min_length = self.get_min_length()
        self.max_length = self.get_max_length()
        
    def get_min_h(self):
        window = self.training_windows[0]
        
        min_h = window.x[:,0].min()
        for window in self.training_windows:
            min_h_in_x = window.x[:,0].min()
            min_h_in_y =  window.y.min().min()
            min_h_in_both = torch.min(min_h_in_x, min_h_in_y)
            if min_h_in_both < min_h:
                min_h = min_h_in_both
        
        return min_h
    
    def get_max_h(self):
        window = self.training_windows[0]
        max_h = window.x[:,0].max()
        for window in self.training_windows:
            max_h_in_x = window.x[:,0].max()
            max_h_in_y =  window.y.max().max()
            max_h_in_both = torch.max(max_h_in_x, max_h_in_y)
            
            if max_h_in_both > max_h:
                max_h = max_h_in_both

        return max_h
    
    def get_min_length(self):
        window = self.training_windows[0]
        min_length = window.length.min()
        return min_length
    
    def get_max_length(self):
        window = self.training_windows[0]
        max_length = window.length.max()
        return max_length
    
    # def get_min_length(self):
    #     window = self.training_windows[0]
    #     min_length = window.length.min()
    #     return min_length
    
    # def get_max_length(self):
    #     window = self.training_windows[0]
    #     max_length = window.length.max()
    #     return max_length