import unittest

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, '')

from my_imports import *

class NormalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = 'config_file.yaml'
        cls.yaml_data = utils.load_yaml(yaml_path)
        
        inp_path = cls.yaml_data['inp_path']
        simulations_path = cls.yaml_data['simulations_path']
        
        simulations = utils.extract_simulations_from_folders(simulations_path, 
                                                             inp_path, 
                                                             max_events = 5)

        training_windows = []
        events_to_train= [0,1,2]
        for event in events_to_train:
            sim = simulations[event]
            training_windows += sim.get_all_windows(steps_ahead = 2)
        
        cls.normalizer = Normalizer(training_windows)
        
        cls.first_window = cls.normalizer.training_windows[0]
        cls.normalized_window = cls.normalizer.normalize_window(cls.first_window)
        
    @classmethod
    def tearDownClass(cls):
        pass
        
    #Unit tests ----------------------------------------------------------------
           
    def test_normalizer_is_right_class(self):
        self.assertIsInstance(self.normalizer, Normalizer)
    
    
    def test_normalizer_has_list_of_windows(self):
        training_windows = self.normalizer.training_windows
        self.assertIsInstance(training_windows, list)
        
        for window in training_windows:
            self.assertIsInstance(window, Data)
    
    def test_max_head_is_greater_than_min_head(self):
        min_h = self.normalizer.min_h
        max_h = self.normalizer.max_h
        self.assertGreater(max_h, min_h)

    def test_h_min(self):
        min_h = self.normalizer.min_h
        for window in self.normalizer.training_windows:
            self.assertLessEqual(min_h, window['h_x'].min())
            self.assertLessEqual(min_h, window['h_y'].min())
            self.assertLessEqual(min_h, window['elevation'].min())

    def test_h_max(self):
        max_h = self.normalizer.max_h
        for window in self.normalizer.training_windows:
            self.assertGreaterEqual(max_h, window['h_x'].max())
            self.assertGreaterEqual(max_h, window['h_y'].max())
            self.assertGreaterEqual(max_h, window['elevation'].min())

    def test_min_length(self):
        min_length = self.normalizer.min_length
        for window in self.normalizer.training_windows:
            self.assertLessEqual(min_length, window.length.min())
    
    def test_max_length(self):
        max_length = self.normalizer.max_length
        for window in self.normalizer.training_windows:
            self.assertGreaterEqual(max_length, window.length.max())
        
    def test_max_geom_1(self):
        max_geom_1 = self.normalizer.max_geom_1
        for window in self.normalizer.training_windows:
            self.assertGreaterEqual(max_geom_1, window.geom_1.max())
        
    def test_min_geom_1(self):
        min_geom_1 = self.normalizer.min_geom_1
        for window in self.normalizer.training_windows:
            self.assertLessEqual(min_geom_1, window.geom_1.min())
        
    def test_min_runoff(self):
        min_runoff = self.normalizer.min_runoff
        for window in self.normalizer.training_windows:
            self.assertLessEqual(min_runoff, window['runoff'].min())

    def test_max_runoff(self):
        max_runoff = self.normalizer.max_runoff
        for window in self.normalizer.training_windows:
            self.assertGreaterEqual(max_runoff, window['runoff'].max())

    def test_normalized_window_is_Data(self):
        self.assertIsInstance(self.normalized_window, Data)

    def test_normalized_window_x_features(self):
        self.assertGreaterEqual(self.normalized_window['x'].min().item() , 0)
        self.assertLessEqual(self.normalized_window['x'].max().item() , 1)
    
    def test_normalized_window_y_features(self):
        self.assertGreaterEqual(self.normalized_window['y'].min().item() , 0)
        self.assertLessEqual(self.normalized_window['y'].max().item() , 1)
    
    def test_normalized_training_windows(self):
        list_norm_windows = self.normalizer.get_list_normalized_training_windows()
        self.assertIsInstance(list_norm_windows, list)
        self.assertIsInstance(list_norm_windows[0], Data)
        
    def test_get_dataloader(self):
        dataloader = self.normalizer.get_dataloader(batch_size = 10)
        self.assertIsInstance(dataloader,DataLoader)
    
if __name__ == '__main__':
    unittest.main()