import unittest

from torch_geometric.data import Data

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
        self.assertGreaterEqual(max_h, min_h)

if __name__ == '__main__':
    unittest.main()