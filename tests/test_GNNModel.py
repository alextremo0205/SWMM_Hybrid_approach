import unittest

import sys
sys.path.insert(0, '')

from my_imports import *
from models.GNNModel import GNNModel

class TestGNNModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = 'config_file.yaml'
        cls.yaml_data = utils.load_yaml(yaml_path)
        
        inp_path = cls.yaml_data['inp_path']
        simulations_path = cls.yaml_data['training_simulations_path']
        
        simulations = utils.extract_simulations_from_folders(simulations_path, 
                                                             inp_path, 
                                                             max_events = 5)
        training_windows = []
        events_to_train = [0,1]
        for event in events_to_train:
            sim = simulations[event]
            training_windows += sim.get_all_windows(steps_ahead = 2)
        
        cls.normalizer = Normalizer(training_windows)
        cls.trial_window = cls.normalizer.get_list_normalized_training_windows()[0]
        cls.GNN_model = GNNModel()
    
    def test_GNN_model_is_valid(self):
        self.assertTrue(self.GNN_model!=None)

    def test_GNN_output_is_tensor(self):
        self.assertIsInstance(self.GNN_model(self.trial_window), torch.Tensor)
    
    def test_GNN_layer_exists(self):
        self.assertTrue(self.GNN_model.DynEM_layer1 != None)

    def test_GNN_output_is_right_shape(self):
        output = self.GNN_model(self.trial_window)
        num_nodes = self.trial_window.num_nodes
        num_timesteps = self.trial_window['steps_ahead']
        
        self.assertEqual(output.shape, (num_nodes, num_timesteps))
    
    def test_GNN_has_trainable_parameters(self):
        self.assertGreater(utils.count_parameters(self.GNN_model), 0)
        
if __name__ == '__main__':
    unittest.main()