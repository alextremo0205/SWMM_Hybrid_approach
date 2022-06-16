import unittest
import networkx as nx
from torch_geometric.data import Data


import sys
sys.path.insert(0, '')
from my_imports import *

class SWMMSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = 'config_file.yaml'
        cls.yaml_data = utils.load_yaml(yaml_path)
        
        inp_path = cls.yaml_data['inp_path']
        simulations_path = cls.yaml_data['simulations_path']
        
        simulations = utils.extract_simulations_from_folders(simulations_path, inp_path, max_events = 5)

        event_to_test= 2
        cls.sim = simulations[event_to_test]

    @classmethod
    def tearDownClass(cls):
        del cls.yaml_data
        del cls.sim

    #Unit tests----------------------------------------------------------------

    def test_swmmSimulation_exists(self):
        self.assertIsInstance(self.sim, SWMMSimulation)

    def test_attributes_are_pd_dataframes(self):
        heads =  self.sim.heads_raw_data
        runoff = self.sim.runoff_raw_data
        
        attributes =[heads, runoff]

        for a in attributes:
            self.assertIsInstance(a, pd.DataFrame)
    
    def test_G_is_nx_graph(self):
        G = self.sim.G
        self.assertIsInstance(G, nx.Graph)
        
    def test_window_is_PyG(self):
        window = self.sim.get_window(steps_ahead=1, time =0)
        self.assertIsInstance(window, Data)
        
    def test_window_x_isNxF(self):
        self.assert_window_h_in_x_has_right_shape(steps_ahead=1, time=0)
        self.assert_window_h_in_x_has_right_shape(steps_ahead=4, time=0)
        self.assert_window_h_in_x_has_right_shape(steps_ahead=30, time=0)
        
        self.assert_window_h_in_x_has_right_shape(steps_ahead=1, time=1)
        self.assert_window_h_in_x_has_right_shape(steps_ahead=1, time=10)
        
    def test_window_not_created_when_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.assert_window_h_in_x_has_right_shape(steps_ahead=1, time=10e6)
    
    def test_window_h_in_y_isNxTimeSteps(self):
        self.assert_window_h_in_y_has_right_shape(steps_ahead=1, time=0)
        
    def test_get_list_of_windows_composition(self):
        steps_ahead = 1
        all_windows = self.sim.get_all_windows(steps_ahead)
        
        self.assertIsInstance(all_windows, list)
        for window in all_windows:
            self.assertIsInstance(window, Data)

    def test_list_of_windows_length(self):
        with self.assertRaises(AssertionError):
            for steps in range(2):
                self.assertNumberOfWindows_with(steps_ahead = steps)   
        
        self.assertNumberOfWindows_with(steps_ahead = int(10e6))

    




    #Auxiliary functions----------------------------------------------------------------
    def assertNumberOfWindows_with(self, steps_ahead):
        all_windows = self.sim.get_all_windows(steps_ahead)
        num_windows_ideal = self.sim.simulation_length//steps_ahead
        num_windows_current = len(all_windows)         
        self.assertEqual(num_windows_current, num_windows_ideal)
    def assert_window_h_in_x_has_right_shape(self, steps_ahead, time):
        window = self.sim.get_window(steps_ahead, time)
        h_x = window['h_x']
        num_nodes = window.num_nodes
        desired_shape = (num_nodes, 1)
        self.assertTrue(h_x.shape, desired_shape)
    
    def assert_window_h_in_y_has_right_shape(self, steps_ahead, time):
        window = self.sim.get_window(steps_ahead, time)
        h_y = window['h_y']
        num_nodes = window.num_nodes
        desired_shape = (num_nodes, steps_ahead)
        self.assertTrue(h_y.shape, desired_shape)
    

if __name__ == '__main__':
    unittest.main()