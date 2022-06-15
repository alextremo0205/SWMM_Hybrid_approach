import unittest
import networkx as nx

from torch_geometric.data import Data
import sys
sys.path.insert(0, '')


#Import custom libraries after this line
from my_imports import *

class SWMMSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = 'config_file.yaml'
        cls.yaml_data = utils.load_yaml(yaml_path)
        
        inp_path = cls.yaml_data['inp_path']
        simulations_path = cls.yaml_data['simulations_path']
        
        list_of_simulations = os.listdir(simulations_path)

        inp_lines = utils.get_lines_from_textfile(inp_path)
        G = utils.inp_to_G(inp_lines)

        simulations =[]
        max_events = 5
        num_saved_events = 0
        for simulation in list_of_simulations:
            hydraulic_heads_path = '\\'.join([simulations_path,simulation,'hydraulic_head.pk'])
            runoff_path = '\\'.join([simulations_path,simulation,'runoff.pk'])
            
            heads_raw_data = utils.get_heads_from_pickle(hydraulic_heads_path)
            runoff_raw_data = utils.get_runoff_from_pickle(runoff_path)
            
            sim = SWMMSimulation(G, heads_raw_data, runoff_raw_data)
            simulations.append(sim)
            if num_saved_events>=max_events:
                break
            
            num_saved_events+= 1
            
        event_to_test= 2
        cls.sim = simulations[event_to_test]

    @classmethod
    def tearDownClass(cls):
        del cls.yaml_data
        del cls.sim

    #Unit tests----------------------------------------------------------------

    def test_swmmSimulation_exists(self):
        self.assertTrue(self.sim != None)

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
        self.assert_window_x_has_right_shape(steps_ahead=1, time=0)
        self.assert_window_x_has_right_shape(steps_ahead=4, time=0)
        self.assert_window_x_has_right_shape(steps_ahead=30, time=0)
        
        self.assert_window_x_has_right_shape(steps_ahead=1, time=1)
        self.assert_window_x_has_right_shape(steps_ahead=1, time=10)
        
    def test_window_not_created_when_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.assert_window_x_has_right_shape(steps_ahead=1, time=10e6)
    
    def test_window_y_isNxTimeSteps(self):
        self.assert_window_y_has_right_shape(steps_ahead=1, time=0)
        
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
    def assert_window_x_has_right_shape(self, steps_ahead, time):
        window = self.sim.get_window(steps_ahead, time)
        x = window.x
        num_nodes = window.num_nodes
        num_x_features = window.num_node_features
        desired_shape = (num_nodes, num_x_features)
        self.assertTrue(x.shape, desired_shape)
    
    def assert_window_y_has_right_shape(self, steps_ahead, time):
        window = self.sim.get_window(steps_ahead, time)
        y = window.y
        num_nodes = window.num_nodes
        desired_shape = (num_nodes, steps_ahead)
        self.assertTrue(y.shape, desired_shape)
    

if __name__ == '__main__':
    unittest.main()