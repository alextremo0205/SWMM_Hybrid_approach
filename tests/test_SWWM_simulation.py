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

        for simulation in list_of_simulations:
            hydraulic_heads_path = '\\'.join([simulations_path,simulation,'hydraulic_head.pk'])
            runoff_path = '\\'.join([simulations_path,simulation,'runoff.pk'])
            
            heads_raw_data = utils.get_heads_from_pickle(hydraulic_heads_path)
            runoff_raw_data = utils.get_runoff_from_pickle(runoff_path)
            
            sim = SWMMSimulation(G, heads_raw_data, runoff_raw_data)
            
            break
    
    
        cls.sim = sim

    @classmethod
    def tearDownClass(cls):
        del cls.yaml_data
        del cls.sim


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
        window = self.sim.get_window(steps_ahead=1)
        self.assertIsInstance(window, Data)
        
    def test_window_x_isNxF(self):
        print(self.sim.heads_raw_data)
        
        steps_ahead=1
        window = self.sim.get_window(steps_ahead)
        num_nodes = window.num_nodes
        num_x_features = window.num_node_features
        x = window.x
        
        
        self.assertTrue(x.shape(), (num_nodes, num_x_features))
    


if __name__ == '__main__':
    unittest.main()