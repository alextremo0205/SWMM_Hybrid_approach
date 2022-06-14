import unittest

import networkx as nx
import sys
from os import path as path_lib

sys.path.insert(0, '')


#Import custom libraries after this line
from my_imports import *


class YAMLTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        yaml_path = 'config_file.yaml'
        cls.yaml_data = utils.load_yaml(yaml_path)
        
    @classmethod
    def tearDownClass(cls):
        del cls.yaml_data
        
        
    def test_valid_dictionary(self):
        yaml_data = self.__class__.yaml_data
        self.assertIsInstance(yaml_data, dict)
        
    def test_paths_validity(self):
        yaml_data = self.__class__.yaml_data
        directory_paths = [path for name, path in yaml_data.items() if 'path' in name]
        for dir_path in directory_paths:
            self.is_a_valid_path(dir_path)
        

    #Auxiliary functions
    def is_a_valid_path(self, c_path):
        self.assertIsInstance(c_path, str)
        self.assertTrue(path_lib.exists(c_path))


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

        for sim in list_of_simulations:
            hydraulic_heads_path = '\\'.join([simulations_path,sim,'hydraulic_head.pk'])
            runoff_path = '\\'.join([simulations_path,sim,'runoff.pk'])
            
            heads_raw_data = utils.get_heads_from_pickle(hydraulic_heads_path)
            runoff_raw_data = utils.get_runoff_from_pickle(runoff_path)
            
            sim = SWMMSimulation(G, heads_raw_data, runoff_raw_data)
            
            break
    
    
        cls.swmmSimulation = sim

    @classmethod
    def tearDownClass(cls):
        del cls.yaml_data
        del cls.swmmSimulation


    def test_create_hydraulic_simulation(self):
        self.assertTrue(self.swmmSimulation != None)

    def test_attributes_are_pd_dataframes(self):
        heads =  self.__class__.swmmSimulation.heads_raw_data
        runoff = self.__class__.swmmSimulation.runoff_raw_data
        
        attributes =[heads, runoff]

        for a in attributes:
            self.assertIsInstance(a, pd.DataFrame)
    
    def test_network_layout(self):
        pass
        
        
        

if __name__ == '__main__':
    unittest.main()