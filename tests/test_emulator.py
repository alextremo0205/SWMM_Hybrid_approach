import unittest

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




class HydraulicSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = 'config_file.yaml'
        cls.yaml_data = utils.load_yaml(yaml_path)
        simulations_path = cls.yaml_data['simulations_path']

        list_of_simulations = os.listdir(simulations_path)

        for sim in list_of_simulations:
            rain_path = '\\'.join([simulations_path,sim,sim+'.dat'])
            hydraulic_heads_path = '\\'.join([simulations_path,sim,'hydraulic_head.pk'])
            runoff_path = '\\'.join([simulations_path,sim,'runoff.pk'])
            break


        # #Rainfall
        rainfall_raw_data = utils.get_rain_in_pandas(rain_path)

        # #Hydraulic head
        heads_raw_data =    utils.get_heads_from_pickle(hydraulic_heads_path)

        # #Runoff
        runoff_raw_data =   utils.get_runoff_from_pickle(runoff_path)
    
    
        cls.swmmObject = SWMMEventSimulation(rainfall_raw_data,
                                            heads_raw_data,
                                            runoff_raw_data)

    @classmethod
    def tearDownClass(cls):
        del cls.yaml_data
        del cls.swmmObject


    def test_create_hydraulic_simulation(self):
        self.assertTrue(self.swmmObject != None)

    def test_read_attributes(self):
        rain =   self.__class__.swmmObject.rainfall_raw_data
        heads =  self.__class__.swmmObject.heads_raw_data
        runoff = self.__class__.swmmObject.runoff_raw_data

        attributes =[rain, heads, runoff]

        for a in attributes:
            self.assertIsInstance(a, pd.DataFrame)
        

if __name__ == '__main__':
    unittest.main()