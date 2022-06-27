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
        


if __name__ == '__main__':
    unittest.main()