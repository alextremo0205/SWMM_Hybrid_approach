import unittest

import sys
from os import path as path_lib


sys.path.insert(0, '')
from my_imports import *

class YAMLTest(unittest.TestCase):

    def setUp(self):
        yaml_path = 'config_file.yaml'
        self.yaml_data = utils.load_yaml(yaml_path)
    
    def tearDown(self):
        pass

    def test_valid_dictionary(self):
        self.assertIsInstance(self.yaml_data, dict)
    
    def test_paths_validity(self):
        directory_paths = [path for name, path in self.yaml_data.items() if 'path' in name]
        for dir_path in directory_paths:
            self.is_a_valid_path(dir_path)
        




#Auxiliary functions
    def is_a_valid_path(self, c_path):
        self.assertIsInstance(c_path, str)
        self.assertTrue(path_lib.exists(c_path))




if __name__ == '__main__':
    unittest.main()