import unittest
import utils_data as ud
from os import path as path_lib


class TestDatabaseYAML(unittest.TestCase):

    def setUp(self):
        self.yaml_directory = r'C:\Users\agarzondiaz\surfdrive\Year 2\Paper 2 - 3.0\data\database_config.yaml'
        self.data = ud.import_config_from_yaml(self.yaml_directory)

    def tearDown(self):
        pass

    
    
    def test_import_config_from_yaml(self):
        self.is_a_valid_path(self.yaml_directory)
        self.assertIsInstance(self.data, dict)
        
    def test_directories_validity(self):
        directory_paths = [dir for name, dir in self.data.items() if 'directory' in name]
        for dir_path in directory_paths:
            self.is_a_valid_path(dir_path)

    #Synthetic rainfalls
    def test_synthetic_rain(self):
        n_rainfalls =self.data['n_synthetic_rainfalls']
        self.assertIsInstance(n_rainfalls, int)
        self.assertGreater(n_rainfalls, 0)
    
    def test_blocks_values_durations_are_lists(self):
        block_values = self.data['block_values']
        block_durations = self.data['block_durations']

        self.assertIsInstance(block_values,list)
        self.assertIsInstance(block_durations,list)

        integers_in_list = block_values + block_durations
        for i in integers_in_list:
            self.assertIsInstance(i,int)



    #Auxiliary functions
    def is_a_valid_path(self, c_path):
        self.assertIsInstance(c_path, str)
        self.assertTrue(path_lib.exists(c_path))







if __name__ == '__main__':
    unittest.main()