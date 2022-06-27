import unittest
import sys
sys.path.insert(0, '')
from my_imports import *

class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = {
            'Training loss':
                {0:1,
                1:2,
                2:3,
                3:4
                },
            'Validation loss':
                {0:2,
                1:3,
                2:4,
                3:5}
        }
        
        
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        self.fig = go.Figure()
    
    def test_dummy(self):
        self.assertEqual(0,0)
    
    # @unittest.skip
    # def test_get_scatter_from_dict(self):
    #     scatter_loss     = vis.get_scatter_from_dict(self.history, 'Training loss')
    #     scatter_val_loss = vis.get_scatter_from_dict(self.history, 'Validation loss')
        
    #     self.assertIsNotNone(scatter_loss)
    #     self.assertIsNotNone(scatter_val_loss)
        
    #     self.fig.add_trace(scatter_loss)
    #     self.fig.add_trace(scatter_val_loss)
        
    #     fig = vis.style_loss_fig(self.fig)
    #     vis.show(fig)
        
    # def test_plot_nodal_variable(self):
    #     trace = vis.plot_nodal_variable()
    #     self.assertIsNotNone(trace)
    #     self.fig.add_trace(trace)
        
    #     vis.show(self.fig)
        
        
if __name__ == '__main__':
    unittest.main()