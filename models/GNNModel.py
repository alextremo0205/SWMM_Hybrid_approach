import copy
import torch
from models.DynEm import DynEm

class GNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DynEM_layer1 = DynEm(in_dims = 6, in_node_features = 2, out_dims = 3)
        self.DynEM_layer2 = DynEm(in_dims = 8, in_node_features = 3, out_dims = 1)
        
    def forward(self, data):
        d = copy.deepcopy(data)
        edge_index      = d.edge_index

        norm_elev       = d.norm_elev
        norm_length     = d.norm_length
        norm_geom_1     = d.norm_geom_1
        norm_in_offset  = d.norm_in_offset
        norm_out_offset = d.norm_out_offset

        num_nodes = d.num_nodes
        steps_ahead = d.x.shape[1] - 1

        x = d.x
        
        x = self.DynEM_layer1(edge_index, 
                                 x, 
                                 norm_elev, 
                                 norm_length, 
                                 norm_geom_1, 
                                 norm_in_offset, 
                                 norm_out_offset
                                 )
        
        x = self.DynEM_layer2(edge_index, 
                            x, 
                            norm_elev, 
                            norm_length, 
                            norm_geom_1, 
                            norm_in_offset, 
                            norm_out_offset
                            )
    
        
        # pred = torch.zeros(num_nodes, steps_ahead)
        # for step in range(steps_ahead):
        #     x = d.x
        #     new_h = self.DynEM_layer1(edge_index, x, norm_elev, norm_length, norm_geom_1, norm_in_offset, norm_out_offset)
        #     pred[:, step] = new_h.reshape(-1)
        #     new_runoff = x[:, 2:]
        #     new_x = torch.cat((new_h, new_runoff), dim = 1)
        #     d['x'] = new_x
        
        return x
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.DynEM_layer1}')
                