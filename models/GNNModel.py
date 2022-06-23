import copy
import torch
from models.DynamicEmulatorLayer import DynEm

class GNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DynEM_layer = DynEm()
        
    def forward(self, data):
        d = copy.deepcopy(data)
        edge_index =    d.edge_index

        norm_elev       = d.norm_elev
        norm_length     = d.norm_length
        norm_geom_1     = d.norm_geom_1
        norm_in_offset  = d.norm_in_offset
        norm_out_offset  = d.norm_out_offset

        num_nodes = d.num_nodes
        steps_ahead = d.x.shape[1] - 1
        
        pred = torch.zeros(num_nodes, steps_ahead)
        for step in range(steps_ahead):
            x = d.x
            new_h = self.DynEM_layer(edge_index, x, norm_elev, norm_length, norm_geom_1, norm_in_offset, norm_out_offset)
            
            pred[:, step] = new_h.reshape(-1)
            
            new_runoff = x[:, 2:]
            
            new_x = torch.cat((new_h, new_runoff), dim = 1)
            
            d['x'] = new_x
            
        return pred
    
    
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.DynEM_layer}')
                