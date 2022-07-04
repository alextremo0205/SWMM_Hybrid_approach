import copy
import torch
from models.DynEm import DynEm

class GNNModel(torch.nn.Module):
    def __init__(self, steps_ahead, steps_behind):
        super().__init__()
        self.steps_ahead    = steps_ahead
        self.steps_behind   = steps_behind
        self.length_window  = (2*steps_behind) + steps_ahead
        
        self.DynEM_layer1 = DynEm(in_dims = 2 * self.length_window + 2, 
                                  in_node_features = self.length_window, 
                                  out_dims = 5)
        self.DynEM_layer2 = DynEm(in_dims = 2*5 + 2, 
                                  in_node_features = 5, 
                                  out_dims = steps_ahead)

    def forward(self, data):
        d = copy.deepcopy(data) 
        
        edge_index      = d.edge_index

        norm_elev       = d.norm_elev
        norm_length     = d.norm_length
        norm_geom_1     = d.norm_geom_1
        norm_in_offset  = d.norm_in_offset
        norm_out_offset = d.norm_out_offset
        
        h0 = d.x[:, :self.steps_behind]
        runoff = d.x[:, self.steps_behind:]

        length_simulation = runoff.shape[1] - self.steps_behind
        num_nodes       = d.num_nodes
        
        pred = torch.zeros(num_nodes, length_simulation)        

        for step in range(0,length_simulation, self.steps_ahead):
            
            runoff_step = runoff[:, step:step+2*self.steps_ahead]
            one_step_x = torch.cat((h0, runoff_step), dim = 1)
            out_mp = self.DynEM_layer1(edge_index,
                                one_step_x,
                                norm_elev,
                                norm_length,
                                norm_geom_1,
                                norm_in_offset,
                                norm_out_offset
                                )
            
            y = self.DynEM_layer2(edge_index,
                                out_mp,
                                norm_elev,
                                norm_length,
                                norm_geom_1,
                                norm_in_offset,
                                norm_out_offset
                                )
            
            pred[:, step:step+self.steps_ahead] = y
            h0 = self.get_new_h0(h0,y)

        return pred

    def get_new_h0(self, h0, y):
        original_size = h0.shape[1]
        concatenated = torch.cat((h0, y), dim =1)
        new_h0 = concatenated[:, -original_size:]
        
        return new_h0
        
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.DynEM_layer1}')


    