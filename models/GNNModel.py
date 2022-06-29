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

        pred = torch.zeros(num_nodes, steps_ahead)
        for step in range(steps_ahead):
            
            one_step_x = d.x[:, :2]

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
            
            pred[:, step] = y.reshape(-1)
            
            new_runoff = d.x[:, 2:]
            new_x = torch.cat((y, new_runoff), dim = 1)
            d['x'] = new_x
            print('new_x', new_x)
            print('new_x.shape', new_x.shape)
        return pred

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.DynEM_layer1}')
