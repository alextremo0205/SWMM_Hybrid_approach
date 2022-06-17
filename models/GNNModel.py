import torch
from models.DynamicEmulatorLayer import DynEm

class GNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DynEM_layer = DynEm()
        
    def forward(self, data):
        edge_index =    data.edge_index
        x =             data.x
        norm_elev =     data.norm_elev
        norm_length =   data.norm_length
        norm_geom_1 =   data.norm_geom_1
        
        pred = self.DynEM_layer(edge_index, x, norm_elev, norm_length, norm_geom_1)

        return pred
    
    
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.DynEM_layer}, ')
                