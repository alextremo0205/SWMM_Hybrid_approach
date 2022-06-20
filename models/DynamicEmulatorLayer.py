import torch
from torch_geometric.nn import MessagePassing

from models.RunoffANN import RunoffANN
from models.InterchangeANN import InterchangeANN


class DynEm(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.interchangeANN = InterchangeANN()
        self.runoffANN = RunoffANN()
        
    def forward(self, edge_index, x, norm_elev, norm_length, norm_geom_1):
        out = self.propagate(edge_index, x=x, 
                             norm_elev = norm_elev, 
                             norm_length=norm_length, 
                             norm_geom_1 = norm_geom_1)
        return out
    
    def message(self, x_i, x_j, norm_elev_i, norm_elev_j, norm_length, norm_geom_1):
        
        hi = x_i[:, 0].reshape(-1,1)
        hj = x_j[:, 0].reshape(-1,1)
        
        mask_flows = self.get_mask_flows(hi, hj, norm_elev_i, norm_elev_j)

        dif = hj-hi
        assert dif.max().item() <= 1, 'Max. difference is greater than 1'
        assert dif.min().item() >= -1, 'Min. difference is less than -1'
        
        x_interchange = torch.concat((dif, norm_length, norm_geom_1, mask_flows), axis=1)
        nn_interchange = self.interchangeANN(x_interchange)

        depth_interchange = torch.mul(nn_interchange, mask_flows)
        
        return depth_interchange

    
    def update(self, inputs, x, norm_elev):
        
        h_runoff = self.runoffANN(x[:, 1].reshape(-1,1))

        hi = x[:, 0].reshape(-1,1)
        
        candidate_h = hi + inputs + h_runoff
        
        new_h = torch.max(candidate_h, norm_elev)
        
        #Outfall condition!
        # new_h[-2:] = norm_elev[-2:]
        
        return new_h
    
    
    
    def get_mask_flows(self, hi, hj, norm_elev_i, norm_elev_j):

        hi_dry = torch.eq(hi, norm_elev_i)
        hj_dry = torch.eq(hj, norm_elev_j)
        
        flow_i_to_j = torch.ge(hi, hj)
        flow_j_to_i = torch.ge(hj, hi)
        
        node_i_should_flow_but_it_is_dry = torch.logical_and(hi_dry, flow_i_to_j) 
        node_j_should_flow_but_it_is_dry = torch.logical_and(hj_dry, flow_j_to_i) 
        
        mask_zeros = torch.logical_or(node_i_should_flow_but_it_is_dry, node_j_should_flow_but_it_is_dry)
        mask_flows = torch.logical_not(mask_zeros)
        return mask_flows

    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.interchangeANN}, aggr={self.aggr}')
                
    