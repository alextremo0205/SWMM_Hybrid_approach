import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from models.NodeFeaturesANN import NodeFeaturesANN
from models.InterchangeANN import InterchangeANN


class DynEm(MessagePassing):
    def __init__(self, in_dims, in_node_features, out_dims):
        
        super().__init__(aggr='add', flow = 'target_to_source')
        self.interchangeANN     = InterchangeANN(in_dims, out_dims)
        self.nodeFeaturesANN    = NodeFeaturesANN(in_node_features, out_dims)
        
        
    def forward(self, edge_index, 
                    x, 
                    norm_elev,
                    norm_length, 
                    norm_geom_1, 
                    norm_in_offset, 
                    norm_out_offset):
        out = self.propagate(edge_index, x=x, 
                             norm_elev = norm_elev, 
                             norm_length=norm_length, 
                             norm_geom_1 = norm_geom_1,
                             norm_in_offset = norm_in_offset,
                             norm_out_offset = norm_out_offset)
        return out
    
    def message(self, x_i, x_j, norm_elev_i, norm_elev_j, norm_length, norm_geom_1, norm_in_offset, norm_out_offset):
        
        # hi = x_i[:, 0].reshape(-1,1)
        # hj = x_j[:, 0].reshape(-1,1)
        
        # mask_flows = self.get_mask_flows(hi, hj, norm_elev_i, norm_elev_j, norm_in_offset, norm_out_offset)

        # dif = (hj-hi) #nn.Tanh()
        
        # assert dif.max().item() <= 1, 'Max. difference is greater than 1 ' + str(dif.max().item())
        # assert dif.min().item() >= -1, 'Min. difference is less than -1 ' + str(dif.min().item())
        
        x_interchange = torch.concat((x_i, x_j, norm_length, norm_geom_1), axis=1)
        result_nn_interchange = self.interchangeANN(x_interchange)
        
        # depth_interchange = torch.mul(result_nn_interchange, mask_flows)
        
        return result_nn_interchange

    
    def update(self, inputs, x, norm_elev):
        
        # print('Dims of message (or inputs): ', inputs.shape)
        
        new_var = self.nodeFeaturesANN(x)
        candidate_h = new_var + inputs
        # print('Dims of self.nodeFeaturesANN(x): ', candidate_h.shape)
        # new_h = torch.max(candidate_h, norm_elev)
               
        return candidate_h
    
    
    
    def get_mask_flows(self, hi, hj, norm_elev_i, norm_elev_j, norm_in_offset, norm_out_offset):
        adjusted_elev_i = norm_elev_i + norm_out_offset
        adjusted_elev_j = norm_elev_j + norm_in_offset
        
        hi_is_over_invert = torch.gt(hi, adjusted_elev_i)
        hj_is_over_invert = torch.gt(hj, adjusted_elev_j)
        
        should_flow_i_to_j = torch.ge(hi, hj)
        should_flow_j_to_i = torch.ge(hj, hi)
        
        node_i_will_flow = torch.logical_and(hi_is_over_invert, should_flow_i_to_j) 
        node_j_will_flow = torch.logical_and(hj_is_over_invert, should_flow_j_to_i) 
        
        mask_flows = torch.logical_or(node_i_will_flow, node_j_will_flow)
            
        return mask_flows

    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.interchangeANN}, aggr={self.aggr}')
                
    
