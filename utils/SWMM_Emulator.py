import torch
import networkx as nx
import utils.head_change_utils as utils
class SWMMEmulator:
    def __init__(self, inp_path):
        self.inp_lines = utils.get_lines_from_textfile(inp_path)
        self.G = utils.inp_to_G(self.inp_lines)
        
        self.original_min =     dict_to_torch( nx.get_node_attributes(self.G, 'elevation') )
        self.original_A_catch = dict_to_torch( nx.get_node_attributes(self.G, 'area_subcatchment') )
        self.pos = nx.get_node_attributes(self.G, 'pos')

def to_torch(object_to_convert):
    return torch.tensor(float(object_to_convert), dtype=torch.float32)

def dict_to_torch(d):
    dict_torch = {k:to_torch(v) for k,v in d.items()}
    return dict_torch
