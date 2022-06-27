import torch
import numpy as np
import networkx as nx
import libraries.utils as utils


def to_torch(object_to_convert):
    return torch.tensor(float(object_to_convert), dtype=torch.float32)

def dict_to_torch(d):
    dict_torch = {k:to_torch(v) for k,v in d.items()}
    return dict_torch



nodes_outfalls = ['j_90552', 'j_90431']

class DynamicEmulator:
    
    def __init__(self, inp_path, q_transfer_ANN, q_runoff_ANN):
        

        self.inp_lines = utils.get_lines_from_textfile(inp_path)
        self.G = utils.inp_to_G(self.inp_lines)
        
        self.original_min =     dict_to_torch( nx.get_node_attributes(self.G, 'elevation') )
        self.original_A_catch = dict_to_torch( nx.get_node_attributes(self.G, 'area_subcatchment') )
        
        # self.h =                dict_to_torch(initial_h0)
        
        self.set_normalized_length()
        self.set_normalized_geom_1()
        
        self.q_transfer_ANN = q_transfer_ANN
        self.q_runoff_ANN = q_runoff_ANN
        
        self.pos = nx.get_node_attributes(self.G, 'pos')

        self.time = 0

    def set_normalized_length(self):
        length_dict = nx.get_edge_attributes(self.G, 'length')
        all_lengths = np.array([float(i) for i in list(length_dict.values())])
        self.max_length, self.min_length = all_lengths.max(), all_lengths.min()
        
        normalized_length = {k:self.normalize_length(float(v)) for k,v in length_dict.items()}
        nx.set_edge_attributes(self.G, normalized_length, name = 'normalized_length')

    def normalize_length(self, length):
        return (length - self.min_length) / (self.max_length-self.min_length)


    def set_normalized_geom_1(self):
        geom_1_dict = nx.get_edge_attributes(self.G, 'geom_1')
        all_geom_1 = np.array([float(i) for i in list(geom_1_dict.values())])
        self.max_geom_1, self.min_geom_1 = all_geom_1.max(), all_geom_1.min()
        
        normalized_geom_1 = {k:self.normalize_geom_1(float(v)) for k,v in geom_1_dict.items()}
        nx.set_edge_attributes(self.G, normalized_geom_1, name = 'normalized_geom_1')

    def normalize_geom_1(self, geom_1):
        return (geom_1 - self.min_geom_1) / (self.max_geom_1-self.min_geom_1)


    def get_h(self):
        return self.h

    def set_h(self, new_h):
        self.h = dict_to_torch(new_h)



    def update_h(self, runoff):#, time, prev_state_pump):
        
        new_h0= {}
        new_h0_2= {}
        nodes_pump = []                         
        outfalls = nodes_outfalls               

        for node, hi in self.h.items():         #links connected to that node
            
            hi_min = self.original_min[node]
            
            total_dh_transfer = 0
            
            for _, neigh in self.G.edges(node):      #Updates for each of the neighbors
                hj = self.h[neigh]
                hj_min = self.original_min[neigh]
                
                #Extract edge attributes
                link = self.G.edges[node, neigh]

                if is_giver_manhole_dry(hi, hi_min, hj, hj_min):     #if the heads are in their minimum, they cannot give water
                    q_transfer = torch.zeros(1, 1)
                else:                                                   #In case there is valid gradient, this function calculates the change in head
                    q_transfer = self.calculate_dh_transfer(hj, hi, link)
                
                total_dh_transfer += q_transfer
                
            if runoff[node]==0:
                q_runoff = torch.zeros(1, 1)
            else:
                q_runoff = self.calculate_dh_runoff(runoff[node])    
                
            total_dh = total_dh_transfer + q_runoff #(q_transfer + q_rain(rain[time], original_A_catch[node], weight_rain) + q_dwf(original_basevalue_dwf[node], dwf_hourly[time%24], weight_dwf))*dt 
            
            hi_min = torch.reshape(hi_min, (1, 1))
            new_h0[node] = max(hi+total_dh, hi_min) #The new head cannot be under the minimimum level. Careful!! A node may be giving more than it has to offer.

        for node_outfalls in outfalls:
            min_outfall=self.original_min[node_outfalls]
            new_h0[node_outfalls] = torch.reshape(min_outfall, (1,1))

        self.h = new_h0

    def draw_nx_layout(self):
        nx.draw(self.G, pos = self.pos, node_size=15)

    def get_depths_to_rows(self,timestep):
        rows = [[timestep, node, (depth - self.original_min[node]).item(), self.pos[node][0], self.pos[node][1]] for node, depth in self.h.items()]
        return rows

    def get_custom_depths_to_rows(self,timestep, custom_h):
        rows = [[timestep, node, (depth - self.original_min[node]).item(), self.pos[node][0], self.pos[node][1]] for node, depth in custom_h.items()]
        return rows

    def calculate_dh_transfer(self, hj, hi, link):
        """
        Function that receives the difference in head and returns the change in head
        """
        dif = hj-hi
        
        
        length = to_torch(link["normalized_length"])
        diameter= to_torch(link["normalized_geom_1"])
        
        dif =       torch.reshape(dif, (1, 1))
        length =    torch.reshape(length, (1, 1))
        diameter =  torch.reshape(diameter, (1, 1))



        if abs(dif)>1:
            print(dif)
        x = torch.cat([dif, length, diameter], dim=1) 
        
        ans = self.q_transfer_ANN(x)
        
        return ans

    def calculate_dh_runoff(self,runoff):
        runoff = to_torch(runoff)
        runoff = torch.reshape(runoff, (1, 1))
        ans = self.q_runoff_ANN(runoff)
        
        return ans
        

def is_giver_manhole_dry(hi, hi_min, hj, hj_min):
    ans = (hi == hi_min and hj < hi) or (hj == hj_min and hi < hj)
    return ans





#-----------------------------------------------------------------
#Governing functions
#-----------------------------------------------------------------

def q_rain(i, A_catch, w):
    
    q = w * i * A_catch
    
    return q

def q_dwf(basevalue, hourly_mult, w):
    
    q = basevalue * hourly_mult * w
    
    return q


def pump_curve(m, b, depth):
    
    q_pump = max(b - m*depth**2, 0)
    
    return(q_pump)


def pump_value(level, invert, prev_state):
    
    pump_curve_m = 0.2
    pump_curve_b = 0.1
    
    on_level = 0.2 #1.0
    off_level = 0.1  #0.5
    
    assert on_level>off_level
    depth = level-invert
    
    pump_capacity = 0.4#pump_curve(pump_curve_m, pump_curve_b, depth) #0.4
    
    if depth > on_level:
        value = pump_capacity
        state = 1 #pump is on
    elif depth < off_level:
        value = 0
        state = 0 #pump is off
    
    else:
        if prev_state ==0:
            value = 0
            state = 0 #pump is off
        else:
            value = pump_capacity
            state = 1 #pump is on
    return value, state


def dummy():
    print('dummy')