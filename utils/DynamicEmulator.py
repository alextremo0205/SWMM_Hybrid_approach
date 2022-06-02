import networkx as nx
import utils.head_change_utils as utils

w_in =0.65
w_out = 0.65
constant = 0.005
nodes_outfalls = ['j_90552', 'j_90431']

class DynamicEmulator:
    
    def __init__(self, inp_path, initial_h0):
        assert type(initial_h0) == dict, "Depths should be a dictionary"

        self.inp_lines = utils.get_lines_from_textfile(inp_path)
        self.G = utils.inp_to_G(self.inp_lines)
        
        self.original_min = nx.get_node_attributes(self.G, 'elevation')
        self.original_A_catch = nx.get_node_attributes(self.G, 'area_subcatchment')
        self.pos = nx.get_node_attributes(self.G, 'pos')

        self.h = initial_h0
        
    def draw_nx_layout(self):
        nx.draw(self.G, pos = self.pos, node_size=15)

    def get_depths_to_pd(self,timestep):
        rows = [[timestep, node, depth - float(self.original_min[node]), self.pos[node][0], self.pos[node][1]] for node, depth in self.h.items()]
        return rows

    def get_h(self):
        return self.h

    def set_h(self, new_h):
        self.h = new_h

    def update_h(self):#, time, prev_state_pump):
        
        new_h0= {}
        nodes_pump = []                         #Nodes that have a pump
        outfalls = nodes_outfalls       #Nodes that are outfalls

        for node, hi in self.h.items():    #links connected to that node
            
            hi_min = float(self.original_min[node])
            
            total_dh = 0
            for _, neigh in self.G.edges(node):      #Updates for each of the neighbors
                hj = self.h[neigh]
                hj_min = float(self.original_min[neigh])
                
                #Extract edge attributes
                link = self.G.edges[node, neigh]

                if is_giver_manhole_dry(hi, hi_min, hj, hj_min):     #if the heads are in their minimum, they cannot give water
                    q_transfer =0
                else:                                                   #In case there is valid gradient, this function calculates the change in head
                    q_transfer = calculate_q_transfer(hj, hi, link)
                
                total_dh += q_transfer - constant #(q_transfer + q_rain(rain[time], original_A_catch[node], weight_rain) + q_dwf(original_basevalue_dwf[node], dwf_hourly[time%24], weight_dwf))*dt 
            
            new_h0[node] = max(hi+total_dh, hi_min) #The new head cannot be under the minimimum level. Careful!! A node may be giving more than it has to offer.

        for node_outfalls in outfalls:
            new_h0[node_outfalls]=float(self.original_min[node_outfalls])

        self.h = new_h0


def is_giver_manhole_dry(hi, hi_min, hj, hj_min):
    ans = (hi == hi_min and hj < hi) or (hj == hj_min and hi < hj)
    return ans


#-----------------------------------------------------------------
#Governing functions
#-----------------------------------------------------------------
def calculate_q_transfer(hj, hi, link):
    """
    Function that receives the difference in head and returns the change in head
    """
    dif = hj-hi
    
    length = float(link["length"])
    diameter= float(link["geom_1"])

    if dif <= 0:
        odd_t = q_interchange_out(abs(dif), length, diameter)
    else:
        odd_t = q_interchange_in(abs(dif), length, diameter)
    
    return odd_t


def q_interchange_in(dh, L, d):
    """
    This function evaluates the magnitude that the difference in head has in the next head. How water flows.
    """

    num = (d**(5/2))*(dh**0.5) #**2.0)
    den = L**0.5
    q = w_in*(num/den)
    
    return q


def q_interchange_out(dh, L, d):
    """
    This function evaluates the magnitude that the difference in head has in the next head. How water flows.
    """

    num = (d**(5/2))*(dh**0.5) #**2.0)
    den = L**0.5
    q = w_out*(num/den)
    
    return -q


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