import networkx as nx


# needed weights for the function (first attempts state)
# weight_in= 0.65
# weight_out=0.65


# weight_rain = 0.0125 #0.0125
# weight_dwf  = 0.05

# dt = 1

def difussion(G, original_h0):#, time, prev_state_pump):
    weight_in, weight_out = 0.65, 0.65
    constant =0.005
    original_min = nx.get_node_attributes(G, 'elevation')
    
    
    new_h0= {}
    nodes_pump = [] #Nodes that have a pump
    outfalls = ['j_90552', 'j_90431']    #Nodes that are outfalls
    #Updates each node
    for node, hi in original_h0.items():    
        #links connected to that node
        hi_min = float(original_min[node])
        total_dh = 0
        #Updates for each of the neighbors
        for _, neigh in G.edges(node):
            #Extract current heads
            hj = original_h0[neigh]
            hj_min = float(original_min[neigh])

            #Extract edge attributes
            length = G.edges[node, neigh]["length"]
            diameter= G.edges[node, neigh]["geom_1"]

            #if the heads are in their minimum, they cannot give water
            if (hi == hi_min and hj<hi) or (hj==hj_min and hi<hj):
                q_transfer =0
            else:
                #In case there is valid gradient, this function calculates the change in head
                q_transfer = odd_transform(hj, hi, float(length), float(diameter), weight_in, weight_out)
            
            # The total change is the influence of all the neighbors
            total_dh += q_transfer - constant #(q_transfer + q_rain(rain[time], original_A_catch[node], weight_rain) + q_dwf(original_basevalue_dwf[node], dwf_hourly[time%24], weight_dwf))*dt 
        # print(total_dh)
        #The new head cannot be under the minimimum level. Careful!! A node may be giving more than it has to offer.
        new_h0[node] = max(hi+total_dh, hi_min)

    for node_outfalls in outfalls:
        new_h0[node_outfalls]=float(original_min[node_outfalls])


    return new_h0#, pump_state



#-----------------------------------------------------------------
#Governing functions
#-----------------------------------------------------------------
def q_interchange_in(dh, L, d, w_in):
    """
    This function evaluates the magnitude that the difference in head has in the next head. How water flows.
    """

    num = (d**(5/2))*(dh**0.5) #**2.0)
    den = L**0.5
    q = w_in*(num/den)
    
    return q


def q_interchange_out(dh, L, d, w_out):
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


# def psi(val, length, diameter):
#     """
#     This function evaluates the magnitude that the difference in head has in the next head. How water flows.
#     """
#     delta_t = 0.05 #related to the timestep
#     # length = 0.2
#     # diameter = 0.4
    
#     d5 = diameter**5
    
#     return delta_t*(val)**0.5

def odd_transform(hj, hi, length, diameter, weight_in, weight_out):
    """
    Function that receives the difference in head and returns the change in head
    """
    
    x = hj-hi
    # odd_t = np.sign(x)*q_interchange(abs(x), length, diameter, weight)
    
    if x <= 0:
        odd_t = q_interchange_out(abs(x), length, diameter, weight_out)
    else:
        odd_t = q_interchange_in(abs(x), length, diameter, weight_in)
    
    return odd_t



# def rain_runoff(in_rain, a_catch):
#     weight =0.1
#     runoff = weight * in_rain * a_catch
#     return runoff


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


