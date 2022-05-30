import yaml
from yaml.loader import SafeLoader

import pickle
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)
    return(yaml_data)

def get_lines_from_textfile(path):
  with open(path, 'r') as fh:
    lines = fh.readlines()
  return lines

def get_info_from_pickle(path):
    with open(path, 'rb') as f:
        info = pickle.load(f)
    return info


def get_rain_in_pandas(rain_path):
    rainfall_raw_data = pd.read_csv(rain_path, sep="\t", header=None)
    rainfall_raw_data.columns = ["station", "year", "month", "day",  "hour", "minute","value"]
    rainfall_raw_data = rainfall_raw_data[:-1] #Drop last row to syncronize it with heads 
    return rainfall_raw_data


def get_h0_h1_couples(heads_timeseries):
    h0_samples = []
    h1_samples = []
    for i in range(len(heads_timeseries)-1):
        h0 = heads_timeseries.iloc[i,:].to_dict()
        h1 = heads_timeseries.iloc[i+1,:].to_dict()

        h0_sample, h1_sample = h0, h1

        h0_samples.append(h0_sample)
        h1_samples.append(h1_sample)
        
    return h0_samples, h1_samples


def get_max_and_min(list_of_samples):
    
    max = np.array(list(list_of_samples[0].values())).max()
    min = np.array(list(list_of_samples[0].values())).min()

    for sample in list_of_samples:
        val_max = np.array(list(sample.values())).max()
        val_min = np.array(list(sample.values())).min()
        if val_max>max:
            max = val_max
        if val_min<min:
            min = val_min

    return max, min

def normalize_min_max(value, max,min):
    return (value - min)/(max-min)

def normalize_sample_values(list_of_samples, max = None, min = None):
    if max == None and min == None:
        max, min = get_max_and_min(list_of_samples)

    normalized_samples = []
    for sample in list_of_samples:
        normalized_sample = dict(map(lambda x: (x[0], normalize_min_max(x[1], max, min )), sample.items()))
        normalized_samples.append(normalized_sample)

    return normalized_samples





def inp_to_G(lines):
    #Reading the headers of the inp file
    inp_dict = get_headers_from_inp(lines)

    #Create NetworkX graph
    G = nx.Graph()

    #Extracting the node coordinates from the inp file and saving them in the nx graph
    # Nodes ---------------------------------------------------------------------------------------------
    # points = []
    nodes_coordinates = get_nodes_coordinates(lines, inp_dict)
    nodes_elevation = get_nodes_elevation(lines, inp_dict)

    nodes_names = list(nodes_coordinates.keys())
    G.add_nodes_from(nodes_names)
    nx.set_node_attributes(G, nodes_coordinates, "pos")
    nx.set_node_attributes(G, nodes_elevation, "elevation")


    #Edges-----------------------------------------------------------------------------------------------
    conduits = []
    try:
        end = inp_dict['[PUMPS]\n']-1
    except Exception as e:
        print("The file does not have pumps. A handled exception occured because of "+str(e))
        end =  inp_dict['[VERTICES]\n']-1


    # with open(working_inp) as f:
    # lines = f.readlines()
    i = inp_dict['[CONDUITS]\n']+3
    link = lines[i].split()[:-6]
    
    while link != []:
        G.add_edge(link[1],link[2])
        conduits.append(link)
        i +=1
        link = lines[i].split()[:-6]

    #Pumps----------------------------------------------------------------------------
    pumps = []

    try:
        end = inp_dict['[PUMPS]\n']-1
    
        # with open(working_inp) as f:
        #     lines = f.readlines()
        for i in range(inp_dict['[PUMPS]\n']+3, inp_dict['[ORIFICES]\n']-1):
            # print(lines[i].split())
            pump = lines[i].split()[:-4]
            G.add_edge(pump[1],pump[2])
            pumps.append(lines[i])
    
    except Exception as e:
        print("The file does not have pumps. A handled exception occured because of "+str(e))

    #Orifices----------------------------------------------------------------------------
    orifices = []

    try:
        end = inp_dict['[ORIFICES]\n']-1
        
        # with open(working_inp) as f:
        #     lines = f.readlines()
        for i in range(inp_dict['[ORIFICES]\n']+3+1, inp_dict['[WEIRS]\n']-1):
            # print(lines[i].split())
            orifice = lines[i].split()[:-4]
            G.add_edge(orifice[1],orifice[2])
            orifices.append(lines[i])
    
    except Exception as e:
        print("The file does not have orifices. A handled exception occured because of "+str(e))


    #Weirs----------------------------------------------------------------------------
    weirs = []

    try:
        end = inp_dict['[WEIRS]\n']-1
        
        # with open(working_inp) as f:
        # lines = f.readlines()
        for i in range(inp_dict['[WEIRS]\n']+3+1, inp_dict['[XSECTIONS]\n']-1):
            # print(lines[i].split())
            weir = lines[i].split()[:-4]
            G.add_edge(weir[1],weir[2])
            weirs.append(lines[i])
        
    except Exception as e:
        print("The file does not have weirs. A handled exception occured because of "+str(e))
    
    
    return(G)#, node_names)


def get_headers_from_inp(lines):
    inp_dict = dict()
    inp_dict = {line:number for (number,line) in enumerate(lines) if line[0] == "["}
    return inp_dict



def get_nodes_coordinates(lines, inp_dict):
    
    index = inp_dict['[COORDINATES]\n']+3
    line = lines[index]
    pos = {}
    while line != '\n':
        name_node, x_coord, y_coord = line.split()
        pos[name_node] = (float(x_coord),float(y_coord))
        index+=1
        line = lines[index]
    return pos


def get_nodes_elevation(lines, inp_dict):
    nodes_elevation={}
    types_nodes = ['[JUNCTIONS]\n', '[OUTFALLS]\n', '[STORAGE]\n']
    for type_of_node in types_nodes:
        try:
            elevations = get_elevation_from_type(type_of_node, lines, inp_dict)
            nodes_elevation.update(elevations)
        except Exception as e:
            print('There are no '+type_of_node+' in the file')
    return nodes_elevation



def get_elevation_from_type(type_of_node, lines, inp_dict):
    nodes_elevation={}
    index = inp_dict[type_of_node]+3
    line = lines[index]
    while line != '\n':
        if ';' not in line:
            name, elevation = line.split()[0], line.split()[1]
            nodes_elevation[name] = elevation
        index+=1
        line = lines[index]
    
    return nodes_elevation


# Plotting


def get_scatter_trace(x, y):
    trace = go.Scatter(x=x, y=y, mode='markers')
    return trace



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



#-----------------------------------------------------------------
#-----------------------------------------------------------------




##
def is_it_ready():
    return True

