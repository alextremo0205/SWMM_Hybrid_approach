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

    subcathments_attributes = get_subcatchments(lines, inp_dict)
    nx.set_node_attributes(G, subcathments_attributes)

    #Edges-----------------------------------------------------------------------------------------------
    ##Conduits
    conduits = get_conduits_from_inp_lines(lines, inp_dict)
    G.add_edges_from(conduits)
    nx.set_edge_attributes(G, conduits)

    x_sections = get_x_sections(lines, inp_dict)
    phonebook = get_conduits_phonebook(G)
    new_x_sections = change_name_xsections_to_edge_tuple(phonebook, x_sections)
    nx.set_edge_attributes(G, new_x_sections)
    

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




def get_conduits_from_inp_lines(lines, inp_dict):
    conduits = {}
    
    index = inp_dict['[CONDUITS]\n']+3
    line = lines[index]
    while line != '\n':
        if ';' not in line:
            edge_attributes ={}
            
            l_split = line.split()
            source_node, destiny_node = l_split[1], l_split[2]

            edge_attributes['name'] = l_split[0]
            edge_attributes['length'] = l_split[3]
            edge_attributes['roughness'] = l_split[4]
            edge_attributes['in_offset'] = l_split[5]
            edge_attributes['out_offset'] = l_split[6]

            
            conduits[(source_node, destiny_node)] = edge_attributes

        index+=1
        line = lines[index]
    
    return conduits


def get_x_sections(lines, inp_dict):
    x_sections={}
    index = inp_dict['[XSECTIONS]\n']+3
    line = lines[index]
    
    while line != '\n':
        if ';' not in line:
            x_sections_attributes ={}
            l_split = line.split()

            x_sections_attributes['shape'] = l_split[1]
            x_sections_attributes['geom_1'] = l_split[2]
            x_sections_attributes['geom_2'] = l_split[3]
            #It can continue, but I don't use the rest of the values

            x_sections[l_split[0]]=x_sections_attributes
        index+=1
        line = lines[index]
    return x_sections

def get_conduits_phonebook(G):
    name_dict = nx.get_edge_attributes(G, 'name')
    name_tuple = {name_dict[k] : k for k in name_dict}
    return name_tuple

def change_name_xsections_to_edge_tuple(phonebook, x_sections):
    new_x_sections = {phonebook[k]:value for k, value in x_sections.items()}   
    return new_x_sections


def get_subcatchments(lines, inp_dict):
    subcathments={}
    index = inp_dict['[SUBCATCHMENTS]\n']+3
    line = lines[index]
    
    while line != '\n':
        if ';' not in line:
            subcatchment_attributes ={}
            l_split = line.split()
            
            subcatchment_attributes['name_subcatchment'] = l_split[0]
            subcatchment_attributes['raingage'] = l_split[1]
            #The name of the outlet goes as key in the dictionary.
            subcatchment_attributes['area_subcatchment'] = l_split[3]
            #It can continue, but I don't use the rest of the values

            subcathments[l_split[2]]=subcatchment_attributes
        index+=1
        line = lines[index]
    return subcathments

# def get_subcatchments_phonebook(G):
#     name_dict = nx.get_node_attributes(G, 'name')
#     name_tuple = {name_dict[k] : k for k in name_dict}
#     return name_tuple


# Plotting ---------------------------------------------------

def get_scatter_trace(x, y):
    trace = go.Scatter(x=x, y=y, mode='markers')
    return trace




#-----------------------------------------------------------------
#-----------------------------------------------------------------




##
def is_it_ready():
    return True

