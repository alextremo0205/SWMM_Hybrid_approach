import os
import yaml
from yaml.loader import SafeLoader

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pygit2 import Repository

import plotly.express as px
import plotly.graph_objects as go

from libraries.SWMM_Simulation import SWMMSimulation


def print_current_git_branch():
    print("The current branch is: " + Repository('.').head.shorthand)

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

def get_heads_from_pickle(path):
    head_raw_data = get_info_from_pickle(path)
    head_raw_data.columns = head_raw_data.columns.str.replace("_Hydraulic_head", "")
    head_raw_data.columns = head_raw_data.columns.str.replace("node_", "")
    return head_raw_data


def get_runoff_from_pickle(path):
    #This code assumes that each node has a subcatchment attached to it
    #and node and subcatchment have the same id number.
    runoff_raw_data = get_info_from_pickle(path)
    runoff_raw_data.columns = runoff_raw_data.columns.str.replace("_Runoff_rate", "")
    runoff_raw_data.columns = runoff_raw_data.columns.str.replace("subcatchment_sub", "j")
    return runoff_raw_data


def get_dry_periods_index(rainfall_raw_data):
    indexes = np.array(rainfall_raw_data[rainfall_raw_data['value']==0].index)
    differences = np.diff(indexes)

    dry_periods_index = []
    single_dry_period_indexes = []
    for i, j in enumerate(differences):
        if j==1:
            single_dry_period_indexes.append(i)
        else:
            dry_periods_index.append(single_dry_period_indexes)
            single_dry_period_indexes = []
    
    return dry_periods_index


def get_rolled_out_target_hydraulic_heads(heads_timeseries):
    
    rolled_out_target_hydraulic_heads = []
    
    for i in range(len(heads_timeseries)):
        h = heads_timeseries.iloc[i,:].to_dict()
        rolled_out_target_hydraulic_heads.append(h)
        
    return rolled_out_target_hydraulic_heads


def extract_simulations_from_folders(simulations_path, inp_path, max_events=-1):
    
    list_of_simulations = os.listdir(simulations_path)
    
    if max_events == -1:
        max_events = len(list_of_simulations)
    
    inp_lines = get_lines_from_textfile(inp_path)
    G = inp_to_G(inp_lines)
    simulations =[]
    
    num_saved_events = 0
    for simulation in list_of_simulations:
        hydraulic_heads_path = '\\'.join([simulations_path,simulation,'hydraulic_head.pk'])
        runoff_path = '\\'.join([simulations_path,simulation,'runoff.pk'])
        
        heads_raw_data = get_heads_from_pickle(hydraulic_heads_path)
        runoff_raw_data = get_runoff_from_pickle(runoff_path)
        
        sim = SWMMSimulation(G, heads_raw_data, runoff_raw_data, simulation)
        simulations.append(sim)
        
        if num_saved_events>=max_events:
            break
        
        num_saved_events+= 1
    return simulations

def inp_to_G(lines):
    #Reading the headers of the inp file
    inp_dict = get_headers_from_inp(lines)
    
    #Create NetworkX graph
    G = nx.Graph()

    #Extracting the node coordinates from the inp file and saving them in the nx graph
    # Nodes ---------------------------------------------------------------------------------------------
    nodes_coordinates = get_nodes_coordinates(lines, inp_dict)
    nodes_elevation = get_nodes_elevation(lines, inp_dict)

    nodes_names = list(nodes_coordinates.keys())
    nodes_names_dict = dict(zip(nodes_names, nodes_names))
    G.add_nodes_from(nodes_names)
    
    nx.set_node_attributes(G, nodes_coordinates, "pos")
    nx.set_node_attributes(G, nodes_elevation, "elevation")
    nx.set_node_attributes(G, nodes_names_dict, 'name_nodes')
    
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
    
        for i in range(inp_dict['[PUMPS]\n']+3, inp_dict['[ORIFICES]\n']-1):
            # print(lines[i].split())
            pump = lines[i].split()[:-4]
            G.add_edge(pump[1],pump[2])
            pumps.append(lines[i])
    
    except Exception as e:
        print("The file does not have "+str(e))

    #Orifices----------------------------------------------------------------------------
    orifices = []

    try:
        end = inp_dict['[ORIFICES]\n']-1
        
        for i in range(inp_dict['[ORIFICES]\n']+3+1, inp_dict['[WEIRS]\n']-1):
            orifice = lines[i].split()[:-4]
            G.add_edge(orifice[1],orifice[2])
            orifices.append(lines[i])
    
    except Exception as e:
        print("The file does not have "+str(e))

    #Weirs----------------------------------------------------------------------------
    weirs = []

    try:
        end = inp_dict['[WEIRS]\n']-1
        
        
        for i in range(inp_dict['[WEIRS]\n']+3+1, inp_dict['[XSECTIONS]\n']-1):
        
            weir = lines[i].split()[:-4]
            G.add_edge(weir[1],weir[2])
            weirs.append(lines[i])
        
    except Exception as e:
        print("The file does not have "+str(e))

    return(G)

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
            print('The file does not have '+type_of_node)
    return nodes_elevation

def get_elevation_from_type(type_of_node, lines, inp_dict):
    nodes_elevation={}
    index = inp_dict[type_of_node]+3
    line = lines[index]
    while line != '\n':
        if ';' not in line:
            name, elevation = line.split()[0], line.split()[1]
            nodes_elevation[name] = float(elevation)
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

            edge_attributes['name_conduits']= l_split[0]
            edge_attributes['length']       = float(l_split[3])
            edge_attributes['roughness']    = float(l_split[4])
            edge_attributes['in_offset']    = float(l_split[5])
            edge_attributes['out_offset']   = float(l_split[6])

            
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

            x_sections_attributes['conduit_shape'] =  l_split[1]
            x_sections_attributes['geom_1'] = float(l_split[2])
            x_sections_attributes['geom_2'] = float(l_split[3])
            #It can continue, but I don't use the rest of the values

            x_sections[l_split[0]]=x_sections_attributes
        index+=1
        line = lines[index]
    return x_sections

def get_conduits_phonebook(G):
    name_dict = nx.get_edge_attributes(G, 'name_conduits')
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def head_to_unnormalized_depth(head, normalizer, ref_window):
    return normalizer.unnormalize_heads(head-ref_window.norm_elev.reshape(-1))-normalizer.min_h

    
def get_all_windows_from_list_simulations(simulations, steps_ahead):
    windows = []
    for sim in simulations:
        windows += sim.get_all_windows(steps_ahead = steps_ahead)
    return windows

def load_windows(windows_path):
    with open(windows_path, 'rb') as handle:
        windows = pickle.load(handle)
    print('Using loaded windows from: ', windows_path)
    return windows

def save_pickle(variable, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable, handle)