import yaml
import numpy as np
import pandas as pd
import datetime
import calendar

import plotly.graph_objects as go
import plotly.express as px

# import pickle
import swmmtoolbox.swmmtoolbox as swm
# import subprocess

import networkx as nx


def import_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''
    with open(config_file) as f:
        data = yaml.safe_load(f)
        networks = data['dataset_names']
        n_samples = data['n_samples']
       
        d_netw = {}
        
    return networks, n_samples

# Tabulator ----------------------------------------------------------------------------------------------
# From C:\Users\agarzondiaz\surfdrive\Year 2\Paper 2\PySWMM\utils.py
def tabulator(list_of_strings):
  new_list = []
  for i in list_of_strings:
    new_list.append(i)
    new_list.append('\t')
  new_list[-1]= "\n"
  return(new_list)


# From rain generator code -------------------------------------------------------------------------------
#C:\Users\agarzondiaz\surfdrive\Year 2\Paper 2\rain_generator.py
#Time - dates conversor
def conv_time(time):
  time/=60
  hours = int(time)
  minutes = (time*60) % 60
  # seconds = (time*3600) % 60
  return("%d:%02d" % (hours, minutes))

# Modified function. Original taken from https://pyhyd.blogspot.com/2017/07/alternating-block-hyetograph-method.html
def altblocks(idf,dur,dt):
    aDur = np.arange(dt,dur+dt,dt)    # in minutes
    aInt = idf['A']/(aDur**idf['n'] + idf['B'])  # idf equation - in mm/h for a given return period
    aDeltaPmm = np.diff(np.append(0,np.multiply(aInt,aDur/60.0))) #Duration: min -> hours
    aOrd=np.append(np.arange(1,len(aDur)+1,2)[::-1],np.arange(2,len(aDur)+1,2))
    prec = np.asarray([aDeltaPmm[x-1] for x in aOrd])
    prec_str = list(map(str, np.round(prec,2)))
    aDur_time = list(map(conv_time, aDur))
    aDur_str = list(map(str, aDur_time))
    
    aAltBl = dict(zip(aDur_str, prec_str))

    return aAltBl


def rain_blocks(values, durations, dt):
    # values = [10, 20, 30, 0]
    values = np.array(values)
    # durations = [20, 20, 20, 40]
    durations = np.array(durations)
    dt = 5
    dur = np.sum(durations)
    aDur = np.arange(dt,dur+dt,dt)    # in minutes
    aDur_time = list(map(conv_time, aDur))


    repetitions = np.int_(durations/dt)

    prec = np.repeat(values, repetitions)
    prec_str = list(map(str, np.round(prec,2)))
    aDur_str = list(map(str, aDur_time))
    ans = dict(zip(aDur_str, prec_str))
    return ans


# Generator of inp-readable lines for the rainfalls
def new_rain_lines(rainfall_dict, name_new_rain = 'name_new_rain'):

  new_lines_rainfall = []
  
  for key, value in rainfall_dict.items():
    year = key[:4]
    month = key[4:6]
    day = key[6:8]
    date =  month + '/' + day + '/' + year #Month / day / year because of the inp format
     
    time = key[8:10]+":"+key[10:12]

    new_lines_rainfall.append(tabulator([name_new_rain, date, time, str(value*0.01)]))

  return new_lines_rainfall






def inp_to_G(lines):
    #Reading the headers of the inp file
    inp_dict = dict()

    inp_dict = {line:number for (number,line) in enumerate(lines) if line[0] == "["}

    #Create NetworkX graph
    G = nx.Graph()

    #Extracting the node coordinates from the inp file and saving them in the nx graph
    # Nodes ---------------------------------------------------------------------------------------------
    points = []
    node_names = []
    # with open(working_inp) as f:
    #     lines = f.readlines()
    for i in range(inp_dict['[COORDINATES]\n']+3, inp_dict['[VERTICES]\n']-1): #TO-DO: this can be turned into a while loop
        point = lines[i].split()
        name_of_node = point[0]
        node_names.append(name_of_node)
        G.add_node(name_of_node, pos=(float(point[1]),float(point[2])) )
        points.append(lines[i])


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
    
    
    return(G, node_names)



#Formatter for rain text lines --------------------------------------------------------------------------------
#Taken from Colab: SWMM - INP Reader and INP Modifier.ipynb
def new_rain_lines(rainfall_dict, name_new_rain = 'name_new_rain', day='1/1/2019'):

  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    new_lines_rainfall.append(tabulator([name_new_rain, day, key, str(value)])) #STA01  2004  6  12  00  00  0.12


    # new_lines_rainfall.append(tabulator([name_new_rain, day, key, str(value)])) #before

  return new_lines_rainfall


def new_rain_lines_dat(rainfall_dict, name_new_rain = 'name_new_rain', day='1', month = '1', year = '2019'):

  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    hour, minute = key.split(':')
    new_lines_rainfall.append(tabulator([name_new_rain, year, month, day, hour, minute, str(value)])) #STA01  2004  6  12  00  00  0.12


    # new_lines_rainfall.append(tabulator([name_new_rain, day, key, str(value)])) #before

  return new_lines_rainfall

def new_rain_lines_real(rainfall_dict, name_new_rain = 'name_new_rain'):#, day='1', month = '1', year = '2019'):
  
  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    year = key[:4]
    month = key[4:6]
    day = key[6:8]
    hour, minute = key[8:10], key[10:12]
    new_lines_rainfall.append(tabulator([name_new_rain, year, month, day, hour, minute, str(value)])) #STA01  2004  6  12  00  00  0.12


    # new_lines_rainfall.append(tabulator([name_new_rain, day, key, str(value)])) #before

  return new_lines_rainfall


# def datetime_range(start, end, delta):
#     current = start
#     while current < end:
#         yield current
#         current += delta

 
# def keep_time():
#   # keep_times stores the used time stamps.
#   keep_times =[]
#   for month in range(1,13):
      
#       end_day = calendar.monthrange(2014, month)[1]
#       date_gen = datetime_range(datetime(2014, month, 1, 0), datetime(2014, month, end_day, 23, 55), timedelta(minutes=5))
      
#       for i in date_gen:
#           time_i = [i.year, i.month, i.day, i.hour, i.minute] #, date_gen.day, date_gen.hour, date_ gen.minute]
#           time_i = ["{0:0=2d}".format(value) for value in time_i] 
#           time_i = "".join(time_i)
#           keep_times.append(time_i)

#   return(keep_times)

# Plotting --------------------------------------------------------------------------------------------------------
def hietograph(rain):
  try:
    rain_times = [datetime.datetime.strptime(i,'%H:%M') for i in rain.keys()]
  except Exception:
    rain_times = [datetime.datetime.strptime(i,'%Y%m%d%H%M') for i in rain.keys()]

  rain_values = [float(i) for i in rain.values()]

  df = pd.DataFrame(dict(
      date=rain_times,
      value=rain_values
  ))
  bar = go.Bar(x=rain_times, y = rain_values) #px.bar(df, x='date', y="value")
  return bar




#From C:\Users\agarzondiaz\surfdrive\Year 2\Paper 2\Temporal approx\output_temporal_outfalls.py
def extract_info_inp(lines, line_where, header, names = [], elevation =[]):
    phase = 3
    c_line = lines[line_where[header]+phase]
    
    while c_line != "\n":
        if c_line[0]!= ";":
            names.append(c_line.split()[0])
            elevation.append(c_line.split()[1])
        phase+=1
        c_line = lines[line_where[header]+phase]
    return names, elevation



# Classes ------------------------------------------------------------------------------------------------------