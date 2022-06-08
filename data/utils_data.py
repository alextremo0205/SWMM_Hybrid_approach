import yaml
import numpy as np
import pandas as pd
import datetime
import calendar

import plotly.graph_objects as go
import plotly.express as px

import swmmtoolbox.swmmtoolbox as swm


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

def get_new_rain_lines_real(rainfall_dict, name_new_rain = 'name_new_rain'):#, day='1', month = '1', year = '2019'):
  
  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    year = key[:4]
    month = key[4:6]
    day = key[6:8]
    hour, minute = key[8:10], key[10:12]
    new_lines_rainfall.append(tabulator([name_new_rain, year, month, day, hour, minute, str(value)])) #STA01  2004  6  12  00  00  0.12


    # new_lines_rainfall.append(tabulator([name_new_rain, day, key, str(value)])) #before

  return new_lines_rainfall



def get_lines_from_textfile(path):
  with open(path, 'r') as fh:
    lines_from_file = fh.readlines()
  return lines_from_file






# Plotting --------------------------------------------------------------------------------------------------------
def hietograph(rain):
  try:
    rain_times = [datetime.datetime.strptime(i,'%H:%M') for i in rain.keys()]
  except Exception:
    rain_times = [datetime.datetime.strptime(i,'%Y%m%d%H%M') for i in rain.keys()]

  rain_values = [float(i) for i in rain.values()]

  # df = pd.DataFrame(dict(
  #     date=rain_times,
  #     value=rain_values
  # ))
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