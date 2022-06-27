import os
import yaml
import pickle
import datetime
import subprocess
import numpy as np

import plotly.graph_objects as go

import swmmtoolbox.swmmtoolbox as swm

def import_config_from_yaml(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''
    with open(config_file) as f:
        data = yaml.safe_load(f)
        
    return data

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

def get_multiple_alt_blocks_rainfalls(n_rainfalls):
  rainfalls = []
  dt = 5
  for i in range(n_rainfalls):
    rand_A = np.random.randint(15000, 25000)
    rand_B = np.random.uniform()*(20) + 30
    rand_dur = np.random.randint(4,36)*dt

    random_idf ={ 'A':rand_A,
                  'B':rand_B,
                  'n':1.09}

    specs_rain = ['Synt_'+str(i), random_idf, rand_dur]
    one_random_rain = altblocks(random_idf, dur = rand_dur, dt=dt)
    # rainfalls.append((specs_rain, one_random_rain))
    rainfalls.append(one_random_rain)
    
  return rainfalls





#Time - dates conversor
def conv_time(time):
  time/=60
  hours = int(time)
  minutes = (time*60) % 60
  # seconds = (time*3600) % 60
  return("%d:%02d" % (hours, minutes))

# Modified function. Original taken from https://pyhyd.blogspot.com/2017/07/alternating-block-hyetograph-method.html
def altblocks(idf,dur,dt, dur_padding=60):
    
    value_padding = np.zeros(int(dur_padding/dt))
    aPad = np.arange(dt,dur_padding+dt,dt)
    aPostPad = np.arange(dur_padding+dur+dt,2*dur_padding+dur+dt,dt)

    aDur = np.arange(dt,dur+dt,dt)    # in minutes
    aInt = idf['A']/(aDur**idf['n'] + idf['B'])  # idf equation - in mm/h for a given return period
    aDeltaPmm = np.diff(np.append(0,np.multiply(aInt,aDur/60.0))) #Duration: min -> hours
    aOrd=np.append(np.arange(1,len(aDur)+1,2)[::-1],np.arange(2,len(aDur)+1,2))
    
    prec = np.asarray([aDeltaPmm[x-1] for x in aOrd])


    aDur = aDur+dur_padding
    prec = np.concatenate((value_padding, prec, value_padding))
    aDur = np.concatenate((aPad, aDur, aPostPad))

    prec_str = list(map(str, np.round(prec,2)))
    
    aDur_time = list(map(conv_time, aDur))
    aDur_str = list(map(str, aDur_time))
    
    aAltBl = dict(zip(aDur_str, prec_str))

    return aAltBl


def rain_blocks(values, durations, dt):
    values = np.array(values)
    durations = np.array(durations)
    dur = np.sum(durations)
    aDur = np.arange(dt,dur+dt,dt)    # in minutes
    aDur_time = list(map(conv_time, aDur))


    repetitions = np.int_(durations/dt)

    prec = np.repeat(values, repetitions)
    prec_str = list(map(str, np.round(prec,2)))
    aDur_str = list(map(str, aDur_time))
    ans = dict(zip(aDur_str, prec_str))
    return ans



def get_max_from_raindict(dict_rain):
    list_of_values = list(dict_rain.values())
    return np.array(list_of_values).max()




# Generator of inp-readable lines for the rainfalls
def new_rain_lines(rainfall_dict, name_new_rain = 'name_new_rain'):

  new_lines_rainfall = []
  
  for key, value in rainfall_dict.items():
    year = key[:4]
    month = key[4:6]
    day = key[6:8]
    date =  month + '/' + day + '/' + year #Month / day / year (because of the inp format)
     
    time = key[8:10]+":"+key[10:12]

    new_lines_rainfall.append(tabulator([name_new_rain, date, time, str(value*0.01)]))

  return new_lines_rainfall



#Formatter for rain text lines --------------------------------------------------------------------------------
def create_datfiles(rainfalls, rainfall_dats_directory, identifier, isReal):
  for idx, single_event in enumerate(rainfalls):
    
    if isReal:
      string_rain = ["".join(i) for i in get_new_rain_lines_real(single_event, name_new_rain = 'R1') ]  
    else:
      string_rain = ["".join(i) for i in new_rain_lines_dat(single_event, name_new_rain = 'R1') ]

    filename = '\\'+identifier+'_'+ str(idx)+'.dat'
    with open(rainfall_dats_directory+filename, 'w') as f:
        f.writelines(string_rain)




#Taken from Colab: SWMM - INP Reader and INP Modifier.ipynb
def new_rain_lines(rainfall_dict, name_new_rain = 'name_new_rain', day='1/1/2019'):

  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    new_lines_rainfall.append(tabulator([name_new_rain, day, key, str(value)])) #STA01  2004  6  12  00  00  0.12

  return new_lines_rainfall


def new_rain_lines_dat(rainfall_dict, name_new_rain = 'name_new_rain', day='1', month = '1', year = '2019'):

  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    hour, minute = key.split(':')
    new_lines_rainfall.append(tabulator([name_new_rain, year, month, day, hour, minute, str(value)])) #STA01  2004  6  12  00  00  0.12


  return new_lines_rainfall

def get_new_rain_lines_real(rainfall_dict, name_new_rain = 'name_new_rain'):#, day='1', month = '1', year = '2019'):
  
  new_lines_rainfall = []

  for key, value in rainfall_dict.items():
    year = key[:4]
    month = key[4:6]
    day = key[6:8]
    hour, minute = key[8:10], key[10:12]
    new_lines_rainfall.append(tabulator([name_new_rain, year, month, day, hour, minute, str(value)])) #STA01  2004  6  12  00  00  0.12


  return new_lines_rainfall



def get_lines_from_textfile(path):
  with open(path, 'r') as fh:
    lines_from_file = fh.readlines()
  return lines_from_file


def run_SWMM(inp_path, rainfall_dats_directory, simulations_path):
  list_of_rain_datfiles = os.listdir(rainfall_dats_directory)

  for event in list_of_rain_datfiles:
      rain_event_path = rainfall_dats_directory + '\\' + event
      
      inp =  get_lines_from_textfile(inp_path)
      dat =  get_lines_from_textfile(rain_event_path)
      

      for ln, line in enumerate(inp):
          splitted_line_dat = dat[0].split('\t')
          new_date = ''.join([splitted_line_dat[2], '/',splitted_line_dat[3],'/', splitted_line_dat[1]])
          splitted_line_dat_last = dat[-1].split('\t')
          new_last_date = ''.join([splitted_line_dat_last[2], '/',splitted_line_dat_last[3],'/', splitted_line_dat_last[1]])

          new_last_time = ''.join([splitted_line_dat_last[4], ':',splitted_line_dat_last[5], ':', '00'])
          if 'START_DATE' in line:
              inp[ln] = line.replace(line.split()[-1], new_date)
              
          elif 'END_DATE' in line:
              inp[ln] = line.replace(line.split()[-1], new_last_date)
              
          elif 'END_TIME' in line:
              inp[ln] = line.replace(line.split()[-1], new_last_time)
              

          elif 'PLACEHOLDER1' in line:
              inp[ln] = line.replace('PLACEHOLDER1', '\\'.join((rainfall_dats_directory, event)))
              # print(inp[ln])

              
          # elif 'PLACEHOLDER2' in line:
          #     inp[ln] = line.replace('PLACEHOLDER2', '\\'.join((rainfall_dats_directory, event, 'datfiles',
          #                                                     'rain' +str(k[1]) + '.dat')))
          # elif 'PLACEHOLDER3' in line:
          #     inp[ln] = line.replace('PLACEHOLDER3', '\\'.join((rainfall_dats_directory, event, 'datfiles',
          #                                                     'rain' +str(k[2]) + '.dat')))

      nf = '\\'.join((simulations_path, event.replace('.dat', '')))
      os.mkdir(nf)

      with open(nf + '\\model.inp', 'w') as fh:
          for line in inp:
              fh.write("%s" % line)   


      subprocess.run([r'C:\Program Files (x86)\EPA SWMM 5.1.015\swmm5.exe',
                      nf+'\\model.inp',
                      nf+'\\model.rpt',
                      nf+'\\model.out'])

      with open(nf + '\\'+ event, 'w') as fh:
          for line in dat:
              fh.write("%s" % line)

def extract_and_pickle_SWMM_results(simulations_path):
  list_of_simulations = os.listdir(simulations_path)
  for sim in list_of_simulations:
      c_simulation_folder_path = '\\'.join([simulations_path, sim])
      working_out ='\\'.join([c_simulation_folder_path, 'model.out'])

      head_out_timeseries  = swm.extract(working_out, 'node,,Hydraulic_head')
      runoff_timeseries  = swm.extract(working_out, 'subcatchment,,Runoff_rate')

      with open(c_simulation_folder_path+'\\hydraulic_head.pk', 'wb') as handle:
          pickle.dump(head_out_timeseries, handle, protocol=pickle.HIGHEST_PROTOCOL)

      with open(c_simulation_folder_path+'\\runoff.pk', 'wb') as handle:
          pickle.dump(runoff_timeseries, handle, protocol=pickle.HIGHEST_PROTOCOL)




# Plotting --------------------------------------------------------------------------------------------------------
def hietograph(rain, title):
  try:
    rain_times = [datetime.datetime.strptime(i,'%H:%M') for i in rain.keys()]
  except Exception:
    rain_times = [datetime.datetime.strptime(i,'%Y%m%d%H%M') for i in rain.keys()]

  rain_values = [float(i) for i in rain.values()]

  bar = go.Bar(x=rain_times, y = rain_values) #px.bar(df, x='date', y="value")


  fig= go.Figure()
  fig.add_trace(bar)

  fig.update_layout(
      title=title,
      xaxis_title="Time (h)",
      yaxis_title="Intensity (mm/h)",
      font=dict(
          family="Times new roman",
          size=18,
      ),
      width=800, 
      height=400
  )



  return fig




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


