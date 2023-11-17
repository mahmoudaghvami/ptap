import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# df = pd.read_csv('O4H_Classifier/o4h_all_events.csv', delimiter=',', usecols=['Time', 'ItemName', 'Value'])
# df['Time'] = pd.to_datetime(df['Time']).dt.round('min')
# df.to_csv('o4h_all_events_minute.csv', index=False)

df = pd.read_csv('O4H_Classifier/o4h_all_events_minute.csv', delimiter=',', usecols=['Time', 'ItemName', 'Value'])
item_names = set(df['ItemName']) - {'label'}
line_count = len(set(df['Time']))
new_df =  pd.DataFrame(np.empty((line_count, len(item_names))),columns=list(item_names), index=sorted(list(set(pd.to_datetime(df['Time'], infer_datetime_format=True)))))
new_df[:] = np.nan
new_df.insert(0, 'activity', np.nan)


time_dict = {item:jtem for item, jtem in zip(new_df.index.strftime('%Y-%m-%d %H:%M:%S'), new_df.index)}
label = ''

for index in df.index:
  if df['ItemName'][index].lower() == "label" and df['Value'][index].lower().startswith("stop"):
    label = 'interActivity'
  elif df['ItemName'][index].lower() == "label" and df['Value'][index].lower().startswith("start"):
    label = df['Value'][index].split(":")[1]
      
  new_df['activity'][time_dict[df['Time'][index]]] = label
  if df['ItemName'][index].lower() != "label":
    new_df[df['ItemName'][index]][time_dict[df['Time'][index]]] = df['Value'][index]

new_df.ffill(axis = 0, inplace=True)
new_df.bfill(axis = 0, inplace=True)
new_df.replace([r'^OFF$', r'^ON$'], [0, 1], regex=True, inplace=True)
new_df.replace([r'^OPEN$', r'^CLOSED$'], [0, 1], regex=True, inplace=True)

global_sensors_boolean_list = new_df.columns.str.startswith('global')
global_sensors_index = np.where(global_sensors_boolean_list)[0]
new_df = new_df.loc[:,~global_sensors_boolean_list]

activity_dict = {jtem:item for item, jtem in enumerate(sorted(set(new_df['activity'])))}   
new_df.replace(activity_dict, inplace=True)

# save activity_dict in a csv file
activity_dict_df = pd.DataFrame(list(activity_dict.items()),columns = ['activity','activity_id'])
# print activity_dict
print(activity_dict_df) 

new_df.to_csv('check_dataset.csv')
