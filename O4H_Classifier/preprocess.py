import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('o4h_all_events.csv', delimiter=',', usecols=['Time', 'ItemName', 'Value'])

item_names = set(df['ItemName']) - {'label'}
line_count = len(set(df['Time']))
# print(line_count)
# print(len(item_names))
new_df =  pd.DataFrame(np.empty((line_count, len(item_names))),columns=list(item_names), index=sorted(list(set(pd.to_datetime(df['Time'], infer_datetime_format=True)))))
new_df[:] = np.nan
# new_df.insert(0, 'location', np.nan)
new_df.insert(0, 'activity', np.nan)

# weekday_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
# new_df.insert(0, 'weekday',  new_df.index.strftime("%a").map(weekday_mapping))
# new_df.insert(1, 'hour', new_df.index.strftime("%H"))
# new_df.insert(2, 'minute', new_df.index.strftime("%M"))
# new_df.insert(3, 'location', np.nan)
# new_df.insert(4, 'activity', np.nan)
# new_df.insert(5, 'activity_duration', np.nan)
# new_df.insert(6, 'next_activity', np.nan)

time_dict = {item:jtem for item, jtem in zip(new_df.index.strftime('%Y-%m-%d %H:%M:%S'), new_df.index)}
label = ''
# location = ''

for index in df.index:
  if df['ItemName'][index].lower() == "label" and df['Value'][index].lower().startswith("stop"):
    label = 'interActivity'
    # location = ''
  elif df['ItemName'][index].lower() == "label" and df['Value'][index].lower().startswith("start"):
    label = df['Value'][index].split(":")[1]
    # location = df['Value'][index].split(":")[1].split("|")[0]
      
  new_df['activity'][time_dict[df['Time'][index]]] = label
  # new_df['location'][time_dict[df['Time'][index]]] = location
  if df['ItemName'][index].lower() != "label":
    new_df[df['ItemName'][index]][time_dict[df['Time'][index]]] = df['Value'][index]

new_df.ffill(axis = 0, inplace=True)
new_df.bfill(axis = 0, inplace=True)
new_df.replace([r'^OFF$', r'^ON$'], [0, 1], regex=True, inplace=True)
new_df.replace([r'^OPEN$', r'^CLOSED$'], [0, 1], regex=True, inplace=True)

global_sensors_boolean_list = new_df.columns.str.startswith('global')
global_sensors_index = np.where(global_sensors_boolean_list)[0]
new_df = new_df.loc[:,~global_sensors_boolean_list]

# new_df = new_df[new_df['activity'] != ""]    //eleminating inter activity timestamps

# location_dict = {jtem:item for item, jtem in enumerate(sorted(set(new_df['location'])))}   
# new_df['location'] = new_df['location'].replace(location_dict)

# new_df["next_activity"] = new_df["activity"].shift(-1)
# new_df["activity_duration"] = new_df.index 
# for i in range(len(new_df)-2,-1,-1):
#     if new_df.at[new_df.index[i], "activity"] == new_df.at[new_df.index[i+1], "activity"]:
#         new_df.at[new_df.index[i], "next_activity"] = new_df.at[new_df.index[i+1], "next_activity"]
#         new_df.at[new_df.index[i], "activity_duration"] = new_df.at[new_df.index[i+1], "activity_duration"]
   
# new_df["activity_duration"] = new_df["activity_duration"] - new_df.index + pd.Timedelta(1, unit='m')     
# new_df["activity_duration"] = new_df["activity_duration"] / pd.Timedelta(minutes=1)

activity_dict = {jtem:item for item, jtem in enumerate(sorted(set(new_df['activity'])))}   
new_df.replace(activity_dict, inplace=True)

# new_df.dropna(subset=['next_activity'], inplace=True)
# ohe = OneHotEncoder()
# transformed = ohe.fit_transform(new_df['next_activity'].values.reshape(-1, 1))
# new_df[ohe.categories_[0]] = transformed.toarray() 

new_df.to_csv('dataset.csv')

