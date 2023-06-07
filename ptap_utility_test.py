import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from art.attacks.evasion import FastGradientMethod, CarliniL0Method, SaliencyMapMethod, CarliniL2Method, DeepFool, NewtonFool, CarliniLInfMethod, ElasticNet, ProjectedGradientDescent, BasicIterativeMethod, SpatialTransformation, HopSkipJump, ZooAttack
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import math
import matplotlib.pyplot as plt
import datetime
from art.attacks.evasion import FastGradientMethod, CarliniL0Method, SaliencyMapMethod, CarliniL2Method, DeepFool, NewtonFool, CarliniLInfMethod, ElasticNet, ProjectedGradientDescent, BasicIterativeMethod, SpatialTransformation, HopSkipJump, ZooAttack
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import math
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D , LSTM
from tensorflow.python.keras import Model
from art.estimators.classification import TensorFlowV2Classifier
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(256)
tf.config.threading.set_intra_op_parallelism_threads(256)
import time
from multiprocessing import Process, Queue
import os
import csv



app1_execution_counter=0
app2_execution_counter=0
app3_execution_counter=0
app4_execution_counter=0
app5_execution_counter=0


def tap_app_runner(sensor, value):
    global app1_execution_counter
    global app2_execution_counter
    global app3_execution_counter
    global app4_execution_counter
    global app5_execution_counter
    actions = [] # list to store actions

    # app1
    if 'livingroom_temperature' in sensor:
        if float(value) > 20.0:
            app1_execution_counter += 1
            # Define action
            action_sensor = 'livingroom_AC'
            action_value = '1'
            # Append action tuple to actions list
            actions.append((action_sensor, action_value))
    
    # app2
    if 'kitchen_hood_power' in sensor:
        if float(value) > 80:
            app2_execution_counter += 1
            # Define action
            action_sensor = 'WashingMachine'
            action_value = '0'
            # Append action tuple to actions list
            actions.append((action_sensor, action_value))

    # app3
    if 'bedroom_presence' in sensor:
        if value == 'ON' or value == '1':
            app3_execution_counter += 1
            # Define action
            action_sensor = 'bedroom_light1'
            action_value = '1'
            # Append action tuple to actions list
            actions.append((action_sensor, action_value))
    
    # app4
    if 'entrance_door' in sensor:
        if value == 'OPEN' or value == '1':
            app4_execution_counter += 1
            # Define action
            action_sensor = 'enterance_light1'
            action_value = '1'
            # Append action tuple to actions list
            actions.append((action_sensor, action_value))
    
    # app5
    if 'livingroom_humidity' in sensor:
        if float(value) >= 30:
            app5_execution_counter += 1
            # Define action
            action_sensor = 'livingroom_AC'
            action_value = '1'
            # Append action tuple to actions list
            actions.append((action_sensor, action_value))
    
    if actions:
        print(actions)

    return actions

def sensor_name_converter(index):
    df = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/o4h_all_events.csv', delimiter=',', usecols=['Time', 'ItemName', 'Value'])
    #list all the unique item names except label
    item_names = set(df['ItemName']) - {'label'}
    sensor = list(item_names)[index]
    return sensor


def load_normal_data():
    skip_rows = 716416
    num_rows = 746767 - 716416 + 1
    df = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/o4h_all_events.csv', delimiter=',', usecols=['Time', 'ItemName', 'Value'], skiprows=range(1, skip_rows+1), nrows=num_rows)
    # df = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/o4h_all_events.csv', delimiter=',', usecols=['Time', 'ItemName', 'Value'])
    #list all the unique item names except label
    item_names = set(df['ItemName']) - {'label'}
    #  the number of unique 'Time' values 
    line_count = len(set(df['Time']))
    new_df =  pd.DataFrame(np.empty((line_count, len(item_names))),columns=list(item_names), index=sorted(list(set(pd.to_datetime(df['Time'], infer_datetime_format=True)))))
    new_df[:] = np.nan
    time_dict = {item:jtem for item, jtem in zip(new_df.index.strftime('%Y-%m-%d %H:%M:%S'), new_df.index)}
    label = ''

    execution_counter = 0

    for index in df.index:
        if df['ItemName'][index].lower() != "label":
            if df['ItemName'][index].lower().startswith("global"):
                #No operation
                continue
            else:
                sensor = df['ItemName'][index]
                value = df['Value'][index]
                actions = tap_app_runner(sensor, value)
                execution_counter += 1
    print(f"The tap_app_runner function was executed {execution_counter} times.")

def normal_main():
    load_normal_data()
    print("Number of app1 executions normal state: ", app1_execution_counter)
    print("Number of app2 executions normal state: ", app2_execution_counter)
    print("Number of app3 executions normal state: ", app3_execution_counter)
    print("Number of app4 executions normal state: ", app4_execution_counter)
    print("Number of app5 executions normal state: ", app5_execution_counter)
    print("Number of all app executions normal state: ", app1_execution_counter+app2_execution_counter+app3_execution_counter+app4_execution_counter+app5_execution_counter)
        
    # Number of app1 executions:  24
    # Number of app2 executions:  46
    # Number of app3 executions:  1474
    # Number of app4 executions:  43
    # Number of app5 executions:  27
    # Number of all app executions:  1614


def perturber(attack_method, attack_params, mask=True, domain_adoption=True):

    #Load the dataset
    df = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/check_dataset.csv')
    df = df.drop(df.columns[0],axis=1) # dropping the timestamp column
    
    # Normalize the dataset    # preserve the first column -activity labels
    class_label = df.iloc[:, 0]
    df = df.drop(df.columns[0], axis=1)

    # normalize the sensor readings
    scaler = MinMaxScaler()
    df_normalized = scaler.fit_transform(df)
    # store the minimum and maximum values of each column
    min_values = scaler.data_min_
    max_values = scaler.data_max_
    # create a new dataframe with normalized sensor readings and the first column
    df = pd.DataFrame(df_normalized, columns=df.columns)
    df.insert(0, 'class_label', class_label)

    #spliting the last week of data as a seprate section for test
    # l=[7918,10469]  #last week of data
    # l=[9430,10469]  #last twoday
    l=[9960,10469]  #last day
    l_mod = [0] + l + [max(l)+1]
    print(l_mod)
    list_of_dfs = [df.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]
    dfTrain = list_of_dfs[0]
    dfTest = list_of_dfs[1]
    x_train = dfTrain.iloc[:, 1:].values # features are in columns 1 to 197
    y_train = dfTrain.iloc[:, 0].values # activity label is in column 0

    x_test = dfTest.iloc[:, 1:].values # features are in columns 1 to 197
    y_test = dfTest.iloc[:, 0].values # activity label is in column 0
    min_pixel_value, max_pixel_value = np.amin(x_train), np.amax(x_train)

    n_timesteps, n_features, n_outputs  = 1, x_train.shape[1], len(np.unique(y_train))
    x_train = x_train.reshape(-1, n_features, n_timesteps)
    y_train = np.eye(n_outputs)[y_train]
    x_test = x_test.reshape(-1, n_features, n_timesteps)
    y_test = np.eye(n_outputs)[y_test]

    # print('Attack method: ', attack_method , ' with params: ', attack_params)
    rel_tol=1e-5
    abs_tol=1e-8

    # rel_tol = 1e-09
    # abs_tol = 1e-09

    def get_possible_values(df, sensor_num):
        if 0 <= sensor_num <= 195:
            sensor_column = df.columns[sensor_num + 1]
            unique_values = df[sensor_column].unique()
            return sensor_column,unique_values
        else:
            raise ValueError("Sensor number should be between 0 and 195")
      

    sensordata_sparsmatrix = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/sensordata_sparsmatrix.csv')
    sensortypes = np.loadtxt('/home/maghvami/ptap/O4H_Classifier/sensor_typo.csv', delimiter=',', dtype=str)

    mask_injection = np.ones(x_test.shape)

    def mask_builder(sensortypes, mask_injection):
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
        return mask_injection
    
    def bc_mask_builder(sensortypes, mask_injection): #binary and categorical mask builder
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensor_type == 'binary' or sensor_type == 'categorical' or sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
        return mask_injection

    if (attack_method=='fgsm'):
        mask_injection = bc_mask_builder(sensortypes, mask_injection)
        domain_adaptation = False
    else:
        mask_injection = mask_builder(sensortypes, mask_injection)
        domain_adaptation = True       
    
    def domain_adaptation(x_test_adv, x_test):
        # Loop over the batches and sensors, and apply domain adaptation based on the sensor type
        for i in range(x_test_adv.shape[0]):  # iterate over the first dimension
            for j in range(x_test_adv.shape[1]):  # iterate over the second dimension (sensor readings)
                sensor_type = sensortypes[j][1]
                if sensor_type == 'binary':
                    if x_test_adv[i][j][0] != x_test[i][j][0]:
                        # x_test_adv[i][j][0] = x_test[i][j][0]  #put binary sensor back to original value
                        x_test_adv[i][j][0] = 1- x_test[i][j][0] #flip binary sensor
                elif sensor_type == 'real':
                    pass  # Do nothing for real sensors
                elif sensor_type == 'categorical':
                    sensor_name,possible_values = get_possible_values(df, j)
                    if possible_values is not None:
                        if len(possible_values) > 1:
                            if x_test_adv[i][j][0] != x_test[i][j][0]:
                            # if not math.isclose(x_test_adv[i][j][0], x_test[i][j][0], rel_tol=rel_tol, abs_tol=abs_tol):
                                # x_test_adv[i][j][0] = x_test[i][j][0]  #put categorical sensor back to original value
                                # Find the nearest possible value that is different from the original value
                                possible_values_not_original = [value for value in possible_values if value != x_test[i][j][0]]
                                nearest_value = min(possible_values_not_original, key=lambda x: abs(x - x_test_adv[i][j][0]))
                                x_test_adv[i][j][0] = nearest_value
                        else:
                            x_test_adv[i][j][0] = x_test[i][j][0]
                    else:
                        # Unknown categorical sensor
                        print(f"Unknown categorical sensor {sensor_name}")
                else:
                    # Unknown sensor type
                    print(f"Unknown sensor type for sensor {j}")
        return x_test_adv
    
    def noise_index_finder(x_test_adv, x_test):
        indices_dict = {} # initialize an empty dictionary
        for i in range(x_test_adv.shape[0]):  # iterate over the first dimension
            for j in range(x_test_adv.shape[1]):  # iterate over the second dimension (sensor readings)
                # if x_test_adv[i][j][0] != x_test[i][j][0]:
                if not math.isclose(x_test_adv[i][j][0], x_test[i][j][0], rel_tol=rel_tol, abs_tol=abs_tol):
                    indices_dict[(i, j)] = True # save the indices where the condition is met
        return indices_dict




    # original model - 1D-CNN
    class TensorFlowModel(Model):  
        def __init__(self):
            super(TensorFlowModel, self).__init__()
            self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(196, 1)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=25)])

        def call(self, x):
            return self.model(x)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    model = TensorFlowModel()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


    # Create the ART classifier
    # load model from file
    # model.load_weights('/home/maghvami/ptap/saved_models/1dcnn/1dcnn_v1')
    # model.load_weights('/home/maghvami/ptap/saved_models/4iter/classifier_4c_itr')
    model = tf.saved_model.load('/home/maghvami/ptap/saved_models/tmp/O4H_ART_saved_model8')

    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=25,
        input_shape=(196,1),
        clip_values=(min_pixel_value,max_pixel_value),
    )

    # Train the ART classifier
    # classifier.fit(x_train, y_train, batch_size=32, nb_epochs=10)
    # save model
    # model.save_weights('QTransfer_O4H_ART_saved_model1-2_weights')

    predictions = classifier.predict(x_test)
    accuracy_benign = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) *100
    print("Accuracy on benign test examples: {}%".format(accuracy_benign))

    # Generate adversarial test examples
    if mask:
        mask_param = mask_injection
    else:
        mask_param = None

    if (attack_method=='fgsm'):
        attack = FastGradientMethod(estimator=classifier, eps=attack_params['eps'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)
        indices_dict = noise_index_finder(x_test_adv, x_test)

    elif attack_method == 'jsma':
        attack = SaliencyMapMethod(classifier, theta=attack_params['theta'], gamma=attack_params['gamma'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)
        indices_dict = noise_index_finder(x_test_adv, x_test)

    df_adv_normalized = pd.DataFrame(x_test_adv.reshape(-1, n_features), columns=dfTest.columns[1:])
    df_adv_denormalized = pd.DataFrame(scaler.inverse_transform(df_adv_normalized), columns=dfTest.columns[1:])
    file_path_denormalized = '/home/maghvami/ptap/utility/df_adv_denormalized.csv'
    df_adv_denormalized.to_csv(file_path_denormalized, index=False)        

    return indices_dict

def load_noisy_data():
    # load_normal_data()    
    # attack_method ='fgsm'
    # attack_params= {'eps': 0.1}
    attack_method ='jsma'
    attack_params= {'theta': 0.1, 'gamma': 0.05}
    indices_dict = perturber(attack_method, attack_params, mask=True, domain_adoption=True)
    file_path_denormalized = '/home/maghvami/ptap/utility/df_adv_denormalized.csv'
    df_adv_denormalized = pd.read_csv(file_path_denormalized)
    c=0
    l=len(indices_dict)
    for index in indices_dict:
        c+=1
        print(c / l * 100, "%")
        i, j = index
        sensor = sensor_name_converter(j)
        value= df_adv_denormalized.iloc[i][j]
        actions = tap_app_runner(sensor, value)

    
def noisy_main():
    load_noisy_data()
    print("Number of app1 executions noisy state: ", app1_execution_counter)
    print("Number of app2 executions noisy state: ", app2_execution_counter)
    print("Number of app3 executions noisy state: ", app3_execution_counter)
    print("Number of app4 executions noisy state: ", app4_execution_counter)
    print("Number of app5 executions noisy state: ", app5_execution_counter)
    print("Number of all app executions noisy state: ", app1_execution_counter+app2_execution_counter+app3_execution_counter+app4_execution_counter+app5_execution_counter)


# noisy_main()

normal_main()