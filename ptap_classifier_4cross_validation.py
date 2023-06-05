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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(train_slices, test_slice):
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

    # Prepare train set
    x_train, y_train = [], []
    for train_slice in train_slices:
        start, end = train_slice
        dfTrain = df.iloc[start:end]
        x = dfTrain.iloc[:, 1:].values # features are in columns 1 to 197
        y = dfTrain.iloc[:, 0].values # activity label is in column 0
        x_train.append(x)
        y_train.append(y)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Prepare test set
    start, end = test_slice
    dfTest = df.iloc[start:end]
    x_test = dfTest.iloc[:, 1:].values # features are in columns 1 to 197
    y_test = dfTest.iloc[:, 0].values # activity label is in column 0

    min_pixel_value, max_pixel_value = np.amin(x_train), np.amax(x_train)

    n_timesteps, n_features, n_outputs  = 1, x_train.shape[1], len(np.unique(y_train))
    x_train = x_train.reshape(-1, n_features, n_timesteps)
    y_train = np.eye(n_outputs)[y_train]
    x_test = x_test.reshape(-1, n_features, n_timesteps)
    y_test = np.eye(n_outputs)[y_test]

    return df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs
    

def perturber(data):

    df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = data


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
    
    # def bc_mask_builder(sensortypes, mask_injection): #binary and categorical mask builder
    #     for i in range(mask_injection.shape[0]):
    #         for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
    #             sensor_type = sensortypes[j][1]
    #             if sensor_type == 'binary' or sensor_type == 'categorical' or sensordata_sparsmatrix.iloc[i, j] == 1:
    #                 mask_injection[i, j, 0] = 0
    #     return mask_injection

    mask_injection = mask_builder(sensortypes, mask_injection)    
    # mask_injection = bc_mask_builder(sensortypes, mask_injection)

    
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
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=25,
        input_shape=(196,1),
        clip_values=(min_pixel_value,max_pixel_value),
    )

    # Train the ART classifier
    classifier.fit(x_train, y_train, batch_size=32, nb_epochs=10)
    # save model
    model.save_weights('classifier-4c')

    predictions = classifier.predict(x_test)
    # accuracy_benign = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) *100
    # print("Accuracy on benign test examples: {}%".format(accuracy_benign))

    # get true and predicted labels
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')  # use 'micro' to handle multi-label case
    recall = recall_score(true_labels, predicted_labels, average='micro')
    fmeasure = f1_score(true_labels, predicted_labels, average='micro')

    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    return accuracy, precision, recall, fmeasure





    return accuracy_benign


def main_runner():
    # 4-fold cross-validation
    data_slices = [(0, 2680), (2681, 5331), (5332, 7916), (7917, 10469)]
    
    # Initialize lists to hold metrics
    accuracies = []
    precisions = []
    recalls = []
    fmeasures = []

    for i, data_slice in enumerate(data_slices):
        train_slices = data_slices[:i] + data_slices[i+1:]
        test_slice = data_slice
        data = load_data(train_slices, test_slice)
        accuracy, precision, recall, fmeasure = perturber(data)
        
        # Append metrics to lists
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fmeasures.append(fmeasure)

    # Calculate and print average metrics
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_fmeasure = np.mean(fmeasures)
    
    print(f'Average accuracy: {avg_accuracy * 100}%')
    print(f'Average precision: {avg_precision * 100}%')
    print(f'Average recall: {avg_recall * 100}%')
    print(f'Average F-measure: {avg_fmeasure * 100}%')

    # Store metrics in a dataframe and save to a csv file
    metrics_df = pd.DataFrame({
        'accuracy': accuracies,
        'precision': precisions,
        'recall': recalls,
        'fmeasure': fmeasures,
    })
    
    metrics_df.to_csv('/home/maghvami/ptap/rep_results/classifier-4_cross_validation.csv', index=False)



main_runner()