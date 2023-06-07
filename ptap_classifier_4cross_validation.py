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
    df = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/aggregated_dataset.csv')
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
    

def classifier(data , index):

    df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = data

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
    # model.load_weights('/home/maghvami/ptap/saved_models/1dcnn/classifier-4c')
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
    # save model + index of iteration
    model.save_weights('/home/maghvami/ptap/saved_models/4iter/classifier_4c_itr')

    predictions = classifier.predict(x_test)
    
    # get true and predicted labels
    true_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')  
    recall = recall_score(true_labels, predicted_labels, average='micro')
    fmeasure = f1_score(true_labels, predicted_labels, average='micro')

    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    return accuracy, precision, recall, fmeasure


def main_runner():
    # 4-fold cross-validation
    data_slices = [(0, 2680), (2681, 5331), (5332, 7916), (7917, 10469)]
    
    # Initialize lists to hold metrics
    accuracies = []
    precisions = []
    recalls = []
    fmeasures = []

    for i, data_slice in enumerate(data_slices):
        if i == 3:
            train_slices = data_slices[:i] + data_slices[i+1:]
            test_slice = data_slice
            data = load_data(train_slices, test_slice)
            accuracy, precision, recall, fmeasure = classifier(data , i+1)
            
            # Append metrics to lists
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            fmeasures.append(fmeasure)

    # # Calculate and print average metrics
    # avg_accuracy = np.mean(accuracies)
    # avg_precision = np.mean(precisions)
    # avg_recall = np.mean(recalls)
    # avg_fmeasure = np.mean(fmeasures)
    
    # print(f'Average accuracy: {avg_accuracy * 100}%')
    # print(f'Average precision: {avg_precision * 100}%')
    # print(f'Average recall: {avg_recall * 100}%')
    # print(f'Average F-measure: {avg_fmeasure * 100}%')

    # Store metrics in a dataframe and save to a csv file
    metrics_df = pd.DataFrame({
        'accuracy': accuracies,
        'precision': precisions,
        'recall': recalls,
        'fmeasure': fmeasures,
    })
    
    metrics_df.to_csv('/home/maghvami/ptap/rep_results/nader.csv', index=False)

main_runner()