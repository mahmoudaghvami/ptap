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



def load_data():
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
    l=[7918,10469]  #last week of data
    # l=[9430,10469]  #last twoday
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

    return df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs
    

def perturber(attack_method, attack_params, data, mask=True, domain_adoption=True, new_run=True):

    print('Attack method: ', attack_method , ' with params: ', attack_params)
    rel_tol = 1e-09
    abs_tol = 1e-09


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
    
    def bc_mask_builder(sensortypes, mask_injection): #binary and categorical mask builder
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensor_type == 'binary' or sensor_type == 'categorical' or sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
        return mask_injection

    # mask_injection = mask_builder(sensortypes, mask_injection)    
    mask_injection = bc_mask_builder(sensortypes, mask_injection)

    
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
    model.load_weights('/home/maghvami/ptap/saved_models/4iter/classifier_4c_itr')
    # model = tf.saved_model.load('/home/maghvami/ptap/saved_models/tmp/O4H_ART_saved_model8')

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
    if new_run == True:
        if mask:
            mask_param = mask_injection
        else:
            mask_param = None

        if (attack_method=='fgsm'):
            attack = FastGradientMethod(estimator=classifier, eps=attack_params['eps'])
            x_test_adv = attack.generate(x=x_test, mask=mask_param)
            if domain_adoption:
                x_test_adv = domain_adaptation(x_test_adv, x_test)
        elif attack_method == 'carliniwagner_l0':
            attack = CarliniL0Method(classifier=classifier, confidence=attack_params['confidence'], max_iter=attack_params['max_iter'])
            x_test_adv = attack.generate(x=x_test, mask=mask_param)
            if domain_adoption:
                x_test_adv = domain_adaptation(x_test_adv, x_test)

        elif attack_method == 'jsma':
            attack = SaliencyMapMethod(classifier, theta=attack_params['theta'], gamma=attack_params['gamma'])
            x_test_adv = attack.generate(x=x_test, mask=mask_param)
            if domain_adoption:
                x_test_adv = domain_adaptation(x_test_adv, x_test)
    else:
        print("Loading saved adversarial examples...")
        # attack_params_str = '_'.join([f"{k}_{v}" for k, v in attack_params.items()])
        # directory = "./saved_adv_examples/"
        # filename = f"x_test_adv_{attack_method}_{attack_params_str}.npy"
        # x_test_adv = np.load(os.path.join(directory, filename))
  
    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy_adversarial = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) * 100
    print("Accuracy on adversarial test examples: {}%".format(accuracy_adversarial))

    # Save the adversarial examples for further testing
    attack_params_str = '_'.join([f"{k}_{v}" for k, v in attack_params.items()])    
    directory = "./saved_adv_examples/"
    os.makedirs(directory, exist_ok=True)
    filename = f"x_test_adv_{attack_method}_{attack_params_str}.npy"
    np.save(os.path.join(directory, filename), x_test_adv)

    return accuracy_benign, accuracy_adversarial, x_test_adv



def tap(model_param, data, x_test_adv , original_accuracy, adversarial_accuracy):

    results = []
    
    class TensorFlowModel2(Model): #LSTM model
        def __init__(self):
            super(TensorFlowModel2, self).__init__()
            n_features = 196
            n_classes = 25
            n_nodes = (n_features + n_classes) // 2
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(units=n_nodes, input_shape=(n_features, 1)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=n_classes)
            ])

        def call(self, x):
            return self.model(x)
        
    class TensorFlowModel3(Model): #MLP model
        def __init__(self):
            super(TensorFlowModel3, self).__init__()
            n_features = 196
            n_classes = 25
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(n_features, 1)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(n_classes),
            ])

        def call(self, x):
            return self.model(x)


    user_choice = model_param
    print("User choice is: ", user_choice)
    df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = data

    if user_choice in ['model2', 'both']:
        print(f"model2 -LSTM ")

        def train_step(model, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model2 = TensorFlowModel2()
            # load model
            model2.load_weights('/home/maghvami/ptap/saved_models/tap/lstm_model2_v1')

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            classifier2 = TensorFlowV2Classifier(
                model=model2,
                loss_object=loss_object,
                train_step=train_step,
                nb_classes=25,
                input_shape=(196,1),
                clip_values=(min_pixel_value,max_pixel_value),
            )

        # Step 4: Train the ART classifier
        # classifier2.fit(x_train, y_train, batch_size=32, nb_epochs=10)

        # model2.save_weights('/home/maghvami/ptap/saved_models/tap/lstm_model2_v1')


        predictions_benign = classifier2.predict(x_test)
        accuracy_benign_new = np.sum(np.argmax(predictions_benign, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) *100
        print("Accuracy on benign test examples for model2: {}%".format(accuracy_benign_new))

        # Evaluate model2 on adversarial examples
        predictions_adv = classifier2.predict(x_test_adv)
        accuracy_adv_new = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) * 100
        print("Accuracy on adversarial test examples for model2: {}%".format(accuracy_adv_new))

        # Add results to list
        results.append({
            'model': 'model2',
            'accuracy_benign': accuracy_benign_new,
            'accuracy_adv': accuracy_adv_new,
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy
        })


    if user_choice in ['model3', 'both']:
        print(f"model3 -MLP ")

        def train_step(model, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model3 = TensorFlowModel3()
            # load model
            model3.load_weights('/home/maghvami/ptap/saved_models/tap/mlp_model3_v1')

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


            classifier3 = TensorFlowV2Classifier(
                model=model3,
                loss_object=loss_object,
                train_step=train_step,
                nb_classes=25,
                input_shape=(196,1),
                clip_values=(min_pixel_value,max_pixel_value),
            )

        # Step 4: Train the ART classifier
        # classifier3.fit(x_train, y_train, batch_size=32, nb_epochs=10)

        # model3.save_weights('/home/maghvami/ptap/saved_models/tap/mlp_model3_v1')


        predictions_benign = classifier3.predict(x_test)
        accuracy_benign_new = np.sum(np.argmax(predictions_benign, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) *100
        print("Accuracy on benign test examples for model3: {}%".format(accuracy_benign_new))


        # Evaluate model3 on adversarial examples
        predictions_adv = classifier3.predict(x_test_adv)
        accuracy_adv_new = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) * 100
        print("Accuracy on adversarial test examples for model3: {}%".format(accuracy_adv_new))

        results.append({
            'model': 'model3',
            'accuracy_benign': accuracy_benign_new,
            'accuracy_adv': accuracy_adv_new,
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy
        })

    # Write results to CSV file
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ['model', 'accuracy_benign', 'accuracy_adv', 'original_accuracy', 'adversarial_accuracy', 'epsilon', 'gamma']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:  # Check if the file is empty
            writer.writeheader()

        for result in results:
            if 'eps' in attack_params:
                result['epsilon'] = attack_params['eps']
            if 'gamma' in attack_params:
                result['gamma'] = attack_params['gamma']
            writer.writerow(result)



def main_runner(user_choice_model, attack_method, attack_params):
    
    start_time = time.time()

    # Load the data
    # df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = load_data()
    data = load_data()

    original_accuracy = 0
    adversarial_accuracy = 0
    # Generate adversarial examples
    original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, mask=True, domain_adoption=False, new_run=True)
    

    # Evaluate the adversarial examples on the chosen model
    # attack_params_str = '_'.join([f"{k}_{v}" for k, v in attack_params.items()])
    # subdirectory = "gamma0.2/"
    # directory = "./saved_adv_examples/transfer/"+subdirectory
    # filename = f"x_test_adv_{attack_method}_{attack_params_str}.npy"
    # x_test_adv = np.load(os.path.join(directory, filename))

    tap(user_choice_model, data, x_test_adv, original_accuracy, adversarial_accuracy)

    end_time = time.time()
    print(f"The code took {(end_time - start_time) /60} minutes to run.")

target_model = 'both'

# attack_name = 'fgsm'
# epsilons = [0.1, 0.2, 0.3]
# for epsilon in epsilons:
#     attack_params = {'eps': epsilon}
#     print(f"Running for epsilon = {epsilon}")
#     main_runner(target_model, attack_name, attack_params)


attack_name = 'jsma'
gammas = [0.05, 0.1, 0.15]
attack_params_base = {'theta': 0.1, 'gamma': None}
for gamma in gammas:
    attack_params = attack_params_base.copy()
    attack_params['gamma'] = gamma
    print(f"Running for gamma = {gamma}")
    main_runner(target_model, attack_name, attack_params)
