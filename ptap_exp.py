from art.attacks.evasion import FastGradientMethod, CarliniL0Method, SaliencyMapMethod, CarliniL2Method, DeepFool, NewtonFool, CarliniLInfMethod, ElasticNet, ProjectedGradientDescent, BasicIterativeMethod, SpatialTransformation, HopSkipJump, ZooAttack, UniversalPerturbation
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
import pandas as pd
import numpy as np
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
from sklearn.preprocessing import MinMaxScaler
import os
import csv
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# Set GPU devices to be empty, effectively using only CPU
tf.config.set_visible_devices([], 'GPU')

def load_data():
    # Define the path to the 'res' folder relative to the current script location
    base_path = os.path.join(os.path.dirname(__file__), 'res')

    #Load the dataset
    df = pd.read_csv(os.path.join(base_path, 'check_dataset.csv'))
    # df = pd.read_csv('/home/amini/ptap/O4H_Classifier/check_dataset.csv')
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

    return df,dfTest ,scaler, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs

def perturber(attack_method, attack_params, data,y_target):

    # Define the path to the 'res' folder relative to the current script location
    base_path = os.path.join(os.path.dirname(__file__), 'res')

    real_event_count = 0  # Counter for real events
    # num_injected_events=0
    print('Attack method: ', attack_method , ' with params: ', attack_params)

    df, dfTest,scaler, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = data


    def get_possible_values(df, sensor_num):
        if 0 <= sensor_num <= 195:
            sensor_column = df.columns[sensor_num + 1]
            unique_values = df[sensor_column].unique()
            return sensor_column,unique_values
        else:
            raise ValueError("Sensor number should be between 0 and 195")
      
    sensordata_sparsmatrix = pd.read_csv(os.path.join(base_path, 'sensordata_sparsmatrix.csv'))
    # sensordata_sparsmatrix = pd.read_csv('/home/amini/ptap/O4H_Classifier/sensordata_sparsmatrix.csv')

    sensortypes = np.loadtxt(os.path.join(base_path, 'sensor_typo.csv'), delimiter=',', dtype=str)
    # sensortypes = np.loadtxt('/home/amini/ptap/O4H_Classifier/sensor_typo.csv', delimiter=',', dtype=str)

    mask_injection = np.ones(x_test.shape)

    def mask_builder(sensortypes, mask_injection):
        real_event_count = 0  # Counter for real events
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
                    real_event_count += 1  
        return mask_injection, real_event_count
    
    def bc_mask_builder(sensortypes, mask_injection): #binary and categorical mask builder
        real_event_count = 0  # Counter for real events
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensor_type == 'binary' or sensor_type == 'categorical' or sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
                    if sensordata_sparsmatrix.iloc[i, j] == 1:
                        real_event_count += 1  
        return mask_injection, real_event_count

    #domain adoption strategy
    if attack_method == 'fgsm'  or 'uap_fgsm'or 'uap_deepfool' or 'uap_pgd' or 'random_noise':
        mask_injection, real_event_count = bc_mask_builder(sensortypes, mask_injection)
        mask = True
        domain_adoption = False
    elif attack_method == 'jsma'  or 'uap_jsma' or 'cwl0' or 'cwl2':
        mask_injection, real_event_count = mask_builder(sensortypes, mask_injection)    
        mask = True
        domain_adoption = True
    
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


    # load model from file
    # model.load_weights('/home/amini/ptap/saved_models/4iter/classifier_4c_itr')
    model_weights_path = os.path.join(base_path, 'saved_models', '4iter', 'classifier_4c_itr')
    model.load_weights(model_weights_path)


    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=25,
        input_shape=(196,1),
        clip_values=(min_pixel_value,max_pixel_value),
    )

    # Train the ART classifier
    # classifier.fit(x_train, y_train, batch_size=32, nb_epochs=50)
    # save model
    # model.save_weights('/home/amini/ptap/saved_models/ptap_e50_v2')

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
        # x_test_adv, num_injected_events = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)

    elif attack_method == 'jsma':
        attack = SaliencyMapMethod(classifier, theta=attack_params['theta'], gamma=attack_params['gamma'])
        x_test_adv = attack.generate(x=x_test, y=y_target, mask=mask_param)
        # x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)

    elif attack_method == 'uap_fgsm':
            attack = UniversalPerturbation(classifier, attacker="fgsm", attacker_params=attack_params, verbose=True)
            x_test_adv = attack.generate(x_test, mask=mask_injection)
            
            # Compute the number of injected events
            # threshold = 0.00001
            # diff = np.where(np.abs(x_test_adv - x_test) > threshold, 1, 0)
            # num_injected_events = np.count_nonzero(diff)
            # average_num_injected_events_per_sample = num_injected_events / 2551
            
            # You can return this value or print it
            # print(f"Total number of injected events: {num_injected_events}")
            # print(f"Average number of injected events per sample: {average_num_injected_events_per_sample}")

            if domain_adoption:
                x_test_adv = domain_adaptation(x_test_adv, x_test)
  
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



def tap(model_param, attack_params, data, x_test_adv , original_accuracy, adversarial_accuracy):

    results = []
    # Define the path to the 'res' folder relative to the current script location
    base_path = os.path.join(os.path.dirname(__file__), 'res')

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
        
    # class TensorFlowModel4(Model):  #Deep1DCNNModel
    #         def __init__(self):
    #             super(TensorFlowModel4, self).__init__()
    #             self.model = tf.keras.models.Sequential([
    #                 tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(196, 1)),
    #                 tf.keras.layers.MaxPooling1D(pool_size=2),
    #                 tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #                 tf.keras.layers.MaxPooling1D(pool_size=2),
    #                 tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    #                 tf.keras.layers.MaxPooling1D(pool_size=2),
    #                 tf.keras.layers.Flatten(),
    #                 tf.keras.layers.Dense(units=128, activation='relu'),
    #                 tf.keras.layers.Dense(units=25)])

    #         def call(self, x):
    #             return self.model(x)

    user_choice = model_param
    print("User choice is: ", user_choice)
    df, dfTest, scaler, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = data

    if user_choice in ['model2', 'both']:
        print(f"model2 -LSTM ")

        def train_step(model, images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        
        model2 = TensorFlowModel2()
        # Load model2 weights from the new relative path
        model2_weights_path = os.path.join(base_path, 'saved_models', 'old', 'lstm_model2_v1')
        model2.load_weights(model2_weights_path)
        

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
        # classifier2.fit(x_train, y_train, batch_size=32, nb_epochs=50)

        # model2.save_weights('/home/amini/ptap/saved_models/ld_03')
        


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

    
        model3 = TensorFlowModel3()

        # Load model3 weights from the new relative path
        model3_weights_path_v1 = os.path.join(base_path, 'saved_models', 'old', 'mlp_model3_v1')
        model3.load_weights(model3_weights_path_v1)
        
        # Load model4 -DeepCNN weights
        # model4_weights_path_ld04 = os.path.join(base_path, 'saved_models', 'ld04')
        # model3.load_weights(model4_weights_path_ld04)

        classifier3 = TensorFlowV2Classifier(
            model=model3,
            loss_object=loss_object,
            train_step=train_step,
            nb_classes=25,
            input_shape=(196,1),
            clip_values=(min_pixel_value,max_pixel_value),
        )

        # Step 4: Train the ART classifier
        # classifier3.fit(x_train, y_train, batch_size=32, nb_epochs=50 )

        # model3.save_weights('/home/amini/ptap/saved_models/tap/mlp_ld_01')


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

    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_base_dir = "/home/amini/ptap/exp_results"

    # Create the directory if it doesn't exist
    save_directory = os.path.join(results_base_dir, 'transfer_challenger')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the file with timestamp inside the directory
    filename = f"transfer_challenger_{timestamp}.csv"
    filepath = os.path.join(save_directory, filename)


    with open(filepath, 'a', newline='') as csvfile:
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



def saver(attack_method, attack_params, accuracy_benign, accuracy_adversarial, x_test_adv, dfTest, scaler):
    results_base_dir = "/home/amini/ptap/exp_results"
    n_features = 196
    

    def save_adversarial_examples(attack_method, attack_params, x_test_adv, dfTest,scaler):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

        attack_method_dir = os.path.join(results_base_dir, attack_method)
        if not os.path.exists(attack_method_dir):
            os.makedirs(attack_method_dir)

        param_str = "_".join([f"{param}{value}" for param, value in attack_params.items()])

        # Save the normalized adversarial examples to a CSV file
        adv_examples_normalized_file_path = os.path.join(attack_method_dir, f"adv_examples_normalized_{param_str}_{current_time}.csv")
        df_adv_normalized = pd.DataFrame(x_test_adv.reshape(-1, n_features), columns=dfTest.columns[1:])
        df_adv_normalized.to_csv(adv_examples_normalized_file_path, index=False)

        # Denormalize the adversarial examples
        df_adv_denormalized = pd.DataFrame(scaler.inverse_transform(df_adv_normalized), columns=dfTest.columns[1:])

        # Save the denormalized adversarial examples to a CSV file
        adv_examples_denormalized_file_path = os.path.join(attack_method_dir, f"adv_examples_denormalized_{param_str}_{current_time}.csv")
        df_adv_denormalized.to_csv(adv_examples_denormalized_file_path, index=False)        
   
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)
    result_file_path = os.path.join(results_base_dir, 'results.csv')
    attack_name = f"{attack_method}"
    for param, value in attack_params.items():
        attack_name += f"_{param}{value}"

    result_data = {
        "Attack": [attack_name],
        "Accuracy_Benign": [accuracy_benign],
        "Accuracy_Adversarial": [accuracy_adversarial]
    }
    result_df = pd.DataFrame(result_data)

    if not os.path.exists(result_file_path):
        # If the file doesn't exist, create it and add the header
        result_df.to_csv(result_file_path, index=False, header=True)
    else:
        # If the file exists, append the data without adding the header
        result_df.to_csv(result_file_path, index=False, mode='a', header=False)

    save_adversarial_examples(attack_method, attack_params, x_test_adv, dfTest, scaler)
    

def plot_attack_accuracy(attack_name, param_values, accuracies, original_accuracy, results_base_dir):
    plt.figure()
    
    # Place the original accuracy to the left of the first gamma value
    plt.scatter(param_values[0] - (0.05 * (param_values[-1] - param_values[0])), original_accuracy, color='r', marker='o', label='Original Accuracy')
    
    # Plot adversarial accuracies
    plt.plot(param_values, accuracies, marker='o', label='Adversarial Accuracy')
    
    if attack_name.lower() == 'fgsm':
        param_name = 'Epsilon'
    elif attack_name.lower() == 'jsma':
        param_name = 'Gamma'
    else:
        param_name = 'Attack Parameter'

    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'{attack_name} Attack: Accuracy vs {param_name}')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # Reversing the order of the legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper right')

    # Save the plot
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    file_name = f"{attack_name}_accuracy_vs_{param_name.lower()}_{current_time}.eps"
    plt.savefig(os.path.join(results_base_dir, file_name), format='eps')
    plt.show()


def multi_plot_attack_accuracy(attack_name, gammas, accuracies, original_accuracy, results_base_dir, num_runs=3):
    plt.figure()  # Removed figsize argument
    
    # Compute average and standard deviation for accuracies
    avg_accuracies = [np.mean(accuracies[i:i+num_runs]) for i in range(0, len(accuracies), num_runs)]
    std_accuracies = [np.std(accuracies[i:i+num_runs]) for i in range(0, len(accuracies), num_runs)]
    
    # Plotting
    plt.errorbar(gammas, avg_accuracies, yerr=std_accuracies, marker='o', capsize=5, label='Adversarial Accuracy')
    
    # Plot original accuracy as a point
    plt.scatter(gammas[0] - (0.05 * (gammas[-1] - gammas[0])), original_accuracy, color='r', marker='o', label='Original Accuracy')
    
    param_name = 'Gamma' if attack_name.lower() == 'jsma' else 'Attack Parameter'
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'{attack_name} Attack: Accuracy vs {param_name}')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend(loc='upper right')

    # Save the plot
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    file_name = f"{attack_name}_accuracy_vs_{param_name.lower()}_{current_time}.eps"
    plt.savefig(os.path.join(results_base_dir, file_name), format='eps')
    plt.show()

def save_time_and_event_measurement(filename, attack_name, attack_params, start_time, end_time, num_injected_events):
    runtime_minutes = (end_time - start_time)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    attack_params_str = str(attack_params)  # Convert dictionary to string for CSV saving
    
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow([attack_name, attack_params_str, runtime_minutes, timestamp, num_injected_events])

def main_runner(experiment_id):
    tfilename = '/home/amini/ptap/exp_results/time_event_measurement.csv'
    accuracies = []
    # Load the data
    # df, min_pixel_value, max_pixel_value, x_train, y_train, x_test, y_test, n_timesteps, n_features, n_outputs = load_data()
    data = load_data()
    results_base_dir = "/home/amini/ptap/exp_results"
    original_accuracy = 0
    adversarial_accuracy = 0
    num_injected_events = 0
    # Generate adversarial examples

    if(experiment_id == 'fgsm'):
        attack_method='fgsm'
        # epsilons = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4, 0.45, 0.5]
        epsilons = [0.15]
        for epsilon in epsilons:
            attack_params = {'eps': epsilon}
            print(f"Running for epsplot_attack_accuracy(attack_method, epsilons, accuracies, original_accuracy,results_base_dir)ilon = {epsilon}")
            start_time = time.time()
            # original_accuracy, adversarial_accuracy, x_test_adv, num_injected_events = perturber(attack_method, attack_params, data, y_target=None)
            original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, y_target=None)
            end_time = time.time()
            save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time,num_injected_events)
            accuracies.append(adversarial_accuracy)
            saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])
        plot_attack_accuracy(attack_method, epsilons, accuracies, original_accuracy,results_base_dir)

    
    elif(experiment_id == 'itr_jsma'):
        attack_method = 'jsma'
        num_runs =50
        run_indices = list(range(1, num_runs + 1))
        # gammas = [0.05, 0.1, 0.15, 0.2]
        gammas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,0.4, 0.45, 0.5]
        # gammas = [0.35]
        attack_params_base = {'theta': 0.1, 'gamma': None}
        
        all_accuracies = {gamma: [] for gamma in gammas}
        
        for run in run_indices:
            print(f"Running JSMA attack iteration {run}/{num_runs}")
            for gamma in gammas:
                attack_params = attack_params_base.copy()
                attack_params['gamma'] = gamma
                print(f"Running for gamma = {gamma}")

                # Randomly select target classes for each sample
                y_target_random = np.eye(25)[np.random.choice(25, data[7].shape[0])]

                start_time = time.time()          
                # original_accuracy, adversarial_accuracy, x_test_adv, num_injected_events= perturber(attack_method, attack_params, data, y_target=y_target_random)
                original_accuracy, adversarial_accuracy, x_test_adv= perturber(attack_method, attack_params, data, y_target=y_target_random)
                end_time = time.time()
                save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time,num_injected_events)
                
                all_accuracies[gamma].append(adversarial_accuracy)
                saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])

        # Flatten the accuracies list to match the structure needed by the plotting function
        accuracies = [item for sublist in all_accuracies.values() for item in sublist]
        
        multi_plot_attack_accuracy('JSMA_Untargeted', gammas, accuracies, original_accuracy, results_base_dir, num_runs=num_runs)


    elif(experiment_id == 'trans_itr_jsma'):
        attack_method = 'jsma'
        num_runs =1
        run_indices = list(range(1, num_runs + 1))
        # gammas = [0.05, 0.1, 0.15, 0.2]
        # gammas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,0.4, 0.45, 0.5]
        gammas = [0.2]
        # gammas = [0.01]
        attack_params_base = {'theta': 0.1, 'gamma': None}
        
        all_accuracies = {gamma: [] for gamma in gammas}
        
        for run in run_indices:
            print(f"Running JSMA attack iteration {run}/{num_runs}")
            for gamma in gammas:
                attack_params = attack_params_base.copy()
                attack_params['gamma'] = gamma
                print(f"Running for gamma = {gamma}")

                # Randomly select target classes for each sample
                y_target_random = np.eye(25)[np.random.choice(25, data[7].shape[0])]

                start_time = time.time()          
                # original_accuracy, adversarial_accuracy, x_test_adv, num_injected_events= perturber(attack_method, attack_params, data, y_target=y_target_random)
                original_accuracy, adversarial_accuracy, x_test_adv= perturber(attack_method, attack_params, data, y_target=y_target_random)
                end_time = time.time()
                save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time,num_injected_events)
                
                # For each generated adversarial example, test against both target models
                target_model = 'model2'
                tap(target_model, attack_params, data, x_test_adv, original_accuracy, adversarial_accuracy)

                all_accuracies[gamma].append(adversarial_accuracy)
                saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])

        # Flatten the accuracies list to match the structure needed by the plotting function
        accuracies = [item for sublist in all_accuracies.values() for item in sublist]
        
        multi_plot_attack_accuracy('JSMA_Untargeted', gammas, accuracies, original_accuracy, results_base_dir, num_runs=num_runs)

    elif(experiment_id == 'uap_fgsm'):
        attack_method='uap_fgsm'
        # epsilons = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4, 0.45, 0.5]
        epsilons = [0.001,0.002,0.003,0.004,0.005, 0.1,0.2,0.3,0.4, 0.5]
        # epsilons = [0.005]
        for epsilon in epsilons:
            attack_params = {'eps': epsilon}
            print(f"Running for epsilon = {epsilon}")
            start_time = time.time()            
            original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, y_target=None)
            end_time = time.time()
            save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time, num_injected_events)
            accuracies.append(adversarial_accuracy)
            saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2] )
        plot_attack_accuracy(attack_method, epsilons, accuracies, original_accuracy,results_base_dir)


    #Black box tests  fgsm  
    elif(experiment_id == 'trans_fgsm'):
        attack_method='fgsm'
        # epsilons = [0.05,0.1,0.15,0.2,0.25,0.3]
        # epsilons = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        # epsilons = [0.05]
        # epsilons = [0.1,0.2,0.3]
        epsilons = [0.2]
        for epsilon in epsilons:
            attack_params = {'eps': epsilon}
            print(f"Running for epsilon = {epsilon}")
            start_time = time.time()
            original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, y_target=None)
            end_time = time.time()
            save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time, num_injected_events)
            accuracies.append(adversarial_accuracy)
            saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])
            target_model = 'both'
            # tap(target_model,attack_params, data, x_test_adv, original_accuracy, adversarial_accuracy)
            tap(target_model, attack_params, data, x_test_adv, original_accuracy, adversarial_accuracy)
        # plot_attack_accuracy(attack_method, epsilons, accuracies, original_accuracy,results_base_dir)


    #Black box tests jsma
    elif(experiment_id == 'trans_jsma'):
        attack_method = 'jsma'
        gammas = [0.2]
        attack_params_base = {'theta': 0.1, 'gamma': None}
        for gamma in gammas:
            attack_params = attack_params_base.copy()
            attack_params['gamma'] = gamma
            print(f"Running for gamma = {gamma}")
            start_time = time.time()
            original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, y_target=None)
            end_time= time.time()
            save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time,num_injected_events)
            accuracies.append(adversarial_accuracy)
            saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])
            target_model = 'both'
            tap(target_model,attack_params, data, x_test_adv, original_accuracy, adversarial_accuracy)
        # plot_attack_accuracy(attack_method, gammas, accuracies, original_accuracy,results_base_dir)


    elif(experiment_id == 'trans_uap_fgsm'):
        attack_method='uap_fgsm'
        # epsilons = [0.1,0.2,0.3,0.4,0.5]
        # epsilons = [0.0,0.05]
        epsilons = [0.2]

        for epsilon in epsilons:
            attack_params = {'eps': epsilon}
            print(f"Running for epsilon = {epsilon}")
            start_time = time.time()
            original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, y_target=None)
            end_time = time.time()
            save_time_and_event_measurement(tfilename, attack_method, attack_params, start_time, end_time, num_injected_events)
            accuracies.append(adversarial_accuracy)
            saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])
            target_model = 'both'
            tap(target_model,attack_params, data, x_test_adv, original_accuracy, adversarial_accuracy)
        
    elif(experiment_id == 'random_noise'):
        attack_method = 'random_noise'
        epsilons = [0.1,0.2,0.3]
        for epsilon in epsilons:
            attack_params = {'eps': epsilon}
            print(f"Running random noise with epsilon = {epsilon}")
            start_time = time.time()
            original_accuracy, adversarial_accuracy, x_test_adv = perturber(attack_method, attack_params, data, y_target=None)
            end_time = time.time()
            accuracies.append(adversarial_accuracy)
            saver(attack_method, attack_params, original_accuracy, adversarial_accuracy, x_test_adv, dfTest=data[1], scaler=data[2])

    end_time = time.time()
    print(f"The code took {(end_time - start_time) } Seconds to run.")

#options: fgsm, itr_jsma, uap_fgsm, trans_fgsm, trans_jsma, trans_uap_fgsm, trans_itr_jsma, random_noise
main_runner('fgsm')
