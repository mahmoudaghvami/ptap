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

def perturber(attack_method, attack_params, mask=True, domain_adoption=True):

    print('Attack method: ', attack_method , ' with params: ', attack_params)
    rel_tol = 1e-09
    abs_tol = 1e-09

    # Step 1: Load the dataset
    df = pd.read_csv('/home/maghvami/ptap/O4H_Classifier/check_dataset.csv')
    df = df.drop(df.columns[0],axis=1) # dropping the timestamp column
    
    # Normalize the dataset
    # preserve the first column
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

    def get_possible_values(df, sensor_num):
        if 0 <= sensor_num <= 195:
            sensor_column = df.columns[sensor_num + 1]
            unique_values = df[sensor_column].unique()
            return sensor_column,unique_values
        else:
            raise ValueError("Sensor number should be between 0 and 195")
    
    
    #spliting the last week of data as a seprate section for test
    l=[7918,10469]  #last week of data
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

    #step 1.1 smart injection and domain adoption
    sensordata_sparsmatrix = pd.read_csv('/home/maghvami/ptap/old_tests/ijcnn19attacks/src/sensordata_sparsmatrix.csv')
    sensortypes = np.loadtxt('/home/maghvami/ptap/old_tests/ijcnn19attacks/src/sensor_typo.csv', delimiter=',', dtype=str)

    # mask is a numpy array of shape (x_test.shape) with all elements set to 1
    mask_injection = np.ones(x_test.shape)

    def mask_builder(sensortypes, mask_injection):
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
        return mask_injection
    
    def bc_mask_builder(sensortypes, mask_injection):
        for i in range(mask_injection.shape[0]):
            for j in range(mask_injection.shape[1]): #iterate over the sensors 0-195
                sensor_type = sensortypes[j][1]
                if sensor_type == 'binary' or sensor_type == 'categorical' or sensordata_sparsmatrix.iloc[i, j] == 1:
                    mask_injection[i, j, 0] = 0
        return mask_injection

    # if(attack_method == 'fgsm'):
        mask_injection = bc_mask_builder(sensortypes, mask_injection)
    # else:
    #     mask_injection = mask_builder(sensortypes, mask_injection)
    
   
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


    # Step 2: Create the model

    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D , LSTM


    class TensorFlowModel(Model):
        """
        Standard TensorFlow model for unit testing.
        """

        def __init__(self):
            super(TensorFlowModel, self).__init__()
            # self.lstm = LSTM(100, input_shape=(n_features, n_timesteps))
            # self.dense = Dense(n_outputs, activation="softmax")

            self.model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(196, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=25) ])

        def call(self, x):
            """
            Call function to evaluate the model.

            :param x: Input to the model
            :return: Prediction of the model
            """
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

    # Step 3: Create the ART classifier
    # load model from file
    # model = tf.saved_model.load('/home/maghvami/ptap/saved_models/tmp/O4H_ART_saved_model8')
    # model.load_weights('/home/maghvami/ptap/saved_models/1dcnn/1dcnn_v1')
    model.load_weights('/home/maghvami/ptap/saved_models/4iter/classifier_4c_itr')


    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=25,
        input_shape=(196,1),
        clip_values=(min_pixel_value,max_pixel_value),
    )

    # Step 4: Train the ART classifier

    # classifier.fit(x_train, y_train, batch_size=32, nb_epochs=10)
        # tf.saved_model.save(model, 'O4H_ART_saved_model9')

    # Step 5: Evaluate the ART classifier on benign test examples

    def print_classification_report(y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(25)], digits=4)
        print("\nClassification report:")
        print(report)

    predictions = classifier.predict(x_test)
    accuracy_benign = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) *100
    print("Accuracy on benign test examples: {}%".format(accuracy_benign))
    # print_classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))


    # Step 6: Generate adversarial test examples
    if mask:
        mask_param = mask_injection
    else:
        mask_param = None

    if (attack_method=='fgsm'):
        attack = FastGradientMethod(estimator=classifier, eps=attack_params['eps'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        # x_test_adv = attack.generate(x=x_test)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)
    elif attack_method == 'carliniwagner_l0':
        attack = CarliniL0Method(classifier=classifier, confidence=attack_params['confidence'], max_iter=attack_params['max_iter'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)
    elif attack_method == 'carliniwagner_l2':
        attack = CarliniL2Method(classifier, confidence=attack_params['confidence'], learning_rate=attack_params['learning_rate'], binary_search_steps=attack_params['binary_search_steps'], max_iter=attack_params['max_iter'], initial_const=attack_params['initial_const'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)

    elif (attack_method=='pgd'):
        attack = ProjectedGradientDescent(estimator=classifier, eps=attack_params['eps'], max_iter=attack_params['max_iter'], batch_size=attack_params['batch_size'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)

    elif attack_method == 'jsma':
        attack = SaliencyMapMethod(classifier, theta=attack_params['theta'], gamma=attack_params['gamma'])
        x_test_adv = attack.generate(x=x_test, mask=mask_param)
        if domain_adoption:
            x_test_adv = domain_adaptation(x_test_adv, x_test)
  
    # Step 7: Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy_adversarial = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) * 100
    print("Accuracy on adversarial test examples: {}%".format(accuracy_adversarial))

    # # Step 8: save the results
    results_base_dir = "aknoon_results"

    def save_adversarial_examples(attack_method, attack_params, x_test_adv, mask, domain_adoption):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

        attack_method_dir = os.path.join(results_base_dir, attack_method)
        if not os.path.exists(attack_method_dir):
            os.makedirs(attack_method_dir)

        param_str = "_".join([f"{param}{value}" for param, value in attack_params.items()])
        param_str += f"_mask{mask}_domain_adoption{domain_adoption}"

        # Save the normalized adversarial examples to a CSV file
        adv_examples_normalized_file_path = os.path.join(attack_method_dir, f"adv_examples_normalized_{param_str}_{current_time}.csv")
        df_adv_normalized = pd.DataFrame(x_test_adv.reshape(-1, n_features), columns=dfTest.columns[1:])
        df_adv_normalized.to_csv(adv_examples_normalized_file_path, index=False)

        # Denormalize the adversarial examples
        df_adv_denormalized = pd.DataFrame(scaler.inverse_transform(df_adv_normalized), columns=dfTest.columns[1:])

        # Save the denormalized adversarial examples to a CSV file
        adv_examples_denormalized_file_path = os.path.join(attack_method_dir, f"adv_examples_denormalized_{param_str}_{current_time}.csv")
        df_adv_denormalized.to_csv(adv_examples_denormalized_file_path, index=False)        
   
    results_base_dir = "aknoon_results"
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)
    result_file_path = os.path.join(results_base_dir, 'results.csv')
    attack_name = f"{attack_method}"
    for param, value in attack_params.items():
        attack_name += f"_{param}{value}"
    attack_name += f"_mask{mask}_domain_adoption{domain_adoption}"

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
    save_adversarial_examples(attack_method, attack_params, x_test_adv, mask, domain_adoption)
    return accuracy_benign, accuracy_adversarial


# step 9 plot the attack results
def plot_attack_accuracy(attack_name, param_values, accuracies, original_accuracy, results_base_dir):
    plt.figure()
    plt.plot(param_values, accuracies, marker='o', label='Adversarial Accuracy')
    plt.scatter(0, original_accuracy, color='r', marker='X', s=100, label='Original Accuracy')
    
    if attack_name.lower() == 'fgsm':
        param_name = 'Epsilon'
    elif attack_name.lower() == 'jsma':
        param_name = 'Gamma'
    else:
        param_name = 'Attack Parameter'

    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'{attack_name} Attack: Accuracy vs {param_name}')
    plt.legend(loc='upper right')

    # Get the current time and format it as a string
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

    file_name = f"{attack_name}_accuracy_vs_{param_name.lower()}_{current_time}.png"

    # Save the plot with the timestamp in the file name
    plt.savefig(os.path.join(results_base_dir, file_name))
    plt.show()


# main
def main():
    # epsilons=[0.2]
    # epsilons = np.arange(0.0, 0.5, 0.05)
    # gammas = np.arange(0.01, 0.051, 0.01)
    gammas=[0.1]
    # confidences = np.arange(0.0, 5.1, 0.5)
    theta = 0.1
    accuracies = []
    original_accuracy = 0
    # attack_name = 'fgsm'
    # attack_name = 'pgd'
    attack_name = 'jsma'
    # attack_name = 'carliniwagner_l2'
    results_base_dir = "aknoon_results"
    # pgd_params = {'eps': 0.1, 'max_iter': 100, 'batch_size': 32}
    # attack_params = {'confidence': 0.0, 'learning_rate': 0.01, 'binary_search_steps': 10, 'max_iter': 10    , 'initial_const': 0.01}
    # for eps in epsilons:
    #     attack_params = {'eps': eps}
    for gamma in gammas:
        attack_params = {'theta': theta, 'gamma': gamma}
    # for confidence in confidences:
    #     attack_params = {'confidence': confidence,
    #                 'learning_rate': 0.01,
    #                 'binary_search_steps': 10,
    #                 'max_iter': 10,
    #                 'initial_const': 0.01}

        original_accuracy, accuracy_adversarial = perturber(attack_name.lower(), attack_params, mask=True, domain_adoption=False)
        accuracies.append(accuracy_adversarial)
    # plot_attack_accuracy(attack_name, epsilons, accuracies, original_accuracy,results_base_dir)
    plot_attack_accuracy(attack_name, gammas, accuracies, original_accuracy,results_base_dir)
    # plot_attack_accuracy(attack_name, confidences, accuracies, original_accuracy,results_base_dir)

main()



