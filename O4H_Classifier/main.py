import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import compute_class_weight
from keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
import csv


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset from CSV file
df = pd.read_csv('dataset.csv')
# df = df.rename(columns={0: 'timestamp'})
# dropping the first colomn - Timestaps colomn
# df = df.drop(df.columns[0],axis=1)

# Normalize the dataset
# select all columns except the first one
cols_to_normalize = df.columns[2:]
scaler = MinMaxScaler(feature_range=(0, 1))
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

#spliting the last week of data as a seprate section for test
l=[266417,383714]
l_mod = [0] + l + [max(l)+1]
print(l_mod)
list_of_dfs = [df.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]

dfTrain = list_of_dfs[0]
# dfTrain= dfTrain.rename(columns={'activity':'0'})

dfTest = list_of_dfs[1]
# dfTest= dfTest.rename(columns={'activity':'0'})

# dfTrain.to_csv('O4H_TRAIN', header=False, index=False)
# dfTest.to_csv('O4H_TEST', header=False, index=False)

# Split the dataset into input (X) and output (y)
X = dfTrain.iloc[:, 2:].values # features are in columns 2 to 197
y = dfTrain.iloc[:, 1].values # activity label is in column 1


# # attemp _for_activity_class_balancing
# # Calculate class weights
# class_weights = compute_class_weight('balanced', classes=np.unique(y),y=y)
# # Convert the class weights to a dictionary
# class_weights_dict = dict(enumerate(class_weights))
# print("##################################")
# print(class_weights)

# Define the LSTM model
n_timesteps, n_features, n_outputs = 1, X.shape[1], len(np.unique(y))
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
model.add(Dense(n_outputs, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])


# Perform 5-fold Time series validation
tscv = TimeSeriesSplit()
print(tscv)
TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Fold {fold}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    # kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # for fold, (train_index, test_index) in enumerate(kf.split(X)):
    #     X_train, y_train = X[train_index], y[train_index]
    #     X_test, y_test = X[test_index], y[test_index]

    # Reshape the input data for LSTM model
    X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # Convert the output labels to one-hot encoding
    y_train = np.eye(n_outputs)[y_train]
    y_test = np.eye(n_outputs)[y_test]

    # model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weights_dict, validation_data=(X_test, y_test))
    
    # Train the LSTM model on this fold
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the trained model on the test set
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print(f'Fold {fold+1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}- F1-Score: {f1_score:.4f}- Precision: {precision:.4f}- Recall: {recall:.4f}')
    model.save('TSC_model3.hdf5')

print("*********************************************************************")
#load seprate test section
# Split the dataset into input (X) and output (y)
XX = dfTest.iloc[:, 2:].values # features are in columns 2 to 197
YY = dfTest.iloc[:, 1].values # activity label is in column 1
# Reshape the input data for LSTM model
XX = XX.reshape((XX.shape[0], n_timesteps, n_features))
# Convert the output labels to one-hot encoding
YY = np.eye(n_outputs)[YY]
# Evaluate the trained model on the test set
loss, accuracy, f1_score, precision, recall = model.evaluate(XX,YY, verbose=0)
print("Test last week of data")
print(f'Loss: {loss:.4f} - Accuracy: {accuracy:.4f}- F1-Score: {f1_score:.4f}- Precision: {precision:.4f}- Recall: {recall:.4f}')

