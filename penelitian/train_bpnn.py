import argparse

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from hybridnn import Backpropagation, NNPSO, Sigmoid, Tanh, MinMaxScaler, DataSplitter 

# input user from command
parser = argparse.ArgumentParser()
parser.add_argument("hidden_layer", type=int, help="Jumlah hidden layer")
parser.add_argument("learning_rate", type=float, help="Learning rate")
args = parser.parse_args()


# Load data
data = pd.read_csv('datasets/datanew.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)


# normalization with min max scaler
X_normalization = MinMaxScaler()
X = X_normalization.fit_transform(x)

Y_normalization = MinMaxScaler()
Y = Y_normalization.fit_transform(y)

# Split data
splitter = DataSplitter()
x_train, x_test, y_train, y_test = splitter.split_data(X, Y, test_size=0.20, random_state=12)

# # model parameters
input_size = x_train.shape[1]
hidden_layer = args.hidden_layer
hidden_sizes = [hidden_layer]
activation_functions = [Tanh(), Sigmoid()]
output_size = y_train.shape[1]
learning_rate = args.learning_rate
num_iterations = 30


bpnn = Backpropagation(input_size, hidden_sizes,activation_functions, output_size, learning_rate)
start_time = time.time()
bpnn.train(x_train, y_train, num_iterations)
bpnn_computing_time = time.time() - start_time
bpnn_error_track = bpnn.get_error_track()

# # Predict and calculate error
bpnn_y_pred = bpnn.predict(x_test)
bpnn_error = bpnn.rmse(bpnn_y_pred, y_test)

print('BPNN Error: ', bpnn_error)

# connect mysql database=bpnn_test
import mysql.connector
from mysql.connector import Error


connection = mysql.connector.connect(host='localhost',
                                             database='nnpso',
                                             user='root',
                                             password='')
mySql_insert_query = """INSERT INTO best_bpnn (hidden, learning_rate, rmse, computing_time) 
                         VALUES (%s, %s, %s, %s) """

recordTuple = (hidden_layer, learning_rate, bpnn_error, bpnn_computing_time)
cursor = connection.cursor()
cursor.execute(mySql_insert_query, recordTuple)
connection.commit()
print("Record inserted successfully into bpnn table")
cursor.close()
