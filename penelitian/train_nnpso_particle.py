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
parser.add_argument("num_particles", type=int, help="Jumlah partikel")
parser.add_argument("w", type=float, help="Nilai w")

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

num_particles = args.num_particles
num_iterations = 30
w = args.w

bpnn = Backpropagation(input_size, hidden_sizes,activation_functions, output_size, learning_rate)
start_time = time.time()
bpnn.train(x_train, y_train, num_iterations)
bpnn_computing_time = time.time() - start_time
bpnn_error_track = bpnn.get_error_track()

# # Predict and calculate error
bpnn_y_pred = bpnn.predict(x_test)
bpnn_error = bpnn.rmse(bpnn_y_pred, y_test)

nnpso = NNPSO(input_size, hidden_sizes, activation_functions, output_size, learning_rate, num_particles, num_iterations, w)
nnpso.initialize()
start_time = time.time()
nnpso.optimize(x_train, y_train)
nnpso_computing_time = time.time() - start_time
print(nnpso_computing_time)
nnpso_error_track = nnpso.get_error_track()

# Predict and calculate error
nnpso_y_pred = nnpso.predict(x_test)
nnpso_error = nnpso.rmse(nnpso_y_pred, y_test)

improve_error = bpnn_error - nnpso_error
improve_error_percent = (improve_error/bpnn_error)*100

improve_time = bpnn_computing_time - nnpso_computing_time
improve_time_percent = (improve_time/bpnn_computing_time)*100

print('eror bpnn :', bpnn_error)
print('eror nnpso :', nnpso_error)

# connect mysql database=nnpso table hasil
import mysql.connector
from mysql.connector import Error


connection = mysql.connector.connect(host='localhost',
                                             database='nnpso',
                                             user='root',
                                             password='')   
mySql_insert_query = """INSERT INTO best_particle (num_particles, w, bpnn_error, nnpso_error, bpnn_time, nnpso_time, improve_error, improve_error_percent, improve_time, improve_time_percent, learning_rate)
                         VALUES (%s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s) """

recordTuple = (num_particles, w, bpnn_error, nnpso_error, bpnn_computing_time, nnpso_computing_time, improve_error, improve_error_percent, improve_time, improve_time_percent, learning_rate)
cursor = connection.cursor()
cursor.execute(mySql_insert_query, recordTuple)
connection.commit()
print("Record inserted successfully into nnpso table")
cursor.close()


