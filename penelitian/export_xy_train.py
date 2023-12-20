import numpy as np
import pandas as pd

from hybridnn import MinMaxScaler, DataSplitter 

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
# export xtest ytest
np.savetxt("models/x_test.csv", x_test, delimiter=",")
np.savetxt("models/y_test.csv", y_test, delimiter=",")
