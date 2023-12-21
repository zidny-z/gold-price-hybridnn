import numpy as np
import pickle

# model import
model_nnpso = pickle.load(open('models/nnpso.pkl','rb'))
model_bpnn = pickle.load(open('models/bpnn.pkl','rb'))
nnpso_error = model_nnpso.get_error_track()
bpnn_error = model_bpnn.get_error_track()

# load data x_test dan Y test from models/x_test.csv and models/y_test.csv
x_test = np.loadtxt('models/x_test.csv', delimiter=',')
y_test = np.loadtxt('models/y_test.csv', delimiter=',')
Y_pred_nnpso = model_nnpso.predict(x_test).flatten()
Y_pred_bpnn = model_bpnn.predict(x_test).flatten()

# merge y test and y pred to csv
y_test = np.reshape(y_test, (-1,1))
Y_pred_nnpso = np.reshape(Y_pred_nnpso, (-1,1))
Y_pred_bpnn = np.reshape(Y_pred_bpnn, (-1,1))
selisih_nnpso = np.subtract(y_test, Y_pred_nnpso)
selisih_bpnn = np.subtract(y_test, Y_pred_bpnn)
y_pred = np.concatenate((y_test, Y_pred_nnpso, Y_pred_bpnn, selisih_nnpso, selisih_bpnn), axis=1)
np.savetxt('models/Y_comparation.txt', y_pred, delimiter=';', fmt='%.6f')

# model error track
rmse_nnpso = model_nnpso.rmse(Y_pred_nnpso, y_test)
rmse_bpnn = model_bpnn.rmse(Y_pred_bpnn, y_test)
# export to csv
nnpso_error = np.reshape(nnpso_error, (-1,1))
bpnn_error = np.reshape(bpnn_error, (-1,1))
error_track = np.concatenate((nnpso_error, bpnn_error), axis=1)
np.savetxt('models/error_track.txt', error_track, delimiter=';', fmt='%.6f')