from flask import Flask, request, render_template
from bokeh.embed import components 
from bokeh.plotting import figure 
from bokeh.models import Slider
import numpy as np
import pickle

# model import
model_nnpso = pickle.load(open('models/nnpso.pkl','rb'))
model_bpnn = pickle.load(open('models/bpnn.pkl','rb'))
X_normalization = pickle.load(open('models/x_normalization.pkl','rb'))
Y_normalization = pickle.load(open('models/y_normalization.pkl','rb'))
nnpso_error = model_nnpso.get_error_track()
bpnn_error = model_bpnn.get_error_track()

# load data x_test dan Y test from models/x_test.csv and models/y_test.csv
x_test = np.loadtxt('models/x_test.csv', delimiter=',')
y_test = np.loadtxt('models/y_test.csv', delimiter=',')
Y_pred_nnpso = model_nnpso.predict(x_test).flatten()
Y_pred_bpnn = model_bpnn.predict(x_test).flatten()
rmse_nnpso = model_nnpso.rmse(Y_pred_nnpso, y_test)
rmse_bpnn = model_bpnn.rmse(Y_pred_bpnn, y_test)

def visualize_errors(nnpso, bpnn):
    plot = figure(title='Grafik Penurunan Eror Algoritma', x_axis_label='Epoch / Iterasi', y_axis_label='Eror', plot_width=550, plot_height=400)
    plot.line(range(1, len(nnpso)+1), nnpso, legend_label='NNPSO', line_width=2, line_color='green')
    plot.line(range(1, len(bpnn)+1), bpnn, legend_label='BPNN', line_width=2, line_color='orange')
    script, div = components(plot)
    return script, div

def visualize_predictions(nnpso, bpnn, y_test):
    plot = figure(title='Grafik Perbandingan Y Prediksi dan Y Aktual', x_axis_label='Data ke-', y_axis_label='Y', plot_width=550, plot_height=400)
    plot.line(range(1, len(nnpso)+1), nnpso, legend_label='NNPSO', line_width=2, line_color='green')
    plot.line(range(1, len(bpnn)+1), bpnn, legend_label='BPNN', line_width=2, line_color='orange')
    plot.line(range(1, len(y_test)+1), y_test, legend_label='Aktual', line_width=2, line_color='dodgerblue')
    script, div = components(plot)
    return script, div

app = Flask(__name__, template_folder='templates')
app.config['STATIC_FOLDER'] = 'static'

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        script, div = visualize_errors(nnpso_error, bpnn_error)
        script2, div2 = visualize_predictions(Y_pred_nnpso, Y_pred_bpnn, y_test)
        return render_template('index.html', 
            script=[script,script2], 
            div=[div,div2], 
            rmse=[rmse_nnpso, rmse_bpnn],
            y_pred=[Y_pred_nnpso, Y_pred_bpnn, y_test]
        )

    if request.method == 'POST':
        script, div = visualize_errors(nnpso_error, bpnn_error)
        script2, div2 = visualize_predictions(Y_pred_nnpso, Y_pred_bpnn, y_test)

        bulan = float(request.form['bulan'])
        tahun = float(request.form['tahun'])
        minyak = float(request.form['minyak'])
        dollar = float(request.form['dollar'])
        inputan = [bulan, tahun, minyak, dollar]
        inputan = X_normalization.transform(inputan)
        output = model_nnpso.predict(inputan)
        output = Y_normalization.inverse_transform(output)
        output = format(output[0], '.2f')

        return render_template('index.html', 
            output=output, 
            script=[script,script2],
            div=[div,div2],
            rmse=[rmse_nnpso, rmse_bpnn],
            y_pred=[Y_pred_nnpso, Y_pred_bpnn, y_test]
        )

if __name__ == "__main__":
    app.run(debug=True)