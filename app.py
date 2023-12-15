from flask import Flask, request, render_template
import pickle

# model import
model = pickle.load(open('models/nnpso.pkl','rb'))
X_normalization = pickle.load(open('models/x_normalization.pkl','rb'))
Y_normalization = pickle.load(open('models/y_normalization.pkl','rb'))

app = Flask(__name__, template_folder='templates')
app.config['STATIC_FOLDER'] = 'static'

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        bulan = float(request.form['bulan'])
        tahun = float(request.form['tahun'])
        minyak = float(request.form['minyak'])
        dollar = float(request.form['dollar'])
        inputan = [bulan, tahun, minyak, dollar]
        print(inputan)
        inputan = X_normalization.transform(inputan)
        output = model.predict(inputan)
        output = Y_normalization.inverse_transform(output)
        output = format(output[0], '.2f')
        return render_template('index.html', output=output)

if __name__ == "__main__":
    app.run(debug=True)