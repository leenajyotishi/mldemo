import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    

data1 = {'Product':['P1', 'P2'],'2017-01-01':['12','92'],'2017-02-01':['13','99'],'2017-03-01':['15','98'],
       '2017-04-01':['12','95']}

prediction = model.predict([[np.array(data1)]])
    return render_template('index.html', prediction_text='Product price hould be $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)