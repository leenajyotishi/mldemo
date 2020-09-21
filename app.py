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
df1 = pd.DataFrame(data1)

gapminder_tidy = df1.melt(id_vars=["Product"], 
                              var_name="year", 
                              value_name="Amount")
gapminder_tidy.head(n=15)
#Converting words to integer values
df = gapminder_tidy.rename(columns={'year': 'ds', 'Amount':'y'})

from fbprophet import Prophet
grouped = df.groupby('Product')
for g in grouped.groups:
    group = grouped.get_group(g)
    m = Prophet()
    m.fit(group)
    future = m.make_future_dataframe(periods=360)
    forecast = m.predict(future)
    print(forecast.tail())
    
final = pd.DataFrame()
for g in grouped.groups:
    group = grouped.get_group(g)
    m = Prophet()
    m.fit(group)
    future = m.make_future_dataframe(periods=12, freq='M')
    forecast = m.predict(future)    
    forecast = forecast.rename(columns={'yhat': g})
    final = pd.merge(final, forecast.set_index('ds'), how='outer', left_index=True, right_index=True)

final = final[[ g for g in grouped.groups.keys()]]
print (g for g in grouped.groups.keys())
final


    return render_template('index.html', prediction_text='Product price hould be $ {}'.format(final))


if __name__ == "__main__":
    app.run(debug=True)