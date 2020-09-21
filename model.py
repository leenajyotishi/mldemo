# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Import Libraries

import seaborn as sns
import datetime as dt
from fbprophet import Prophet
# Statsmodels widely known for forecasting than Prophet
import statsmodels.api as sm
from scipy import stats
from plotly import tools
import plotly.figure_factory as ff
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")

# plt.style.available
plt.style.use("seaborn-whitegrid")

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


# Saving model to disk
pickle.dump(final, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))