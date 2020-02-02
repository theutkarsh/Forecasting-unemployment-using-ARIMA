
# coding: utf-8

# In[69]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pandas as pd
from datetime import datetime 


# In[70]:


import statsmodels.api as sm


# In[4]:


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[7]:


df = pd.read_csv("Downloads/data/Data_distribution.csv")


# In[58]:


df.head(10)


# In[100]:


df['Year']=df['Year'].apply(lambda x: pd.to_datetime(x,format ="%Y"))


# In[101]:


data = df.set_index('Year')


# In[102]:


data.index


# In[103]:


#data.index.astype("datetime")


# In[106]:


data.plot(figsize=(15, 6))
plt.show()


# In[107]:


data.isna().sum()


# In[108]:


data.shape


# In[109]:


data.columns


# In[110]:


grad=['Graduates - Medical*',
       'Graduates - Agriculture*', 'Graduates  - Veterinary*',
       'Graduates - Science*', 'Graduates - Engineering*', 'Graduates - Total']


# In[111]:


graduates=data[grad]


# In[112]:


graduates


# In[113]:


graduates.plot(figsize=(15,10))
plt.show()


# In[121]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(graduates['Graduates - Medical*'], model='additive')
fig = decomposition.plot()
plt.show()


# In[120]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(graduates, model='additive')
fig = decomposition.plot()
plt.show()


# In[122]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[130]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(graduates['Graduates - Medical*'],order=param,seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            #print("ds")
            results = mod.fit()
            #print("ds")
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# # The above output suggests that SARIMAX ARIMA(1, 1, 0)x(1, 1, 0, 12)12  yields the lowest AIC value of 185.27. Therefore we should consider this to be optimal option.

# In[170]:


y=graduates['Graduates - Medical*']


# In[171]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 0),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[172]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[196]:


pred = results.get_prediction(start=pd.to_datetime('2005-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['1971':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))


# In[198]:


pred_ci

import pandas.tseries.converter as converter

c = converter.DatetimeConverter()

type(c.convert(y.index.values, None, None))

# In[202]:


y_forecasted = pred.predicted_mean
y_truth = y['2005-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[203]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

n statistics, the mean squared error (MSE) of an estimator measures the average of the squares of the errors — that is, the average squared difference between the estimated values and what is estimated. The MSE is a measure of the quality of an estimator — it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit.
Root Mean Square Error (RMSE) tells us that our model was able to forecast yearly graduates in the test set within 34.14 of the real sales. Our graduates yearly ranges from around 40000 to over 80000. In my opinion, this is a pretty good model so far.
# In[217]:


pred_uc = results.get_forecast(steps=25)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Graduates - Medical*')
plt.legend()
plt.show()


# In[213]:


final['Graduates - Medical*']=pred_ci


# In[305]:


def arima(field):
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(graduates[field],order=param,seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                #print("ds")
                results = mod.fit()
                #print("ds")
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    y=graduates[field]

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 0),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])


    pred_uc = results.get_forecast(steps=25)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Year')
    ax.set_ylabel('Graduates '+field)
    plt.legend()
    plt.show()
    y_forecasted = pred_uc.predicted_mean
    dfa=pd.DataFrame({'Year':y_forecasted.index, 'Predicited_'+field :y_forecasted.values})
    dfa['Year']= dfa['Year'].dt.year
    dfa.set_index('Year',inplace= True)
    return dfa


# In[306]:


colum=graduates.columns


# In[307]:


colum


# In[308]:


ds=arima("Graduates - Agriculture*")


# In[309]:


ds


# In[310]:


outcome=ds
del(outcome['Predicited_Graduates - Agriculture*'])
for c in colum:
    cm="'"+c+"'"
    print(cm)
    outcome = pd.merge(outcome, arima(c), left_index=True, right_index=True)


# In[311]:


outcome.head(30)


# In[312]:


outcome.to_csv("Downloads/data/Results.csv")

