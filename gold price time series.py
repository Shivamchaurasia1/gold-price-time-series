#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys


# In[7]:


data=pd.read_csv("gold_price_data.csv")
data


# In[8]:


data["Date"]=pd.to_datetime(data["Date"])
data=data.set_index("Date")
data


# In[9]:


plt.style.use('ggplot')
plt.figure(figsize=(18,8)) 
plt.grid(True) 
plt.xlabel('Date', fontsize = 20) 
plt.xticks(fontsize = 15)
plt.ylabel('Value', fontsize = 20)
plt.yticks(fontsize = 15) 
plt.plot(data['Value'], linewidth = 3, color = 'blue')
plt.title('gold price data', fontsize = 30)
plt.show()


# In[11]:


rolmean=data["Value"].rolling(100).mean()
rolstd=data["Value"].rolling(100).std()


# In[12]:


plt.plot(data.Value)
plt.plot(rolmean)
plt.plot(rolstd)


# In[13]:


from statsmodels.tsa.stattools import adfuller
adft=adfuller(data["Value"])


# In[14]:


pd.Series(adft[0:4],index=["test stats","p-value","lag","data points"])


# In[15]:


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(48).mean() # rolling mean
    rolstd = timeseries.rolling(48).std() # rolling standard deviation
    # Plot rolling statistics:
    plt.figure(figsize = (18,8))
    plt.grid('both')
    plt.plot(timeseries, color='blue',label='Original', linewidth = 3)
    plt.plot(rolmean, color='red', label='Rolling Mean',linewidth = 3)
    plt.plot(rolstd, color='black', label = 'Rolling Std',linewidth = 4)
    plt.legend(loc='best', fontsize = 20, shadow=True,facecolor='lightpink',edgecolor = 'k')
    plt.title('Rolling Mean and Standard Deviation', fontsize = 25)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)


# In[16]:


test_stationarity(data.Value)


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(data[["Value"]],period=12)


# In[18]:


result.seasonal


# In[19]:


fig=plt.figure(figsize=(20,10))
fig=result.plot()
fig.set_size_inches(17,10)


# In[20]:


#converting non statinary into stationary(differencing, log)

df=data["Value"].diff(periods=5)

data_log=np.log(data["Value"])
data_sqrt=np.sqrt(data["Value"])
df=data["Value"].dropna()
test_stationarity(df)
test_stationarity(data_log)
test_stationarity(data_sqrt)

rolling_mean = data.rolling(window=3).mean()  # Adjust window size as needed
data_ma = data - rolling_mean
data_ma=data_ma.dropna()
test_stationarity(data_ma)


# In[21]:


#split dAta into training and test( ARIMA model takes care of the non stationary data)
train=data["Value"][0:-60]
test=data["Value"][-60: ]
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[22]:


history=[x for x in train]


# In[23]:


model=ARIMA(history,order=(1,1,1))
model=model.fit()


# In[24]:


model.summary()


# In[25]:


def train_arima_model(X, y, arima_order):
    # prepare training dataset
    # make predictions list
    history = [x for x in X]
    predictions = list()
    for t in range(len(y)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(y[t])
    # calculate out of sample error
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return rmse


# In[26]:


history = [x for x in train]
predictions = list()
conf_list = list()
for t in range(len(test)):
    model = ARIMA(history,order=(2,1,0))
    model_fit = model.fit()
    fc = model_fit.forecast(alpha = 0.05)
    predictions.append(fc)
    history.append(test[t])
print('RMSE of ARIMA Model:', np.sqrt(mean_squared_error(test, predictions)))


# In[27]:


plt.figure(figsize=(18,8))
plt.grid(True)
plt.plot(range(len(test)),test, label = 'True Test Close Value', linewidth = 5)
plt.plot(range(len(predictions)), predictions, label = 'Predictions on test data', linewidth = 5)
plt.xticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.legend(fontsize = 20, shadow=True,facecolor='lightpink',edgecolor = 'k')
plt.show()


# In[ ]:




