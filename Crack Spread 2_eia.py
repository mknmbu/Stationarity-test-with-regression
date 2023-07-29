#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import math
from datetime import datetime

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


# In[2]:


##importing data file
## Importing data file
df = pd.read_csv('Crack_eia.csv',index_col='Date', parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()


# In[3]:


print (df.info())


# In[4]:


# create new variables
df[ 'lcrack'] = np.log(df['crack'])
df['lwti' ] = np.log(df['wti'])
df['lgas' ] = np.log(df['gasoline'])
df['lheat' ] = np.log(df['heatoil'])


# In[5]:


#
# non-stationary - use first difference
df['dcrack'] = df.crack.diff(1)
df['dlcrack'] = df.lcrack.diff(1)
df['dwti'] = df.wti.diff(1)
df['dlwti'] = df.lwti.diff(1)


# In[6]:


# create lagged variables
df['lcrack_L1'] = df['lcrack'].shift(1)
df['lcrack_L2'] = df['lcrack'].shift(2)
df['lcrack_L3'] = df['lcrack'].shift(3)
df['lcrack_L4'] = df['lcrack'].shift(4)


# In[7]:


# Regression wti oil, crack spread and it's lag variables
mod1 = smf.ols('lwti ~ lgas + lheat + lcrack + lcrack_L1 + lcrack_L2 + lcrack_L3 + lcrack_L4',data=df).fit()
print(mod1.summary())
print()


# In[8]:


# Regression WTI crude oil, crack spread and it's lag variables
mod2 = smf.ols('lwti ~ lgas + lheat + lcrack + lcrack_L1 + lcrack_L2 + lcrack_L3 + lcrack_L4',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=4)
print(mod2.summary())
print()


# In[20]:


# Regression WTI crude oil, crack spread and it's lag variables
mod3 = smf.ols('lcrack ~ lwti + lgas + lheat + lcrack_L1 + lcrack_L2 + lcrack_L3 + lcrack_L4',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=4)
print(mod3.summary())
print()


# In[9]:


from statsmodels.stats.stattools import durbin_watson

## perform Durbin-Watson test for serial correlation
durbin_watson(mod2.resid)


# In[10]:



# In[ ]: The test statistic is 2.13. Since this is within the range of 1.5 and 2.5, we would consider autocorrelation not to 
# be problematic in this regression model.


# H0 (null hypothesis): There is no correlation among the residuals. HA (alternative hypothesis): residuals are autocorrelated.
# A test statistic of 2.04 indicates there is no serial correlation.


# In[11]:


from statsmodels.tsa.stattools import adfuller


# In[16]:


# Stationary test of WTI crude oil
print("Observations of Dickey-fuller test for WTI crude oil")
x = df["wti"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[17]:


# Stationary test of WTI crude oil after necessary modification to make it stationary
print("Observations of Dickey-fuller test for WTI crude oil after modificatioin")
x = df["lwti"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[18]:


# Stationary test of crack spread
print("Observations of Dickey-fuller test for crack spread")
x = df["crack"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[19]:


# Stationary test of crack spread after necessary modification to make it stationary
print("Observations of Dickey-fuller test for crack spread after modification")
x = df["lcrack"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[ ]:




