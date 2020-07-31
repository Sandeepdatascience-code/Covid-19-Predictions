#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary libraries.
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.externals import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#Reading the data from the csv

data = pd.read_csv('covid_data_production.csv')
data = data.drop('id',axis=1)
data = data.fillna(np.nan,axis=0)


# In[3]:


#Fetching the encoder object from the development model
filename='ModelEncoding_object.sav'
EncodeModel = joblib.load(filename)


# In[4]:


#Encoding the categorical data.
data['location'] = EncodeModel.fit_transform(data['location'].astype(str))
data['country'] = EncodeModel.fit_transform(data['country'].astype(str))
data['gender'] = EncodeModel.fit_transform(data['gender'].astype(str))
data[['symptom1']] = EncodeModel.fit_transform(data['symptom1'].astype(str))
data[['symptom2']] = EncodeModel.fit_transform(data['symptom2'].astype(str))
data[['symptom3']] = EncodeModel.fit_transform(data['symptom3'].astype(str))
data[['symptom4']] = EncodeModel.fit_transform(data['symptom4'].astype(str))
data[['symptom5']] = EncodeModel.fit_transform(data['symptom5'].astype(str))
data[['symptom6']] = EncodeModel.fit_transform(data['symptom6'].astype(str))


# In[5]:


#Get the no of days taken from symptoms observed to the hospital visit

data['sym_on'] = pd.to_datetime(data['sym_on'])
data['hosp_vis'] = pd.to_datetime(data['hosp_vis'])
data['sym_on']= data['sym_on'].map(dt.datetime.toordinal)
data['hosp_vis']= data['hosp_vis'].map(dt.datetime.toordinal)
data['diff_sym_hos']= data['hosp_vis'] - data['sym_on']


# In[6]:


data = data.drop(['sym_on','hosp_vis'],axis=1)


# In[7]:


#Isnull check
data['age'] = data['age'].fillna(value=data['age'].mean())
for column in ['from_wuhan','vis_wuhan']:
    data[column].fillna(data[column].mode()[0], inplace=True)


# In[14]:


data.isna().sum()


# In[8]:


#Fetching the scaling object from the development model
filename='ModelScaling_object.sav'
ScalingModel = joblib.load(filename)


# In[9]:


X = data[['location','country','gender','age','vis_wuhan','from_wuhan','symptom1','symptom2','symptom3','symptom4','symptom5','symptom6','diff_sym_hos']]


# In[10]:


#Scaling the input data
scaled_X = ScalingModel.fit(X).transform(X)


# In[11]:


#Fetching the model object from the development model
filename='ModelClassification_object.sav'
Model = joblib.load(filename)


# In[26]:


Y_Predict = Model.predict(scaled_X)


# In[46]:


Y_Predict


# In[43]:


Z =pd.DataFrame(Y_Predict)
Z['Mortality'] = Z


# In[44]:


Z['Mortality']


# In[39]:


actualdata = pd.read_csv('covid_data_production.csv')


# In[40]:


FinalResult = pd.concat([actualdata,Z[['Mortality']]],axis=1)


# In[41]:


FinalResult


# In[42]:


FinalResult.to_csv('Output.csv')


# In[ ]:




