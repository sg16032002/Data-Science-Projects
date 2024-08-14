#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Project  
# 
# **GOAL: Creation of a Classification Model that can predict whether or not a person has presence of heart disease based on physical features of that person (age,sex, cholesterol, etc...)**
# 

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv("F:/TGT AXP/Machine Learning_Data Science Course/Udemy_Data/DATA/heart.csv")


# In[9]:


df.head()


# In[10]:


df['target'].unique()


# In[16]:


df.info()
# CODE HERE

# Did this to see memory usage, since it is a very small dataframe
# I am not using astype to reduce memory. Otherwise, most columns would have been typecasted from int64 to int8/int16 to save memory


# In[13]:


df.describe()


# In[15]:


df['target'].value_counts()

# dataset seems pretty much balanced, This implies, we can also use accuracy for performany measuring alongside other classification metrics


# In[ ]:





# In[18]:


sns.countplot(x='target', data=df)


# In[30]:


select_plot_data = df[['age','trestbps','chol','thalach','target']]
select_plot_data.head()


# In[31]:


sns.pairplot(select_plot_data, hue='target')


# In[35]:


plt.figure (figsize = (12,8))

sns.heatmap(df.corr(), annot=True)


# In[37]:


plt.figure (figsize = (12,8), dpi=200)

sns.heatmap(df.corr(), annot=True, cmap='viridis')

#cmap = viridis, this is a colorcode, that helps improve readability of the heatmap


# In[38]:


X = df.drop('target', axis = 1)


# In[39]:


y = df['target']


# In[41]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[44]:


scaler = StandardScaler()


# In[45]:


scaled_X_train = scaler.fit_transform(X_train)


# In[47]:


scaled_X_test = scaler.transform(X_test)


# In[50]:


from sklearn.linear_model import LogisticRegressionCV


# In[ ]:


# help(LogisticRegressionCV)


# In[51]:


log_model = LogisticRegressionCV()


# In[52]:


log_model.fit(scaled_X_train,y_train)


# In[53]:


log_model.C_


# In[54]:


log_model.get_params()


# In[55]:


log_model.coef_


# In[56]:


#Interpreting coeffecients using charts. 

coefs = pd.Series(index=X.columns,data=log_model.coef_[0])


# In[57]:


coefs


# In[61]:


coefs = coefs.sort_values()

plt.figure(figsize=(10,8))

sns.barplot(x=coefs.index,y=coefs.values)


# ---------
# 
# ## Model Performance Evaluation

# In[71]:


from sklearn.metrics import confusion_matrix,classification_report


# In[62]:


y_pred = log_model.predict(scaled_X_test)


# In[72]:


confusion_matrix(y_test,y_pred)


# In[ ]:


# CODE HERE


# **Final Task: A patient with the following features has come into the medical office:**
# 
#     age          48.0
#     sex           0.0
#     cp            2.0
#     trestbps    130.0
#     chol        275.0
#     fbs           0.0
#     restecg       1.0
#     thalach     139.0
#     exang         0.0
#     oldpeak       0.2
#     slope         2.0
#     ca            0.0
#     thal          2.0

# In[77]:


patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]


# In[79]:


log_model.predict(patient)


# In[80]:


log_model.predict_proba(patient)

