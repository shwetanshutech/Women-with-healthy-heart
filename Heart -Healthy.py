#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[4]:


heart= pd.read_csv('./heart.csv')
print(heart.shape)
heart.head()


# In[5]:


heart= heart.fillna(0)
heart.columns
y = heart['target'].values
heart = heart.fillna(0)
X = heart.drop(columns=['target'], axis=1).values


# In[6]:


#Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
model = GradientBoostingClassifier(random_state=39, n_estimators=50)
model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[7]:


#Model with RandomForest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
model = RandomForestClassifier(random_state=39, n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[10]:


# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




