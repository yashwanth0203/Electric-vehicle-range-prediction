#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[61]:


data=pd.read_csv('evdataset.csv')
data.head()


# In[62]:


data.nunique()


# In[63]:


data.shape


# In[64]:


data.replace({'Drive':{'Rear':2,'Front':0,'AWD':1}},inplace=True)


# In[65]:


cols_to_use=['id','Acceleration 0 - 100 km/h','Top Speed','Electric Range','Total Power','Total Torque','Drive','Battery Capacity','Charge Power','Charge Speed','Fastcharge Speed','Wheelbase','Gross Vehicle Weight (GVWR)','Max. Payload','Cargo Volume']
data=data[cols_to_use]
data.head()


# In[66]:


data.shape


# In[67]:


data.isna().sum()


# In[68]:


x=data.drop('Electric Range',axis=1)
y=data['Electric Range']


# In[ ]:





# In[69]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit_transform(x)


# In[70]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[71]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(X_train,Y_train)


# In[72]:


reg.score(X_test,Y_test)


# In[73]:


reg.score(X_train,Y_train)


# In[81]:


training_data_prediction=reg.predict(X_train)
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.title("Actual price vs Predicted price")
plt.show()


# In[74]:


from sklearn import linear_model
lasso_reg=linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
lasso_reg.fit(X_train,Y_train)


# In[75]:


lasso_reg.score(X_test,Y_test)


# In[82]:


lasso_reg.score(X_train,Y_train)


# In[83]:


training_data_prediction=lasso_reg.predict(X_train)
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.title("Actual price vs Predicted price")
plt.show()


# In[77]:


from sklearn.linear_model import Ridge
ridge_reg=linear_model.Ridge(alpha=50,max_iter=100,tol=0.1)
ridge_reg.fit(X_train,Y_train)


# In[78]:


ridge_reg.score(X_test,Y_test)


# In[79]:


ridge_reg.score(X_train,Y_train)


# In[84]:


training_data_prediction=ridge_reg.predict(X_train)
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.title("Actual price vs Predicted price")
plt.show()


# In[ ]:




