#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd#read in data using pandas
df = pd.read_csv('data_new.csv')#check data has been read in properly
df.head()


# In[14]:


#create a dataframe with all training data except the target column
X = df.drop(columns=['Weld_Quality'])

#check that the target variable has been removed
X.head()


# In[60]:


y = df[['Weld_Quality']]
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=.35,random_state=8)


# In[102]:


from keras.models import Sequential
from keras.layers import Dense, Dropout #create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(6, activation='relu', input_shape=(n_cols,)))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))


# In[103]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[104]:


from keras.callbacks import EarlyStopping #set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)#train model
model.fit(train_X, train_y, validation_split=0.4, epochs=25, callbacks=[early_stopping_monitor])


# In[95]:


test_y_predictions = model.predict(test_X)


# In[69]:


test_y_predictions


# In[105]:


accuracy = model.evaluate(test_X, test_y)
# Print accuracy
print('Accuracy:', accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




