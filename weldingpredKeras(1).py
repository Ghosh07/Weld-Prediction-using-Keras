#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd#read in data using pandas
df = pd.read_csv('data_new.csv')#check data has been read in properly
df.head()


# In[2]:


#create a dataframe with all training data except the target column
X = df.drop(columns=['Weld_Quality'])

#check that the target variable has been removed
X.head()
y = df[['Weld_Quality']]


# In[3]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)


# In[12]:



from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,encoded_y,test_size=.25,random_state=6)


# In[13]:


from keras.models import Sequential
from keras.layers import Dense, Dropout #create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(400, activation='relu', input_shape=(n_cols,)))
model.add(Dense(300, activation='relu'))

model.add(Dense(200, activation='softmax'))
model.add(Dense(1))


# In[14]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# In[15]:


from keras.callbacks import EarlyStopping #set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)#train model
history= model.fit(train_X, train_y, validation_split=0.2, epochs=25, callbacks=[early_stopping_monitor])


# In[8]:


test_y_predictions = model.predict(test_X)


# In[9]:


test_y_predictions


# In[16]:


accuracy = model.evaluate(test_X, test_y)
# Print accuracy
print('Accuracy:', accuracy)


# In[17]:



import matplotlib.pyplot as plt
# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


# In[ ]:





# In[ ]:





# In[ ]:




