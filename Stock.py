#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Descrip: Program uses LTSM long short term memory
# to predict the closing stock price of a company
#past 60 day stock price


# In[2]:


#Import Directories
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[3]:


#get the stock quote
df = web.DataReader('LUV',data_source='yahoo',start='2012-01-01',end='2020-08-15')
#df = web.DataReader('AAPL',data_source='yahoo',start = '2012-01-01', end='2019-12-17')
#show data
print(df)


# In[4]:


#get number of rows and colums in data set
df.shape


# In[5]:


#Visualize he closing price history
plt.figure(figsize=(16,8))
plt.title('Closed Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($), fontsize = 18')
plt.show()
max_price = max(df['Close'])


# In[6]:


#Create a new data frame with only close column
data = df.filter(['Close'])
#convert data frame to Nympy array
dataset = data.values 
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)

training_data_len


# In[7]:


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[8]:


#Create The training Data Set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

#Split the data into x_train and y_train
x_train = [] #Independent
y_train = [] #Dependent Target Vars

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0]) #will contain 60 vals - to 59
    y_train.append(train_data[i,0]) #61th val at position 60
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[9]:


#convert the x_train and y_train to numpy arrrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[10]:


#Reshape the x_train data: LSTM expects 3d data #~samples #~timesteps #~features
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
x_train


# In[11]:


#Build the LTSM model 
model = Sequential()
model.add(LSTM(32, return_sequences=True,input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50,return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


# In[12]:


#compil the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[23]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[24]:


#Test Dataset
#create new array containing scaled values from index 1676 to 203
test_data = scaled_data[training_data_len - 60:, :]
#create data sets x_tests and y_test
x_test = []
y_test = dataset[training_data_len:, : ] #vals we want the model to predict

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[25]:


#Convert Data to numpy array
x_test = np.array(x_test)


# In[26]:


x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[27]:


#Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[28]:


#Get the root mean squarred error (RMSE) (response lower the better)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[29]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the Model
plt.figure(figsize=(16,8))
plt.title('South West Airlines LUV')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Closed Price USD ($) ', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()


# In[30]:


#Get the quote
lUV_quote = web.DataReader('LUV',data_source='yahoo',start='2012-01-01',end='2020-08-15')
#Create a new dataframe
new_df = lUV_quote.filter(['Close'])

#Get the last 60 day closing vals and convert
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#create an empty list
X_test = []
#append the past 60 days
X_test.append(last_60_days_scaled)
#convert the X_test data set to numpy array
X_test = np.array(X_test)
#Reshape the Data 
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
#Get the predicted scaled price
prid_price = model.predict(X_test)
#undo scale
pred_price = scaler.inverse_transform(prid_price)
print(pred_price-1)


# In[31]:


lUV_quote2 = web.DataReader('LUV',data_source='yahoo',start='2020-08-14',end='2020-08-14')
print(lUV_quote2)

